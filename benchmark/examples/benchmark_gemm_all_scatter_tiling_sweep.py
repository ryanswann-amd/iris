#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
Tiling-parameter sweep for GEMM+AllScatter on AMD GPUs.

Sweeps over (BLK_M, BLK_N), BLK_K, gsize_m, and num_stages for
a representative set of M values, then generates TFLOPS charts.

Usage
-----
    # Run the full sweep (8 GPUs)
    python benchmark/examples/benchmark_gemm_all_scatter_tiling_sweep.py \
        --num_ranks 8 --output_dir /tmp/sweep_results

    # Skip benchmarking and only regenerate the chart from existing results
    python benchmark/examples/benchmark_gemm_all_scatter_tiling_sweep.py \
        --chart_only --output_dir /tmp/sweep_results
"""

import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import importlib.util

import iris

# ---------------------------------------------------------------------------
# Load the kernel from examples/23_gemm_all_scatter_tracing/
# ---------------------------------------------------------------------------
current_dir = Path(__file__).parent
kernel_path = (current_dir / "../../examples/23_gemm_all_scatter_tracing/gemm_all_scatter.py").resolve()
wrapper_path = (current_dir / "../../examples/23_gemm_all_scatter_tracing/matmul_wrapper.py").resolve()

kernel_spec = importlib.util.spec_from_file_location("gemm_all_scatter", kernel_path)
kernel_module = importlib.util.module_from_spec(kernel_spec)
sys.modules["gemm_all_scatter"] = kernel_module
kernel_spec.loader.exec_module(kernel_module)

wrapper_spec = importlib.util.spec_from_file_location("matmul_wrapper", wrapper_path)
wrapper_module = importlib.util.module_from_spec(wrapper_spec)
wrapper_spec.loader.exec_module(wrapper_module)
matmul = wrapper_module.matmul

torch.manual_seed(123)
random.seed(123)

# ---------------------------------------------------------------------------
# Sweep configuration space
# ---------------------------------------------------------------------------
M_VALUES = [128, 256, 512, 1024]
N, K = 4096, 14336
DATATYPE = "fp16"

TILE_CONFIGS = [
    # (BLK_M, BLK_N, BLK_K)  — common fp16 choices for gfx942
    (64, 64, 64),
    (128, 64, 64),
    (128, 128, 64),
    (256, 64, 64),
    (256, 128, 64),
]

GSIZE_M_VALUES = [4, 6, 8]
NUM_STAGES_VALUES = [1, 2]  # stages=3 exceeds MI300X 64 KB LDS limit for BLK_M≥256

# The "default" tile used when sweeping other parameters
DEFAULT_TILE = (256, 64, 64)
DEFAULT_GSIZE_M = 6
DEFAULT_NUM_STAGES = 2


def build_sweep_configs():
    """Return all (M, BLK_M, BLK_N, BLK_K, gsize_m, num_stages) tuples."""
    configs = []
    # 1. Tile sweep (fixed gsize_m & num_stages)
    for m, (blk_m, blk_n, blk_k) in itertools.product(M_VALUES, TILE_CONFIGS):
        configs.append(
            dict(
                m=m,
                n=N,
                k=K,
                BLK_M=blk_m,
                BLK_N=blk_n,
                BLK_K=blk_k,
                gsize_m=DEFAULT_GSIZE_M,
                num_stages=DEFAULT_NUM_STAGES,
                datatype=DATATYPE,
                sweep_group="tile",
            )
        )
    # 2. gsize_m sweep (fixed default tile & num_stages)
    blk_m, blk_n, blk_k = DEFAULT_TILE
    for m, gsize_m in itertools.product(M_VALUES, GSIZE_M_VALUES):
        if gsize_m == DEFAULT_GSIZE_M:
            continue  # already covered above
        configs.append(
            dict(
                m=m,
                n=N,
                k=K,
                BLK_M=blk_m,
                BLK_N=blk_n,
                BLK_K=blk_k,
                gsize_m=gsize_m,
                num_stages=DEFAULT_NUM_STAGES,
                datatype=DATATYPE,
                sweep_group="gsize_m",
            )
        )
    # 3. num_stages sweep (fixed default tile & gsize_m)
    for m, num_stages in itertools.product(M_VALUES, NUM_STAGES_VALUES):
        if num_stages == DEFAULT_NUM_STAGES:
            continue  # already covered above
        configs.append(
            dict(
                m=m,
                n=N,
                k=K,
                BLK_M=blk_m,
                BLK_N=blk_n,
                BLK_K=blk_k,
                gsize_m=DEFAULT_GSIZE_M,
                num_stages=num_stages,
                datatype=DATATYPE,
                sweep_group="num_stages",
            )
        )
    return configs


def config_to_filename(cfg, base="gemm_as_sweep"):
    """Unique JSON filename for a config."""
    return (
        f"{base}_m{cfg['m']}"
        f"_blkm{cfg['BLK_M']}_blkn{cfg['BLK_N']}_blkk{cfg['BLK_K']}"
        f"_gs{cfg['gsize_m']}_st{cfg['num_stages']}.json"
    )


# ---------------------------------------------------------------------------
# Worker (one process per GPU rank)
# ---------------------------------------------------------------------------
def worker(rank: int, world_size: int, init_url: str, configs: list, output_dir: str):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=init_url,
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{rank}"),
    )
    torch.cuda.set_device(rank)
    shmem = iris.iris(1 << 33)
    world_size = shmem.get_num_ranks()
    context_tensor = shmem.get_device_context()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    shmem.barrier()

    num_sms = torch.cuda.get_device_properties(rank).multi_processor_count
    gemm_stream = torch.cuda.Stream()

    for cfg in configs:
        M = cfg["m"]
        N_cfg = cfg["n"]
        K_cfg = cfg["k"]
        datatype = dtype_map[cfg["datatype"]]
        BLK_M, BLK_N, BLK_K = cfg["BLK_M"], cfg["BLK_N"], cfg["BLK_K"]
        gsize_m = cfg["gsize_m"]
        num_stages = cfg["num_stages"]

        N_local = N_cfg // world_size

        A = shmem.randn(M, K_cfg, device="cuda", dtype=datatype)
        B_full = shmem.randn(N_cfg, K_cfg, device="cuda", dtype=datatype).T
        local_B = B_full[:, rank * N_local : (rank + 1) * N_local].clone()
        global_C = shmem.zeros((M, N_cfg), device="cuda", dtype=datatype)
        local_C = shmem.zeros((M, N_local), device="cuda", dtype=datatype)

        kernel_timing = {
            "start": torch.cuda.Event(enable_timing=True),
            "end": torch.cuda.Event(enable_timing=True),
            "ms": 0.0,
            "count": 0,
        }

        def run_experiment():
            shmem.barrier()
            with torch.cuda.stream(gemm_stream):
                kernel_timing["start"].record()
                matmul.apply(
                    A,
                    local_B,
                    local_C,
                    global_C,
                    None,
                    rank,
                    world_size,
                    num_sms,
                    BLK_M,
                    BLK_N,
                    BLK_K,
                    gsize_m,
                    num_stages,
                    context_tensor,
                    "gfx942",
                )
                kernel_timing["end"].record()
                kernel_timing["count"] += 1
            shmem.barrier()
            kernel_timing["ms"] += kernel_timing["start"].elapsed_time(kernel_timing["end"])

        # Warmup
        run_experiment()
        shmem.barrier()
        kernel_timing["ms"] = 0.0
        kernel_timing["count"] = 0

        # Benchmark
        total_ms = iris.do_bench(run_experiment, barrier_fn=shmem.barrier)
        tflops = 2 * M * N_cfg * K_cfg * 1e-12 / (total_ms * 1e-3)
        avg_kernel_ms = kernel_timing["ms"] / max(kernel_timing["count"], 1)

        label = f"M={M} BLK({BLK_M},{BLK_N},{BLK_K}) gs={gsize_m} st={num_stages}"
        shmem.info(f"{label}: {total_ms:.3f} ms  {tflops:.3f} TFLOPS  kernel={avg_kernel_ms:.3f} ms")

        if rank == 0:
            result = {**cfg, "num_sms": num_sms, "total_ms": total_ms, "tflops": tflops, "kernel_ms": avg_kernel_ms}
            fname = config_to_filename(cfg)
            out_path = os.path.join(output_dir, fname)
            with open(out_path, "w") as fp:
                json.dump(result, fp, indent=4)

    shmem.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------
def generate_charts(output_dir: str, chart_path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    # Load all result JSONs
    results = []
    for fname in os.listdir(output_dir):
        if fname.startswith("gemm_as_sweep_") and fname.endswith(".json"):
            with open(os.path.join(output_dir, fname)) as fp:
                results.append(json.load(fp))

    if not results:
        print(f"No sweep results found in {output_dir}")
        return

    m_vals = sorted(set(r["m"] for r in results))

    # ------------------------------------------------------------------
    # Figure layout: 3 subplots stacked vertically
    #   1. TFLOPS vs M for each (BLK_M, BLK_N, BLK_K) tile
    #   2. TFLOPS vs M for each num_stages (best tile fixed)
    #   3. TFLOPS vs M for each gsize_m (best tile fixed)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("GEMM+AllScatter Tiling Parameter Sweep\n8×MI300X  fp16  N=4096  K=14336", fontsize=13)

    # ---- helper ----
    def get_tflops(r_list, m, **filters):
        for r in r_list:
            if r["m"] != m:
                continue
            if all(r.get(k) == v for k, v in filters.items()):
                return r["tflops"]
        return None

    # ---- 1. Tile sweep ----
    ax = axes[0]
    tile_results = [r for r in results if r.get("sweep_group") == "tile"]
    tiles = sorted(set((r["BLK_M"], r["BLK_N"], r["BLK_K"]) for r in tile_results))
    colors = cm.tab10(np.linspace(0, 1, len(tiles)))
    for (blk_m, blk_n, blk_k), color in zip(tiles, colors):
        ys = [
            get_tflops(
                tile_results,
                m,
                BLK_M=blk_m,
                BLK_N=blk_n,
                BLK_K=blk_k,
                gsize_m=DEFAULT_GSIZE_M,
                num_stages=DEFAULT_NUM_STAGES,
            )
            for m in m_vals
        ]
        valid = [(m, y) for m, y in zip(m_vals, ys) if y is not None]
        if valid:
            xs, ys_v = zip(*valid)
            ax.plot(xs, ys_v, marker="o", label=f"({blk_m},{blk_n},{blk_k})", color=color)
    ax.set_title(f"Tile Size  (gsize_m={DEFAULT_GSIZE_M}, stages={DEFAULT_NUM_STAGES})")
    ax.set_xlabel("M")
    ax.set_ylabel("TFLOPS")
    ax.set_xscale("log", base=2)
    ax.set_xticks(m_vals)
    ax.set_xticklabels([str(m) for m in m_vals])
    ax.legend(title="(BLK_M,BLK_N,BLK_K)", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- 2. num_stages sweep ----
    ax = axes[1]
    blk_m, blk_n, blk_k = DEFAULT_TILE
    stage_results = [r for r in results if r.get("sweep_group") in ("tile", "num_stages")]
    all_stages = sorted(set(r["num_stages"] for r in stage_results))
    colors_s = cm.Set1(np.linspace(0, 0.8, len(all_stages)))
    for num_stages, color in zip(all_stages, colors_s):
        ys = [
            get_tflops(
                stage_results,
                m,
                BLK_M=blk_m,
                BLK_N=blk_n,
                BLK_K=blk_k,
                gsize_m=DEFAULT_GSIZE_M,
                num_stages=num_stages,
            )
            for m in m_vals
        ]
        valid = [(m, y) for m, y in zip(m_vals, ys) if y is not None]
        if valid:
            xs, ys_v = zip(*valid)
            ax.plot(xs, ys_v, marker="s", label=f"stages={num_stages}", color=color)
    ax.set_title(f"Pipeline Stages  BLK({blk_m},{blk_n},{blk_k}) gsize_m={DEFAULT_GSIZE_M}")
    ax.set_xlabel("M")
    ax.set_ylabel("TFLOPS")
    ax.set_xscale("log", base=2)
    ax.set_xticks(m_vals)
    ax.set_xticklabels([str(m) for m in m_vals])
    ax.legend(title="num_stages", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- 3. gsize_m sweep ----
    ax = axes[2]
    gsize_results = [r for r in results if r.get("sweep_group") in ("tile", "gsize_m")]
    all_gsizes = sorted(set(r["gsize_m"] for r in gsize_results))
    colors_g = cm.Set2(np.linspace(0, 0.9, len(all_gsizes)))
    for gsize_m, color in zip(all_gsizes, colors_g):
        ys = [
            get_tflops(
                gsize_results,
                m,
                BLK_M=blk_m,
                BLK_N=blk_n,
                BLK_K=blk_k,
                gsize_m=gsize_m,
                num_stages=DEFAULT_NUM_STAGES,
            )
            for m in m_vals
        ]
        valid = [(m, y) for m, y in zip(m_vals, ys) if y is not None]
        if valid:
            xs, ys_v = zip(*valid)
            ax.plot(xs, ys_v, marker="^", label=f"gsize_m={gsize_m}", color=color)
    ax.set_title(f"Group Size M  BLK({blk_m},{blk_n},{blk_k}) stages={DEFAULT_NUM_STAGES}")
    ax.set_xlabel("M")
    ax.set_ylabel("TFLOPS")
    ax.set_xscale("log", base=2)
    ax.set_xticks(m_vals)
    ax.set_xticklabels([str(m) for m in m_vals])
    ax.legend(title="gsize_m", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {chart_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep GEMM+AllScatter tiling parameters and generate performance charts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_ranks", type=int, default=8, help="Number of GPUs.")
    parser.add_argument(
        "--output_dir", type=str, default="/tmp/gemm_as_sweep_results", help="Directory for per-config JSON results."
    )
    parser.add_argument("--chart_only", action="store_true", help="Skip benchmarking; only regenerate the chart.")
    parser.add_argument("--chart_path", type=str, default=None, help="Output PNG path (default: <output_dir>/chart.png)")
    return parser.parse_args()


def main():
    args = parse_args()
    chart_path = args.chart_path or os.path.join(args.output_dir, "gemm_all_scatter_tiling_sweep.png")

    if not args.chart_only:
        configs = build_sweep_configs()
        print(f"Running {len(configs)} configurations on {args.num_ranks} GPUs …")
        init_url = "tcp://127.0.0.1:29503"
        mp.spawn(
            fn=worker,
            args=(args.num_ranks, init_url, configs, args.output_dir),
            nprocs=args.num_ranks,
            join=True,
        )
        print("Benchmarking complete.")

    generate_charts(args.output_dir, chart_path)


if __name__ == "__main__":
    main()
