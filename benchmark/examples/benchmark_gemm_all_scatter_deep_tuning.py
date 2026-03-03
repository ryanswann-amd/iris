#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
Deep GEMM utilization sweep for GEMM+AllScatter.

Explores num_warps, mfma (matrix_instr_nonkdim), BLK_K, num_stages, and
num_sms (partial SM assignment) to maximize GEMM compute efficiency.

Motivation
----------
The strong/weak scaling analysis showed a 3.5–4.3× TFLOPS gap between
rocBLAS (GEMM-only) and the Triton fused kernel.  This script isolates how
much of that gap can be closed by tuning the low-level GEMM knobs that are
currently hardcoded in matmul_wrapper.py:

  - num_warps    : wave-front occupancy per CU (currently 8)
  - mfma         : MFMA instruction dimension (currently 16 → v_mfma_f32_16x16x16f16)
                   mfma=32 → v_mfma_f32_32x32x8f16  (4× more MACs per instruction)
  - BLK_K        : tile depth → halving K-iterations halves s_barrier count
  - num_stages   : software-pipeline depth for global→LDS prefetch
  - num_sms      : launch fewer CUs to improve per-CU occupancy

Usage
-----
    # Full sweep (8 GPUs)
    python benchmark/examples/benchmark_gemm_all_scatter_deep_tuning.py \\
        --num_ranks 8 --output_dir results/deep_tuning

    # Chart-only from existing results
    python benchmark/examples/benchmark_gemm_all_scatter_deep_tuning.py \\
        --chart_only --output_dir results/deep_tuning
"""

import argparse
import itertools
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import importlib.util

import iris

# ---------------------------------------------------------------------------
# Dynamically load kernel + wrapper from examples directory
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
matmul_cls = wrapper_module.matmul
gemm_kernel = kernel_module.persistent_gemm_all_scatter

torch.manual_seed(123)
random.seed(123)

# ---------------------------------------------------------------------------
# Sweep configuration space
# Fixed: BLK_M=64, BLK_N=64, gsize_m=8 (established best tile)
# ---------------------------------------------------------------------------
M_VALUES = [256, 512, 1024]
N, K = 4096, 14336
DATATYPE = "fp16"
BLK_M, BLK_N = 64, 64
GSIZE_M = 8

# Knobs to sweep
NUM_WARPS_VALUES = [4, 8]          # wavefront occupancy
MFMA_VALUES = [16, 32]             # matrix_instr_nonkdim (16×16 and 32×32 MFMA)
BLK_K_STAGES = [                   # (BLK_K, num_stages) pairs that fit in 64 KB LDS
    (64, 2),                       # LDS = (64*64*2 + 64*64*2)*2 = 32 KB  (baseline)
    (64, 3),                       # LDS = (64*64*3 + 64*64*3)*2 = 48 KB  (our current best)
    (128, 2),                      # LDS = (64*128*2 + 128*64*2)*2 = 64 KB (half barriers!)
]
# num_sms fractions: 1.0 = all CUs, "tiles" = exactly total_tiles CUs (100% SM utilisation)
NUM_SMS_MODES = ["full", "tiles"]   # "full"=304 CUs, "tiles"=ceil(M/64)*ceil(N/8/64)


def build_sweep_configs(total_sms: int, world_size: int):
    """Return all config dicts for the deep tuning sweep."""
    configs = []
    for m in M_VALUES:
        n_local = N // world_size
        total_tiles = math.ceil(m / BLK_M) * math.ceil(n_local / BLK_N)
        for (blk_k, num_stages), num_warps, mfma, sms_mode in itertools.product(
            BLK_K_STAGES, NUM_WARPS_VALUES, MFMA_VALUES, NUM_SMS_MODES
        ):
            num_sms_launch = total_sms if sms_mode == "full" else max(1, total_tiles)
            configs.append(
                dict(
                    m=m,
                    n=N,
                    k=K,
                    BLK_M=BLK_M,
                    BLK_N=BLK_N,
                    BLK_K=blk_k,
                    gsize_m=GSIZE_M,
                    num_stages=num_stages,
                    num_warps=num_warps,
                    mfma=mfma,
                    num_sms_mode=sms_mode,
                    num_sms_launch=num_sms_launch,
                    total_tiles=total_tiles,
                    datatype=DATATYPE,
                )
            )
    return configs


def config_to_filename(cfg, base="deep_tune"):
    return (
        f"{base}_m{cfg['m']}"
        f"_blkk{cfg['BLK_K']}_st{cfg['num_stages']}"
        f"_nw{cfg['num_warps']}_mfma{cfg['mfma']}"
        f"_sms{cfg['num_sms_mode']}.json"
    )


# ---------------------------------------------------------------------------
# Custom kernel launcher that overrides hardcoded matmul_wrapper knobs
# ---------------------------------------------------------------------------
def run_kernel(
    a, b, c, c_global, bias_placeholder, rank, world_size,
    num_sms_launch, BLK_M, BLK_N, BLK_K, gsize_m, num_stages,
    num_warps, mfma, context_tensor, arch="gfx942",
):
    """Launch persistent_gemm_all_scatter with full knob control."""
    import math as _math
    M, K = a.shape
    _, N = b.shape
    num_xcds = matmul_cls._num_xcds
    even_k = K % BLK_K == 0

    gemm_kernel[(num_sms_launch,)](
        a, b, c, c_global,
        bias_placeholder,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        c_global.stride(0), c_global.stride(1),
        0,                          # stride_bias (bias not used)
        BLOCK_SIZE_M=BLK_M,
        BLOCK_SIZE_N=BLK_N,
        BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=gsize_m,
        NUM_SMS=num_sms_launch,
        NUM_XCDS=num_xcds,
        BIAS=False,
        EVEN_K=even_k,
        num_stages=num_stages,
        num_warps=num_warps,
        waves_per_eu=0,
        matrix_instr_nonkdim=mfma,
        kpack=1,
        context_tensor=context_tensor,
        cur_rank=rank,
        world_size=world_size,
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

    total_sms = torch.cuda.get_device_properties(rank).multi_processor_count
    gemm_stream = torch.cuda.Stream()

    for cfg in configs:
        M = cfg["m"]
        N_cfg = cfg["n"]
        K_cfg = cfg["k"]
        datatype = dtype_map[cfg["datatype"]]
        blk_k = cfg["BLK_K"]
        num_stages = cfg["num_stages"]
        num_warps = cfg["num_warps"]
        mfma = cfg["mfma"]
        num_sms_launch = cfg["num_sms_launch"]
        gsize_m = cfg["gsize_m"]

        N_local = N_cfg // world_size

        A = shmem.randn(M, K_cfg, device="cuda", dtype=datatype)
        B_full = shmem.randn(N_cfg, K_cfg, device="cuda", dtype=datatype).T
        local_B = B_full[:, rank * N_local: (rank + 1) * N_local].clone()
        global_C = shmem.zeros((M, N_cfg), device="cuda", dtype=datatype)
        local_C = shmem.zeros((M, N_local), device="cuda", dtype=datatype)
        # bias placeholder (unused, but kernel expects a tensor)
        bias_ph = shmem.zeros((M,), device="cuda", dtype=datatype)

        kernel_timing = {"start": torch.cuda.Event(enable_timing=True),
                         "end": torch.cuda.Event(enable_timing=True),
                         "ms": 0.0, "count": 0}

        def run_experiment():
            shmem.barrier()
            with torch.cuda.stream(gemm_stream):
                kernel_timing["start"].record()
                run_kernel(
                    A, local_B, local_C, global_C, bias_ph,
                    rank, world_size, num_sms_launch,
                    BLK_M, BLK_N, blk_k, gsize_m, num_stages,
                    num_warps, mfma, context_tensor, "gfx942",
                )
                kernel_timing["end"].record()
                kernel_timing["count"] += 1
            shmem.barrier()
            kernel_timing["ms"] += kernel_timing["start"].elapsed_time(kernel_timing["end"])

        # Warmup
        try:
            run_experiment()
        except Exception as exc:
            if rank == 0:
                print(f"[SKIP] M={M} BLK_K={blk_k} st={num_stages} nw={num_warps} mfma={mfma}: {exc}")
            shmem.barrier()
            continue

        shmem.barrier()
        kernel_timing["ms"] = 0.0
        kernel_timing["count"] = 0

        try:
            total_ms = iris.do_bench(run_experiment, barrier_fn=shmem.barrier)
        except Exception as exc:
            if rank == 0:
                print(f"[SKIP bench] M={M} BLK_K={blk_k} st={num_stages} nw={num_warps} mfma={mfma}: {exc}")
            shmem.barrier()
            continue

        tflops = 2 * M * N_cfg * K_cfg * 1e-12 / (total_ms * 1e-3)
        label = (f"M={M} BLK_K={blk_k} st={num_stages} nw={num_warps} "
                 f"mfma={mfma} sms={cfg['num_sms_mode']}")
        shmem.info(f"{label}: {total_ms:.3f} ms  {tflops:.3f} TFLOPS")

        if rank == 0:
            result = {
                **cfg,
                "total_sms": total_sms,
                "total_ms": total_ms,
                "tflops": tflops,
            }
            out_path = os.path.join(output_dir, config_to_filename(cfg))
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
    import numpy as np

    results = []
    for fname in os.listdir(output_dir):
        if fname.startswith("deep_tune_") and fname.endswith(".json"):
            with open(os.path.join(output_dir, fname)) as fp:
                results.append(json.load(fp))

    if not results:
        print(f"No deep-tuning results in {output_dir}")
        return

    m_vals = sorted(set(r["m"] for r in results))

    def get_tflops(m, blk_k, num_stages, num_warps, mfma, sms_mode):
        for r in results:
            if (r["m"] == m and r["BLK_K"] == blk_k and r["num_stages"] == num_stages
                    and r["num_warps"] == num_warps and r["mfma"] == mfma
                    and r["num_sms_mode"] == sms_mode):
                return r["tflops"]
        return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "GEMM+AllScatter Deep Tuning — 8×MI300X  fp16  BLK_M=64, BLK_N=64\n"
        "N=4096, K=14336  (N_local=512/GPU)",
        fontsize=12,
    )

    COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

    # ── Panel A: (BLK_K, num_stages) sweep ──────────────────────────────
    ax = axes[0, 0]
    blk_k_st_configs = [(64, 2), (64, 3), (128, 2)]
    labels_a = ["BLK_K=64 st=2 (baseline)", "BLK_K=64 st=3 (current best)", "BLK_K=128 st=2 (half barriers)"]
    for i, ((bk, st), lbl) in enumerate(zip(blk_k_st_configs, labels_a)):
        # Best over num_warps, mfma, sms for each M
        ys = []
        for m in m_vals:
            best = max(
                (get_tflops(m, bk, st, nw, mf, sm) or 0)
                for nw in NUM_WARPS_VALUES for mf in MFMA_VALUES for sm in NUM_SMS_MODES
            )
            ys.append(best if best > 0 else None)
        valid = [(m, y) for m, y in zip(m_vals, ys) if y is not None]
        if valid:
            xs, ys_v = zip(*valid)
            ax.plot(xs, ys_v, marker="o", label=lbl, color=COLORS[i])
    ax.set_title("(A) BLK_K / num_stages  [best over other knobs]")
    ax.set_xlabel("M (sequence length)")
    ax.set_ylabel("TFLOPS (8-GPU total)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel B: num_warps sweep ─────────────────────────────────────────
    ax = axes[0, 1]
    for i, nw in enumerate(NUM_WARPS_VALUES):
        ys = []
        for m in m_vals:
            best = max(
                (get_tflops(m, bk, st, nw, mf, sm) or 0)
                for bk, st in BLK_K_STAGES for mf in MFMA_VALUES for sm in NUM_SMS_MODES
            )
            ys.append(best if best > 0 else None)
        valid = [(m, y) for m, y in zip(m_vals, ys) if y is not None]
        if valid:
            xs, ys_v = zip(*valid)
            ax.plot(xs, ys_v, marker="s", label=f"num_warps={nw}", color=COLORS[i])
    ax.set_title("(B) num_warps  [best over other knobs]")
    ax.set_xlabel("M (sequence length)")
    ax.set_ylabel("TFLOPS (8-GPU total)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel C: mfma sweep ──────────────────────────────────────────────
    ax = axes[1, 0]
    mfma_labels = {16: "mfma=16 (16×16 MFMA, current)", 32: "mfma=32 (32×32 MFMA, 4× MACs)"}
    for i, mf in enumerate(MFMA_VALUES):
        ys = []
        for m in m_vals:
            best = max(
                (get_tflops(m, bk, st, nw, mf, sm) or 0)
                for bk, st in BLK_K_STAGES for nw in NUM_WARPS_VALUES for sm in NUM_SMS_MODES
            )
            ys.append(best if best > 0 else None)
        valid = [(m, y) for m, y in zip(m_vals, ys) if y is not None]
        if valid:
            xs, ys_v = zip(*valid)
            ax.plot(xs, ys_v, marker="^", label=mfma_labels.get(mf, f"mfma={mf}"), color=COLORS[i])
    ax.set_title("(C) MFMA instruction size  [best over other knobs]")
    ax.set_xlabel("M (sequence length)")
    ax.set_ylabel("TFLOPS (8-GPU total)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel D: num_sms_mode sweep + overall best config table ─────────
    ax = axes[1, 1]
    sms_labels = {"full": "num_sms=304 (all CUs)", "tiles": "num_sms=tiles (100% util)"}
    for i, sm in enumerate(NUM_SMS_MODES):
        ys = []
        for m in m_vals:
            best = max(
                (get_tflops(m, bk, st, nw, mf, sm) or 0)
                for bk, st in BLK_K_STAGES for nw in NUM_WARPS_VALUES for mf in MFMA_VALUES
            )
            ys.append(best if best > 0 else None)
        valid = [(m, y) for m, y in zip(m_vals, ys) if y is not None]
        if valid:
            xs, ys_v = zip(*valid)
            ax.plot(xs, ys_v, marker="D", label=sms_labels.get(sm, f"sms={sm}"), color=COLORS[i])

    # Overlay current best (BLK_K=64, st=3, nw=8, mfma=16, full) for reference
    ref_ys = []
    for m in m_vals:
        t = get_tflops(m, 64, 3, 8, 16, "full")
        ref_ys.append(t)
    valid_ref = [(m, y) for m, y in zip(m_vals, ref_ys) if y is not None]
    if valid_ref:
        xs_r, ys_r = zip(*valid_ref)
        ax.plot(xs_r, ys_r, marker="*", linestyle="--", color="gray",
                label="prev best (BLK_K=64,st=3,nw=8,mfma=16,full)", zorder=5)

    ax.set_title("(D) num_sms mode  [best over other knobs]")
    ax.set_xlabel("M (sequence length)")
    ax.set_ylabel("TFLOPS (8-GPU total)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {chart_path}")

    # Print summary table of best configs per M
    print("\n=== Best config per M ===")
    print(f"{'M':>5}  {'BLK_K':>5}  {'st':>2}  {'nw':>4}  {'mfma':>4}  {'sms':>6}  {'TFLOPS':>8}")
    print("-" * 55)
    for m in m_vals:
        best_t, best_cfg = 0, {}
        for bk, st in BLK_K_STAGES:
            for nw in NUM_WARPS_VALUES:
                for mf in MFMA_VALUES:
                    for sm in NUM_SMS_MODES:
                        t = get_tflops(m, bk, st, nw, mf, sm) or 0
                        if t > best_t:
                            best_t, best_cfg = t, dict(BLK_K=bk, st=st, nw=nw, mfma=mf, sms=sm)
        if best_cfg:
            print(f"{m:>5}  {best_cfg['BLK_K']:>5}  {best_cfg['st']:>2}  "
                  f"{best_cfg['nw']:>4}  {best_cfg['mfma']:>4}  "
                  f"{best_cfg['sms']:>6}  {best_t:>8.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Deep GEMM utilization sweep for GEMM+AllScatter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_ranks", type=int, default=8, help="Number of GPUs.")
    parser.add_argument("--output_dir", type=str, default="results/deep_tuning",
                        help="Directory for per-config JSON results.")
    parser.add_argument("--chart_only", action="store_true",
                        help="Skip benchmarking and only regenerate the chart.")
    return parser.parse_args()


def main():
    args = parse_args()
    chart_path = os.path.join(
        os.path.dirname(args.output_dir),
        "gemm_all_scatter_deep_tuning_mi300x.png",
    )

    if not args.chart_only:
        world_size = args.num_ranks
        # total_sms is only used to compute "full" num_sms_launch in build_sweep_configs.
        # The actual per-rank SM count is read inside worker() via get_device_properties.
        total_sms_for_config = torch.cuda.get_device_properties(0).multi_processor_count if torch.cuda.is_available() else 304
        configs = build_sweep_configs(total_sms=total_sms_for_config, world_size=world_size)

        if not configs:
            print("No configs to run.")
            return

        init_url = "tcp://127.0.0.1:18188"
        mp.start_processes(
            worker,
            args=(world_size, init_url, configs, args.output_dir),
            nprocs=world_size,
            start_method="spawn",
        )

    generate_charts(args.output_dir, chart_path)


if __name__ == "__main__":
    main()
