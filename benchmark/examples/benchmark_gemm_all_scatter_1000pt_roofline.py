#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
1000-point comprehensive roofline scatter sweep for GEMM+AllScatter.

Generates a single scatter plot per world size with:
  - X-axis = M × N × K (log scale)
  - Y-axis = TFLOPS (8-GPU total)
  - Each unique (BLK_M, BLK_N, BLK_K, num_stages, num_warps, mfma, sms_mode)
    kernel configuration uniquely colored

Parameter space (targeting ~1000 valid data points):
  Tile + pipeline configs (BLK_M, BLK_N, BLK_K, stages):
    (64,  64,  64,  2)  baseline small tile
    (64,  64,  64,  3)  small tile + extra pipeline stage
    (64,  64, 128,  2)  doubled K-depth (halves s_barrier count)
    (128, 64,  64,  2)  medium M tile
    (256, 64,  64,  2)  large M tile (default)
  num_warps : {4, 8}
  mfma      : {16, 32}
  sms_mode  : {"full" (304 CUs), "tiles" (exactly total_tiles CUs)}

  5 tile configs × 2 warps × 2 mfma × 2 sms = 40 kernel configs
  M ∈ {32, 64, 128, 256, 512, 1024} × (N,K) ∈ {5 shapes} = 30 problem sizes
  → up to 1200 attempted; ~1000 expected valid after OOM skips

Usage
-----
    # Full sweep (8 GPUs, ~6–8 hours)
    python benchmark/examples/benchmark_gemm_all_scatter_1000pt_roofline.py \\
        --num_ranks 8 --output_dir results/roofline_1000pt

    # Chart-only from existing results
    python benchmark/examples/benchmark_gemm_all_scatter_1000pt_roofline.py \\
        --chart_only --output_dir results/roofline_1000pt
"""

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the correct Python paths are set for triton + iris + tritonblas stub
# ---------------------------------------------------------------------------
_TRITON_PYTHON = Path("/opt/triton/python")
_VENV_SITE = Path("/opt/venv/lib/python3.13/site-packages")
_TRITONBLAS_STUB = Path("/tmp/tritonblas_stub")

for _p in [_TRITON_PYTHON, _VENV_SITE, _TRITONBLAS_STUB]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Use the persistent triton cache if TRITON_CACHE_DIR is not already set
# (Users can override this via environment variable before running the script)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
import importlib.util  # noqa: E402

import iris  # noqa: E402

# ---------------------------------------------------------------------------
# Dynamically load kernel + wrapper from examples directory
# ---------------------------------------------------------------------------
_current_dir = Path(__file__).parent
_kernel_path = (_current_dir / "../../examples/23_gemm_all_scatter_tracing/gemm_all_scatter.py").resolve()
_wrapper_path = (_current_dir / "../../examples/23_gemm_all_scatter_tracing/matmul_wrapper.py").resolve()

_kernel_spec = importlib.util.spec_from_file_location("gemm_all_scatter", _kernel_path)
_kernel_module = importlib.util.module_from_spec(_kernel_spec)
sys.modules["gemm_all_scatter"] = _kernel_module
_kernel_spec.loader.exec_module(_kernel_module)

_wrapper_spec = importlib.util.spec_from_file_location("matmul_wrapper", _wrapper_path)
_wrapper_module = importlib.util.module_from_spec(_wrapper_spec)
_wrapper_spec.loader.exec_module(_wrapper_module)
matmul_cls = _wrapper_module.matmul
gemm_kernel = _kernel_module.persistent_gemm_all_scatter

# ---------------------------------------------------------------------------
# Sweep configuration space
# ---------------------------------------------------------------------------

# Problem sizes: M values × (N, K) shapes
M_VALUES = [32, 64, 128, 256, 512, 1024]
NK_SHAPES = [
    (4096, 4096),
    (4096, 14336),
    (8192, 4096),
    (8192, 14336),
    (8192, 28672),
]

# Kernel parameter sweeps
# (BLK_M, BLK_N, BLK_K, num_stages) — only LDS-valid combinations
TILE_STAGE_CONFIGS = [
    (64, 64, 64, 2),  # 32 KB LDS — baseline small tile
    (64, 64, 64, 3),  # 48 KB LDS — extra pipeline stage
    (64, 64, 128, 2),  # 64 KB LDS — half the barriers
    (128, 64, 64, 2),  # 48 KB LDS — medium M tile
    (256, 64, 64, 2),  # 80 KB LDS — large M tile (default); may spill on some configs
]
NUM_WARPS_LIST = [4, 8]
MFMA_LIST = [16, 32]  # matrix_instr_nonkdim: 16 → 16×16 MFMA, 32 → 32×32 MFMA
SMS_MODES = ["full", "tiles"]  # "full"=all CUs, "tiles"=exactly ceil(M/BLK_M)*ceil(N_local/BLK_N)

GSIZE_M = 8  # group-size-M (minimal impact; fixed at best value)


def lds_bytes(blk_m: int, blk_n: int, blk_k: int, stages: int) -> int:
    """Conservative LDS estimate (A-tile + B-tile, fp16, double-buffered × stages)."""
    return (blk_m * blk_k + blk_n * blk_k) * 2 * stages


def build_sweep_configs(total_sms: int, world_size: int):
    """Return all (kernel_config, problem_size) dicts for the sweep."""
    configs = []
    for m, (n, k) in itertools.product(M_VALUES, NK_SHAPES):
        # Skip shapes that aren't divisible by world_size
        if n % world_size != 0:
            continue
        n_local = n // world_size
        for (blk_m, blk_n, blk_k, stages), num_warps, mfma, sms_mode in itertools.product(
            TILE_STAGE_CONFIGS, NUM_WARPS_LIST, MFMA_LIST, SMS_MODES
        ):
            total_tiles = math.ceil(m / blk_m) * math.ceil(n_local / blk_n)
            num_sms_launch = total_sms if sms_mode == "full" else max(1, total_tiles)
            configs.append(
                dict(
                    m=m,
                    n=n,
                    k=k,
                    BLK_M=blk_m,
                    BLK_N=blk_n,
                    BLK_K=blk_k,
                    gsize_m=GSIZE_M,
                    num_stages=stages,
                    num_warps=num_warps,
                    mfma=mfma,
                    sms_mode=sms_mode,
                    num_sms_launch=num_sms_launch,
                    total_tiles=total_tiles,
                    lds_kb=lds_bytes(blk_m, blk_n, blk_k, stages) // 1024,
                    mnk=m * n * k,
                )
            )
    return configs


def config_key(cfg) -> str:
    return (
        f"m{cfg['m']}_n{cfg['n']}_k{cfg['k']}"
        f"_bm{cfg['BLK_M']}_bn{cfg['BLK_N']}_bk{cfg['BLK_K']}"
        f"_st{cfg['num_stages']}_nw{cfg['num_warps']}"
        f"_mfma{cfg['mfma']}_sms{cfg['sms_mode']}"
    )


def config_to_filename(cfg) -> str:
    return f"rfl_{config_key(cfg)}.json"


def kernel_config_label(cfg) -> str:
    """Short human-readable label for kernel params only (no M/N/K)."""
    return (
        f"BLK({cfg['BLK_M']},{cfg['BLK_N']},{cfg['BLK_K']})"
        f" st{cfg['num_stages']}"
        f" nw{cfg['num_warps']}"
        f" mfma{cfg['mfma']}"
        f" sms={cfg['sms_mode']}"
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

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    dist.barrier()

    datatype = torch.float16
    n_total = len(configs)
    total_sms = torch.cuda.get_device_properties(rank).multi_processor_count

    # Single iris heap for the entire sweep.
    # Pre-allocate tensors for every unique (M, N, K) shape upfront so that
    # the bump allocator never exceeds its budget regardless of iteration order.
    # Total pre-allocated size (sum over 30 shapes) is ~1.7 GB, well under 8 GB.
    shmem = iris.iris(1 << 33)
    real_world_size = shmem.get_num_ranks()
    context_tensor = shmem.get_device_context()
    num_xcds = matmul_cls._num_xcds

    # Collect unique (M, N, K) shapes
    shapes = sorted({(cfg["m"], cfg["n"], cfg["k"]) for cfg in configs})

    # Pre-allocate tensors for each shape
    tensor_cache = {}
    for M, N_cfg, K_cfg in shapes:
        N_local = N_cfg // real_world_size
        A = shmem.randn(M, K_cfg, device="cuda", dtype=datatype)
        local_B = shmem.randn(K_cfg, N_local, device="cuda", dtype=datatype)
        global_C = shmem.zeros((M, N_cfg), device="cuda", dtype=datatype)
        local_C = shmem.zeros((M, N_local), device="cuda", dtype=datatype)
        bias_ph = shmem.zeros((M,), device="cuda", dtype=datatype)
        tensor_cache[(M, N_cfg, K_cfg)] = (A, local_B, global_C, local_C, bias_ph)

    if rank == 0:
        heap_mb = (
            sum(
                (m * k + k * (n // real_world_size) + m * n + m * (n // real_world_size) + m) * 2
                for (m, n, k) in shapes
            )
            / 1e6
        )
        print(f"Pre-allocated tensors for {len(shapes)} shapes (~{heap_mb:.0f} MB)", flush=True)

    n_done = 0
    for cfg in configs:
        out_path = os.path.join(output_dir, config_to_filename(cfg))
        if os.path.exists(out_path):
            n_done += 1
            shmem.barrier()
            continue

        M = cfg["m"]
        N_cfg = cfg["n"]
        K_cfg = cfg["k"]
        blk_m = cfg["BLK_M"]
        blk_n = cfg["BLK_N"]
        blk_k = cfg["BLK_K"]
        gsize_m = cfg["gsize_m"]
        num_stages = cfg["num_stages"]
        num_warps = cfg["num_warps"]
        mfma = cfg["mfma"]
        sms_mode = cfg["sms_mode"]
        num_sms_launch = cfg["num_sms_launch"]
        N_local = N_cfg // real_world_size
        even_k = K_cfg % blk_k == 0

        A, local_B, global_C, local_C, bias_ph = tensor_cache[(M, N_cfg, K_cfg)]

        def run_kernel():
            gemm_kernel[(num_sms_launch,)](
                A,
                local_B,
                local_C,
                global_C,
                bias_ph,
                M,
                N_cfg,
                K_cfg,
                A.stride(0),
                A.stride(1),
                local_B.stride(0),
                local_B.stride(1),
                local_C.stride(0),
                local_C.stride(1),
                global_C.stride(0),
                global_C.stride(1),
                0,
                BLOCK_SIZE_M=blk_m,
                BLOCK_SIZE_N=blk_n,
                BLOCK_SIZE_K=blk_k,
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
                world_size=real_world_size,
            )

        def run_experiment():
            shmem.barrier()  # noqa: F821
            run_kernel()
            shmem.barrier()  # noqa: F821

        # Warmup
        try:
            run_experiment()
        except Exception as exc:
            if rank == 0:
                print(
                    f"[SKIP] M={M} N={N_cfg} K={K_cfg}"
                    f" BLK=({blk_m},{blk_n},{blk_k}) st={num_stages}"
                    f" nw={num_warps} mfma={mfma}: {exc}",
                    flush=True,
                )
            n_done += 1
            shmem.barrier()
            continue

        # Benchmark
        try:
            total_ms = iris.do_bench(run_experiment, barrier_fn=shmem.barrier)
        except Exception as exc:
            if rank == 0:
                print(
                    f"[SKIP bench] M={M} N={N_cfg} K={K_cfg}"
                    f" BLK=({blk_m},{blk_n},{blk_k}) st={num_stages}"
                    f" nw={num_warps} mfma={mfma}: {exc}",
                    flush=True,
                )
            n_done += 1
            shmem.barrier()
            continue

        tflops = 2 * M * N_cfg * K_cfg * 1e-12 / (total_ms * 1e-3)
        n_done += 1

        if rank == 0:
            pct = 100 * n_done / n_total
            if n_done % 10 == 0:
                print(
                    f"[{pct:.0f}% {n_done}/{n_total}] "
                    f"M={M} N={N_cfg} K={K_cfg} "
                    f"BLK=({blk_m},{blk_n},{blk_k}) st={num_stages} "
                    f"nw={num_warps} mfma={mfma} sms={sms_mode}: "
                    f"{total_ms:.3f}ms  {tflops:.2f}T",
                    flush=True,
                )
            result = {
                **cfg,
                "total_sms": total_sms,
                "world_size": real_world_size,
                "total_ms": total_ms,
                "tflops": tflops,
            }
            with open(out_path, "w") as fp:
                json.dump(result, fp, indent=2)

        shmem.barrier()

    if rank == 0:
        n_files = len([f for f in os.listdir(output_dir) if f.startswith("rfl_")])
        print(f"[rank 0] Complete. {n_files} results saved to {output_dir}", flush=True)
    shmem.barrier()
    del shmem
    dist.destroy_process_group()


def generate_chart(output_dir: str, chart_path: str, world_size: int = 8):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Load all results
    results = []
    for fname in sorted(os.listdir(output_dir)):
        if fname.startswith("rfl_") and fname.endswith(".json"):
            with open(os.path.join(output_dir, fname)) as fp:
                try:
                    results.append(json.load(fp))
                except json.JSONDecodeError:
                    pass

    if not results:
        print(f"No roofline results found in {output_dir}")
        return

    print(f"Loaded {len(results)} result(s) from {output_dir}")

    # Filter to matching world_size
    ws_results = [r for r in results if r.get("world_size", world_size) == world_size]
    if not ws_results:
        ws_results = results  # fallback: use all
    print(f"Using {len(ws_results)} result(s) for world_size={world_size}")

    # Build unique kernel config labels and assign colors
    unique_labels = []
    label_order = {}
    for r in ws_results:
        lbl = kernel_config_label(r)
        if lbl not in label_order:
            label_order[lbl] = len(label_order)
            unique_labels.append(lbl)

    n_configs = len(unique_labels)
    print(f"Found {n_configs} unique kernel configurations")

    # Color palette: combine multiple colormaps for 40+ distinct colors
    # Use HSV-based palette for maximum distinctiveness
    cmap_colors = []
    # Use tab20 + tab20b + tab20c for up to 60 colors
    tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
    tab20b = plt.cm.tab20b(np.linspace(0, 1, 20))
    tab20c = plt.cm.tab20c(np.linspace(0, 1, 20))
    full_palette = np.vstack([tab20, tab20b, tab20c])
    # Shuffle to maximize color distance between adjacent labels
    np.random.seed(42)
    palette_idx = np.arange(len(full_palette))
    np.random.shuffle(palette_idx)
    full_palette = full_palette[palette_idx]

    color_map = {}
    for i, lbl in enumerate(unique_labels):
        color_map[lbl] = full_palette[i % len(full_palette)]

    # Marker shapes cycle (useful secondary visual indicator)
    marker_cycle = ["o", "s", "^", "v", "D", "P", "X", "*", "h", "+"]

    # Build a structured marker assignment:
    # (BLK_M, BLK_N, BLK_K, stages) → marker shape
    tile_marker = {}
    tile_keys = []
    for r in ws_results:
        tk = (r["BLK_M"], r["BLK_N"], r["BLK_K"], r["num_stages"])
        if tk not in tile_marker:
            tile_marker[tk] = marker_cycle[len(tile_marker) % len(marker_cycle)]
            tile_keys.append(tk)

    # Group results by label
    label_to_points = {lbl: {"x": [], "y": []} for lbl in unique_labels}
    for r in ws_results:
        lbl = kernel_config_label(r)
        x_val = r["m"] * r["n"] * r["k"]
        y_val = r["tflops"]
        label_to_points[lbl]["x"].append(x_val)
        label_to_points[lbl]["y"].append(y_val)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig_w = 20
    fig_h = 14
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Hardware ceiling lines
    n_gpus = world_size
    fp16_peak_per_gpu = 1307.4  # TFLOPS (MI300X)
    hbm_bw_tb = 5.3  # TB/s per GPU
    xgmi_bw_tb = 3.15 / n_gpus  # TB/s aggregate XGMI divided across GPUs

    # Draw ceiling lines (use full x range)
    x_min_mnk = min(r["m"] * r["n"] * r["k"] for r in ws_results)
    x_max_mnk = max(r["m"] * r["n"] * r["k"] for r in ws_results)
    x_range = np.logspace(np.log10(x_min_mnk * 0.5), np.log10(x_max_mnk * 2.0), 200)

    ax.axhline(
        fp16_peak_per_gpu * n_gpus,
        color="red",
        linewidth=2.5,
        linestyle="--",
        alpha=0.8,
        zorder=1,
        label=f"FP16 tensor peak ({fp16_peak_per_gpu * n_gpus:.0f} TFLOPS, {n_gpus}×MI300X)",
    )
    ax.axhline(
        fp16_peak_per_gpu * n_gpus * 0.42,  # observed max SM util at M=1024
        color="orange",
        linewidth=1.5,
        linestyle=":",
        alpha=0.8,
        zorder=1,
        label=f"SM utilisation ceiling (42% of peak, {fp16_peak_per_gpu * n_gpus * 0.42:.0f} TFLOPS)",
    )

    # Plot each config
    for lbl in unique_labels:
        pts = label_to_points[lbl]
        if not pts["x"]:
            continue
        xs = np.array(pts["x"])
        ys = np.array(pts["y"])
        sort_idx = np.argsort(xs)
        xs, ys = xs[sort_idx], ys[sort_idx]

        # Determine tile key for first point with this label
        tk = None
        for r in ws_results:
            if kernel_config_label(r) == lbl:
                tk = (r["BLK_M"], r["BLK_N"], r["BLK_K"], r["num_stages"])
                break
        marker = tile_marker.get(tk, "o")

        ax.scatter(
            xs,
            ys,
            color=color_map[lbl],
            marker=marker,
            s=40,
            alpha=0.75,
            linewidths=0.3,
            edgecolors="none",
            label=lbl,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("M × N × K (FLOPs / 2)", fontsize=13)
    ax.set_ylabel("TFLOPS (8-GPU total)", fontsize=13)
    ax.set_title(
        f"GEMM+AllScatter Roofline — {n_gpus}×MI300X  fp16\n"
        f"{len(ws_results)} data points  ·  {n_configs} kernel configurations\n"
        f"Each color = unique (tile, stages, warps, mfma, sms_mode)",
        fontsize=11,
    )
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)

    # Legend — split into two: hardware ceilings + configs
    # Put hardware lines in main legend, configs in small inset legend
    handles, labels_leg = ax.get_legend_handles_labels()
    hw_handles = [h for h, l in zip(handles, labels_leg) if "peak" in l or "ceiling" in l]
    hw_labels = [l for l in labels_leg if "peak" in l or "ceiling" in l]
    cfg_handles = [h for h, l in zip(handles, labels_leg) if l not in hw_labels]
    cfg_labels = [l for l in labels_leg if l not in hw_labels]

    # Primary legend (hardware ceilings) — top-left
    ax.legend(hw_handles, hw_labels, loc="upper left", fontsize=9, framealpha=0.9)

    # Config legend — outside right, small
    if cfg_handles:
        config_legend = ax.legend(
            cfg_handles,
            cfg_labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0,
            fontsize=6.5,
            ncol=1,
            framealpha=0.9,
            title="Kernel configurations",
            title_fontsize=7,
        )
        ax.add_artist(config_legend)
        # Re-add hardware legend
        ax.legend(hw_handles, hw_labels, loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 0.72, 1.0])
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {chart_path}")

    # Print summary statistics
    tflops_all = [r["tflops"] for r in ws_results]
    print(f"\n=== Summary ({len(ws_results)} results) ===")
    print(f"  Min TFLOPS : {min(tflops_all):.2f}")
    print(f"  Max TFLOPS : {max(tflops_all):.2f}")
    print(f"  Mean TFLOPS: {sum(tflops_all) / len(tflops_all):.2f}")

    # Best config per (M, N, K)
    print("\n=== Best config per (M, N, K) ===")
    mnk_best = {}
    for r in ws_results:
        key = (r["m"], r["n"], r["k"])
        if key not in mnk_best or r["tflops"] > mnk_best[key]["tflops"]:
            mnk_best[key] = r
    for key in sorted(mnk_best):
        r = mnk_best[key]
        lbl = kernel_config_label(r)
        print(f"  M={key[0]:5d} N={key[1]:6d} K={key[2]:6d}: {r['tflops']:7.1f} TFLOPS  [{lbl}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="1000-point GEMM+AllScatter roofline sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_ranks", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results/roofline_1000pt")
    parser.add_argument("--chart_only", action="store_true")
    parser.add_argument(
        "--chart_path", type=str, default=None, help="Override output path for the PNG. Default: next to output_dir."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    chart_path = args.chart_path or os.path.join(
        os.path.dirname(os.path.abspath(args.output_dir)),
        "gemm_all_scatter_roofline_1000pt_mi300x.png",
    )

    if not args.chart_only:
        world_size = args.num_ranks
        total_sms = torch.cuda.get_device_properties(0).multi_processor_count if torch.cuda.is_available() else 304
        configs = build_sweep_configs(total_sms=total_sms, world_size=world_size)
        print(f"Total configs to sweep: {len(configs)}")

        init_url = "tcp://127.0.0.1:18189"
        mp.start_processes(
            worker,
            args=(world_size, init_url, configs, args.output_dir),
            nprocs=world_size,
            start_method="spawn",
        )

    generate_chart(args.output_dir, chart_path, world_size=args.num_ranks)


if __name__ == "__main__":
    main()
