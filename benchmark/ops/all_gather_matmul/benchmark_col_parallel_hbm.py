#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for column-parallel fused all-gather + GEMM using the HBM buffer kernel.

Column-parallel pattern:
  - A is M-sharded: each GPU has A_local[M/world_size, K]
  - B is N-sharded: each GPU has B_local[K, N/world_size]
  - Fused kernel gathers A along dim=0 (M) into HBM staging buffer,
    then computes C_local[M, N/world_size] = A_full[M, K] @ B_local[K, N/world_size]

Usage with torchrun:
    torchrun --nproc_per_node=8 benchmark/ops/all_gather_matmul/benchmark_col_parallel_hbm.py \\
        -m 262144 -n 8192 -k 8192 -v --benchmark

    torchrun --nproc_per_node=8 benchmark/ops/all_gather_matmul/benchmark_col_parallel_hbm.py \\
        -m 262144 -n 8192 -k 8192 --benchmark --benchmark_pytorch --no-trace
"""

import os
import time
import torch
import torch.distributed as dist
import random
import argparse
import numpy as np

import iris
from iris.ops.all_gather_matmul_col_parallel import (
    all_gather_matmul_col_parallel,
    all_gather_matmul_col_parallel_preamble,
)
from iris.ops import FusedConfig

torch.manual_seed(123)
random.seed(123)

TICKS_PER_US = 100  # s_memrealtime runs at 100 MHz


_FALLBACK_DEFAULTS = {
    "block_size_m": 256,
    "block_size_n": 128,
    "block_size_k": 64,
    "group_size_m": 8,
    "k_per_flag": 128,
    "num_fetch_stages": 1,
}


def _plot_trace(trace_data, output_path, rank, M, N_local, K, num_fetch_sms_cfg):
    """Generate a Gantt chart showing per-workgroup activity over time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    starts = trace_data["start"].numpy().astype(np.int64)
    ends = trace_data["end"].numpy().astype(np.int64)
    waits = trace_data["wait"].numpy().astype(np.int64)
    xcds = trace_data["xcd"].numpy().astype(np.int32)
    grid_size = trace_data["grid_size"]
    n_fetch_per_stage = trace_data["num_fetch_sms"]
    n_stages = trace_data.get("num_fetch_stages", 1)
    total_fetch = trace_data.get("total_fetch_wgs", n_fetch_per_stage)
    first_stage_fetch = trace_data.get("first_stage_fetch_sms", n_fetch_per_stage)
    first_stage_size = trace_data.get("first_stage_size", grid_size)
    rest_stage_size = trace_data.get("rest_stage_size", grid_size)

    t_min = starts.min()
    starts_us = (starts - t_min) / TICKS_PER_US
    ends_us = (ends - t_min) / TICKS_PER_US
    waits_us = waits / TICKS_PER_US

    roles = np.empty(grid_size, dtype=np.int32)
    for i in range(grid_size):
        if i < first_stage_size:
            stage = 0
            local = i
            fetch_thresh = first_stage_fetch
        else:
            adjusted = i - first_stage_size
            stage = 1 + adjusted // rest_stage_size
            local = adjusted % rest_stage_size
            fetch_thresh = n_fetch_per_stage
        if local < fetch_thresh:
            roles[i] = stage
        else:
            roles[i] = n_stages

    order = np.argsort(starts_us)

    row_h = 0.012
    fig_h = max(12, grid_size * row_h + 2)
    fig, ax = plt.subplots(figsize=(18, fig_h))

    fetch_blues = ["#1565C0", "#42A5F5", "#90CAF9", "#BBDEFB"]
    wait_color = "#F44336"
    compute_color = "#4CAF50"

    for y_idx, wg_idx in enumerate(order):
        s = starts_us[wg_idx]
        e = ends_us[wg_idx]
        dur = e - s
        role = roles[wg_idx]

        if role < n_stages:
            c = fetch_blues[role % len(fetch_blues)]
            ax.barh(y_idx, dur, left=s, height=0.8, color=c, edgecolor="none", linewidth=0)
        else:
            w = waits_us[wg_idx]
            comp = max(0, dur - w)
            ax.barh(y_idx, w, left=s, height=0.8, color=wait_color, edgecolor="none", linewidth=0)
            ax.barh(y_idx, comp, left=s + w, height=0.8, color=compute_color, edgecolor="none", linewidth=0)

    x_max = ends_us.max() * 1.02
    n_gemm = grid_size - total_fetch
    ax.set_xlabel("Time (us)", fontsize=12)
    ax.set_ylabel("Workgroup (sorted by start time)", fontsize=12)
    ax.set_title(
        f"Rank {rank}  |  Col-Parallel AG+GEMM Trace  |  "
        f"M={M} N_local={N_local} K={K}  |  "
        f"{total_fetch} fetchers + {n_gemm} GEMM workgroups",
        fontsize=13,
    )
    ax.set_ylim(-1, grid_size + 1)
    ax.set_xlim(0, x_max)
    ax.invert_yaxis()

    legend_elements = []
    for s_idx in range(min(n_stages, len(fetch_blues))):
        legend_elements.append(Line2D([0], [0], color=fetch_blues[s_idx], lw=6, label=f"Fetch stage {s_idx}"))
    legend_elements.append(Line2D([0], [0], color=wait_color, lw=6, label="GEMM: waiting on data"))
    legend_elements.append(Line2D([0], [0], color=compute_color, lw=6, label="GEMM: compute"))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    fetch_mask = roles < n_stages
    gemm_mask = roles == n_stages
    gemm_dur = (ends_us - starts_us)[gemm_mask]
    gemm_wait = waits_us[gemm_mask]
    gemm_compute = gemm_dur - gemm_wait

    stats_lines = []
    for s_idx in range(n_stages):
        s_mask = roles == s_idx
        s_dur = (ends_us - starts_us)[s_mask]
        s_start = starts_us[s_mask]
        if len(s_dur) > 0:
            stats_lines.append(
                f"Fetch stg{s_idx}: {s_dur.mean():.1f} us avg  "
                f"({s_dur.min():.1f}-{s_dur.max():.1f})  "
                f"first@{s_start.min():.0f}us"
            )
    stats_lines += [
        f"GEMM total: {gemm_dur.mean():.1f} us avg  ({gemm_dur.min():.1f}-{gemm_dur.max():.1f})",
        f"  wait: {gemm_wait.mean():.1f} us avg  ({gemm_wait.min():.1f}-{gemm_wait.max():.1f})",
        f"  compute: {gemm_compute.mean():.1f} us avg  ({gemm_compute.min():.1f}-{gemm_compute.max():.1f})",
        f"  wait%: {100 * gemm_wait.sum() / gemm_dur.sum():.1f}%",
        f"Wall time: {ends_us.max():.1f} us",
    ]
    stats_text = "\n".join(stats_lines)
    ax.text(
        0.01, 0.99, stats_text,
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Rank {rank}] Trace plot saved to: {output_path}")
    print(f"  {stats_text}")


def _plot_multi_gpu_trace(all_trace_data, output_path, M, N_local, K):
    """Generate a combined Gantt chart showing all ranks' workgroup activity on one timeline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    wait_color = "#F44336"
    compute_color = "#4CAF50"
    fetch_blues = ["#1565C0", "#42A5F5", "#90CAF9", "#BBDEFB"]

    # Find global time origin across all ranks
    global_t_min = min(td["start"].numpy().astype(np.int64).min() for td in all_trace_data.values())

    # Build per-rank data: list of (rank, grid_size, starts_us, ends_us, waits_us, roles, n_stages)
    rank_infos = []
    for r in sorted(all_trace_data.keys()):
        td = all_trace_data[r]
        starts = td["start"].numpy().astype(np.int64)
        ends = td["end"].numpy().astype(np.int64)
        waits = td["wait"].numpy().astype(np.int64)
        grid_size = td["grid_size"]
        n_fetch_per_stage = td["num_fetch_sms"]
        n_stages = td.get("num_fetch_stages", 1)
        total_fetch = td.get("total_fetch_wgs", n_fetch_per_stage)
        first_stage_fetch = td.get("first_stage_fetch_sms", n_fetch_per_stage)
        first_stage_size = td.get("first_stage_size", grid_size)
        rest_stage_size = td.get("rest_stage_size", grid_size)

        starts_us = (starts - global_t_min) / TICKS_PER_US
        ends_us = (ends - global_t_min) / TICKS_PER_US
        waits_us = waits / TICKS_PER_US

        roles = np.empty(grid_size, dtype=np.int32)
        for i in range(grid_size):
            if i < first_stage_size:
                stage = 0
                local = i
                fetch_thresh = first_stage_fetch
            else:
                adjusted = i - first_stage_size
                stage = 1 + adjusted // rest_stage_size
                local = adjusted % rest_stage_size
                fetch_thresh = n_fetch_per_stage
            if local < fetch_thresh:
                roles[i] = stage
            else:
                roles[i] = n_stages

        order = np.argsort(starts_us)
        rank_infos.append((r, grid_size, starts_us, ends_us, waits_us, roles, n_stages, total_fetch, order))

    # Layout: ranks stacked vertically with 2-row gap between them
    gap = 2
    total_rows = sum(ri[1] for ri in rank_infos) + gap * (len(rank_infos) - 1)
    row_h = 0.012
    fig_h = max(14, total_rows * row_h + 3)
    fig, ax = plt.subplots(figsize=(22, fig_h))

    y_offset = 0
    rank_label_positions = []  # (y_center, rank)

    for r, grid_size, starts_us, ends_us, waits_us, roles, n_stages, total_fetch, order in rank_infos:
        rank_label_positions.append((y_offset + grid_size / 2, r))

        for y_idx, wg_idx in enumerate(order):
            y = y_offset + y_idx
            s = starts_us[wg_idx]
            e = ends_us[wg_idx]
            dur = e - s
            role = roles[wg_idx]

            if role < n_stages:
                c = fetch_blues[role % len(fetch_blues)]
                ax.barh(y, dur, left=s, height=0.8, color=c, edgecolor="none", linewidth=0)
            else:
                w = waits_us[wg_idx]
                comp = max(0, dur - w)
                ax.barh(y, w, left=s, height=0.8, color=wait_color, edgecolor="none", linewidth=0)
                ax.barh(y, comp, left=s + w, height=0.8, color=compute_color, edgecolor="none", linewidth=0)

        # Draw horizontal separator line after this rank (except the last)
        if r != rank_infos[-1][0]:
            sep_y = y_offset + grid_size + gap / 2
            ax.axhline(y=sep_y, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

        y_offset += grid_size + gap

    # Global x_max
    x_max = max(ri[3].max() for ri in rank_infos) * 1.02

    ax.set_xlabel("Time (us)", fontsize=12)
    ax.set_ylabel("Workgroups by Rank", fontsize=12)
    ax.set_title(
        f"Multi-GPU Col-Parallel AG+GEMM Trace  |  "
        f"M={M} N_local={N_local} K={K}  |  {len(all_trace_data)} ranks",
        fontsize=14,
    )
    ax.set_ylim(-1, total_rows + 1)
    ax.set_xlim(0, x_max)
    ax.invert_yaxis()

    # Rank labels on Y-axis
    ax.set_yticks([pos for pos, _ in rank_label_positions])
    ax.set_yticklabels([f"Rank {r}" for _, r in rank_label_positions], fontsize=10, fontweight="bold")

    # Legend
    max_stages = max(ri[6] for ri in rank_infos)
    legend_elements = []
    for s_idx in range(min(max_stages, len(fetch_blues))):
        legend_elements.append(Line2D([0], [0], color=fetch_blues[s_idx], lw=6, label=f"Fetch stage {s_idx}"))
    legend_elements.append(Line2D([0], [0], color=wait_color, lw=6, label="GEMM: waiting on data"))
    legend_elements.append(Line2D([0], [0], color=compute_color, lw=6, label="GEMM: compute"))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    # Per-rank wall time summary
    summary_lines = []
    for r, grid_size, starts_us, ends_us, waits_us, roles, n_stages, total_fetch, order in rank_infos:
        wall = ends_us.max() - starts_us.min()
        gemm_mask = roles == n_stages
        gemm_wait_pct = 100 * waits_us[gemm_mask].sum() / (ends_us - starts_us)[gemm_mask].sum() if gemm_mask.any() else 0
        summary_lines.append(f"Rank {r}: wall={wall:.0f}us  wait%={gemm_wait_pct:.1f}%  fetch={total_fetch} GEMM={grid_size - total_fetch}")
    summary_text = "\n".join(summary_lines)
    ax.text(
        0.01, 0.99, summary_text,
        transform=ax.transAxes, fontsize=8, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Rank 0] Multi-GPU trace plot saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark column-parallel fused AG+GEMM (HBM buffer, M-sharded A).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=262144, help="M dimension (total)")
    parser.add_argument("-n", type=int, default=8192, help="N dimension (total)")
    parser.add_argument("-k", type=int, default=8192, help="K dimension")
    parser.add_argument("-v", "--validate", action="store_true", help="Validate correctness")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument(
        "--datatype", type=str, default="fp16",
        choices=["fp16", "fp32", "bf16"], help="Tensor datatype",
    )
    parser.add_argument("--heap_size", type=int, default=1 << 34, help="Iris heap size")
    parser.add_argument("--comm_sms", type=int, default=None, help="Number of SMs (auto if None)")
    parser.add_argument(
        "--benchmark_pytorch", action="store_true",
        help="Also benchmark PyTorch (all_gather_into_tensor + matmul)",
    )
    parser.add_argument("--block_size_m", type=int, default=None, help="Block size M")
    parser.add_argument("--block_size_n", type=int, default=None, help="Block size N")
    parser.add_argument("--block_size_k", type=int, default=None, help="Block size K")
    parser.add_argument("--group_size_m", type=int, default=None, help="Group size M")
    parser.add_argument("--num_xcds", type=int, default=None, help="Number of XCDs (auto if None)")
    parser.add_argument("--single-run", action="store_true", help="1 iteration (for profiling)")
    parser.add_argument("--num_fetch_sms", type=int, default=None, help="Fetcher SMs (auto if None)")
    parser.add_argument("--k_per_flag", type=int, default=None, help="K-blocks per ready flag")
    parser.add_argument("--num_warps", type=int, default=None, help="Triton num_warps (auto if None)")
    parser.add_argument("--num_stages", type=int, default=None, help="Triton num_stages (auto if None)")
    parser.add_argument("--num_fetch_stages", type=int, default=None, help="Number of fetch stages")
    parser.add_argument(
        "--first_stage_fetch_sms", type=int, default=None,
        help="Fetcher WGs for stage 0 (defaults to num_fetch_sms)",
    )
    parser.add_argument(
        "--fetch_pipe_depth", type=int, default=4,
        help="Fetcher software pipeline depth (1-4 XGMI loads in flight)",
    )
    parser.add_argument(
        "--trace", action=argparse.BooleanOptionalAction, default=True,
        help="Collect per-workgroup trace and save Gantt chart PNG",
    )
    parser.add_argument("--trace_output", type=str, default="trace_col_parallel.png", help="Trace output path")
    parser.add_argument(
        "--multi-gpu-trace", action=argparse.BooleanOptionalAction, default=True,
        help="Create combined multi-GPU trace visualization (default: True when --trace is used)",
    )
    parser.add_argument(
        "--split_kernels", action=argparse.BooleanOptionalAction, default=True,
        help="Use split fetch/GEMM kernels on separate streams (default: True)",
    )
    parser.add_argument(
        "--gemm_sms", type=int, default=None,
        help="Number of SMs for GEMM kernel in split mode (default: total - fetch_sms)",
    )
    parser.add_argument(
        "--gemm_wgs", type=int, default=None,
        help="Number of persistent GEMM WGs in fused mode (default: one per tile)",
    )
    return vars(parser.parse_args())


def _worker(args):
    """Worker function for torchrun."""
    local_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world_size_env = int(os.environ.get("WORLD_SIZE", 1))

    t0 = time.perf_counter()

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            device_id=torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None,
        )
    else:
        dist.init_process_group(
            backend=backend,
            init_method="tcp://127.0.0.1:29530",
            world_size=world_size_env,
            rank=local_rank,
            device_id=torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None,
        )

    t1 = time.perf_counter()

    shmem = iris.iris(args["heap_size"])
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    t2 = time.perf_counter()
    shmem.info(f"Startup: dist.init={t1 - t0:.1f}s, iris.init={t2 - t1:.1f}s, total={t2 - t0:.1f}s")

    datatype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    datatype = datatype_map.get(args["datatype"], torch.float16)
    dtype_bytes = torch.tensor([], dtype=datatype).element_size()

    # Apply defaults for any None parameters
    for name, fallback in _FALLBACK_DEFAULTS.items():
        if args.get(name) is None:
            args[name] = fallback

    M = args["m"]
    N = args["n"]
    K = args["k"]
    M_local = M // world_size
    N_local = N // world_size

    config_kwargs = {
        "block_size_m": args["block_size_m"],
        "block_size_n": args["block_size_n"],
        "block_size_k": args["block_size_k"],
        "group_size_m": args["group_size_m"],
    }
    if args["comm_sms"] is not None:
        config_kwargs["num_sms"] = args["comm_sms"]
    if args["num_xcds"] is not None:
        config_kwargs["num_xcds"] = args["num_xcds"]
    config = FusedConfig(**config_kwargs)

    buffer_mb = M * K * dtype_bytes / (1024**2)
    num_m_tiles = M // config.block_size_m
    num_k_blocks = K // config.block_size_k
    shmem.info(
        f"Col-Parallel HBM-Buffer: M={M} N={N} K={K} "
        f"M_local={M_local} N_local={N_local} "
        f"block=({config.block_size_m},{config.block_size_n},{config.block_size_k}) "
        f"buffer={buffer_mb:.0f}MB flags={num_m_tiles}x{num_k_blocks}"
    )

    # ── Allocate tensors ─────────────────────────────────────────────────
    # A_sharded: this GPU's M-shard [M_local, K]
    A_sharded = shmem.zeros((M_local, K), dtype=datatype)

    # B_local: this GPU's N-shard [K, N_local]
    B_local = shmem.zeros((K, N_local), dtype=datatype)

    # Output: [M, N_local]
    C = shmem.zeros((M, N_local), dtype=datatype)

    shmem.info(f"A_sharded={list(A_sharded.shape)} strides={A_sharded.stride()}, "
               f"B_local={list(B_local.shape)} strides={B_local.stride()}")

    # Fill with deterministic data
    torch.manual_seed(123 + rank)
    A_data = torch.randn((M_local, K), dtype=datatype, device=f"cuda:{rank}")
    A_sharded.copy_(A_data)

    torch.manual_seed(456 + rank)
    B_data = torch.randn((K, N_local), dtype=datatype, device=f"cuda:{rank}")
    B_local.copy_(B_data)

    # Expected result for validation
    expected_tensor = None
    if args["validate"]:
        # Gather A along M dimension (dim=0)
        A_list = [torch.zeros((M_local, K), dtype=datatype, device=f"cuda:{rank}") for _ in range(world_size)]
        dist.all_gather(A_list, A_data)
        A_gathered = torch.cat(A_list, dim=0)  # [M, K]
        expected_tensor = shmem.zeros((M, N_local), dtype=datatype)
        expected_tensor.copy_(torch.matmul(A_gathered, B_data))  # [M, K] @ [K, N_local] -> [M, N_local]

    # Pre-allocate workspace
    k_per_flag = args["k_per_flag"]
    workspace = all_gather_matmul_col_parallel_preamble(shmem, A_sharded, B_local, config, k_per_flag=k_per_flag)

    # ── Timing ───────────────────────────────────────────────────────────
    comm_stream = torch.cuda.Stream()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    num_experiments = 0
    all_iter_times = []

    num_fetch_sms = args["num_fetch_sms"]
    num_warps = args["num_warps"]
    num_stages = args["num_stages"]
    num_fetch_stages = args["num_fetch_stages"]
    first_stage_fetch_sms = args["first_stage_fetch_sms"]
    fetch_pipe_depth = args["fetch_pipe_depth"]
    split_kernels = args.get("split_kernels", True)
    gemm_sms = args.get("gemm_sms")
    gemm_wgs = args.get("gemm_wgs")

    shmem.info(f"Mode: {'split' if split_kernels else 'fused'} kernels"
               f"{f', fetch_sms={num_fetch_sms}, gemm_sms={gemm_sms}' if split_kernels else ''}")

    def run_experiment():
        nonlocal total_ms, num_experiments
        shmem.barrier()
        # Zero flags BEFORE timing — this is setup, not kernel work
        if workspace is not None:
            workspace.locks.zero_()
        torch.cuda.current_stream().synchronize()
        start_ev.record()
        all_gather_matmul_col_parallel(
            shmem,
            C,
            A_sharded,
            B_local,
            config=config,
            async_op=False,
            workspace=workspace,
            num_fetch_sms=num_fetch_sms,
            k_per_flag=k_per_flag,
            num_warps=num_warps,
            num_stages=num_stages,
            num_fetch_stages=num_fetch_stages,
            first_stage_fetch_sms=first_stage_fetch_sms,
            fetch_pipe_depth=fetch_pipe_depth,
            split_kernels=split_kernels,
            gemm_sms=gemm_sms,
            gemm_wgs=gemm_wgs,
        )
        end_ev.record()
        num_experiments += 1
        shmem.barrier()
        iter_ms = start_ev.elapsed_time(end_ev)
        total_ms += iter_ms
        all_iter_times.append(iter_ms)

    shmem.barrier()

    # ── Warmup (compile kernel) ──────────────────────────────────────────
    shmem.info("Warmup (compiling kernel)...")
    C.zero_()
    shmem.barrier()
    run_experiment()
    torch.cuda.synchronize()
    shmem.barrier()
    total_ms = 0.0
    num_experiments = 0

    # ── Validate ─────────────────────────────────────────────────────────
    if args["validate"]:
        shmem.info("Validating col-parallel fused AG+GEMM...")
        C.zero_()
        shmem.barrier()
        run_experiment()
        torch.cuda.synchronize()
        shmem.barrier()

        atol = 1e-1 if datatype == torch.float16 else 1e-3
        rtol = 1e-2 if datatype == torch.float16 else 1e-5
        success = torch.allclose(C, expected_tensor, atol=atol, rtol=rtol)
        if not success:
            max_diff = torch.abs(C - expected_tensor).max().item()
            shmem.error(f"Rank {rank}: Validation FAILED, max diff: {max_diff}")
        else:
            shmem.info("Validation PASSED!")
        shmem.barrier()

    # ── Benchmark ────────────────────────────────────────────────────────
    if args["benchmark"]:
        if args.get("single_run"):
            n_warmup, n_repeat = 0, 1
        else:
            n_warmup, n_repeat = 25, 100

        # Warmup
        total_ms = 0.0
        num_experiments = 0
        if n_warmup > 0:
            iris.do_bench(run_experiment, shmem.barrier, n_warmup=n_warmup, n_repeat=1, clear_l2=False)

        total_ms = 0.0
        num_experiments = 0
        C.zero_()
        shmem.barrier()

        all_iter_times.clear()
        iris.do_bench(run_experiment, shmem.barrier, n_warmup=0, n_repeat=n_repeat, clear_l2=False)
        avg_ms = total_ms / num_experiments if num_experiments > 0 else 0
        if all_iter_times:
            sorted_times = sorted(all_iter_times)
            median_ms = sorted_times[len(sorted_times) // 2]
            p10_ms = sorted_times[len(sorted_times) // 10] if len(sorted_times) >= 10 else sorted_times[0]
            p90_ms = sorted_times[9 * len(sorted_times) // 10] if len(sorted_times) >= 10 else sorted_times[-1]
            shmem.info(f"Timing: mean={avg_ms:.3f} median={median_ms:.3f} p10={p10_ms:.3f} p90={p90_ms:.3f} n={len(all_iter_times)}")
            avg_ms = median_ms  # Report median instead of mean

        # Per-GPU FLOPs: C_local[M, N_local] = A_full[M, K] @ B_local[K, N_local]
        per_gpu_flops = 2 * M * N_local * K
        tflops = (per_gpu_flops * 1e-12) / (avg_ms * 1e-3) if avg_ms > 0 else 0
        element_size = torch.tensor([], dtype=datatype).element_size()
        # AG transfer: each rank sends M_local*K to others
        ag_bytes = M_local * K * element_size * (world_size - 1)
        bw_gbps = (ag_bytes / (1024**3)) / (avg_ms * 1e-3) if avg_ms > 0 else 0

        shmem.info(
            f"Col-Parallel HBM-Buffer (M={M}, M_local={M_local}, K={K}, N_local={N_local}, "
            f"ws={world_size}, dtype={args['datatype']}): "
            f"{avg_ms:.3f} ms, {tflops:.3f} TFLOPS, AG_BW={bw_gbps:.3f} GB/s"
        )
        shmem.barrier()

        # ── Per-rank finish time measurement ─────────────────────────────
        shmem.barrier()
        torch.cuda.synchronize()
        dist.barrier()

        dist.barrier()
        t_start = time.perf_counter()

        all_gather_matmul_col_parallel(
            shmem,
            C,
            A_sharded,
            B_local,
            config=config,
            async_op=False,
            workspace=workspace,
            num_fetch_sms=num_fetch_sms,
            k_per_flag=k_per_flag,
            num_warps=num_warps,
            num_stages=num_stages,
            num_fetch_stages=num_fetch_stages,
            first_stage_fetch_sms=first_stage_fetch_sms,
            fetch_pipe_depth=fetch_pipe_depth,
            split_kernels=split_kernels,
            gemm_sms=gemm_sms,
            gemm_wgs=gemm_wgs,
        )
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        finish_ms = (t_end - t_start) * 1000.0

        finish_tensor = torch.tensor([finish_ms], dtype=torch.float64, device=f"cuda:{rank}")
        all_finish = [torch.zeros(1, dtype=torch.float64, device=f"cuda:{rank}") for _ in range(world_size)]
        dist.all_gather(all_finish, finish_tensor)

        if rank == 0:
            times = [t.item() for t in all_finish]
            min_t = min(times)
            max_t = max(times)
            print("\n  Per-rank finish times (single run):")
            print(f"  {'Rank':>6}  {'Finish ms':>10}  {'Delta ms':>10}")
            print(f"  {'-' * 30}")
            for r, t in enumerate(times):
                delta = t - min_t
                print(f"  {r:>6}  {t:>10.3f}  {delta:>+10.3f}")
            print(f"  {'-' * 30}")
            print(f"  Spread (max - min): {max_t - min_t:.3f} ms")
            print()

        shmem.barrier()

    # ── Trace ────────────────────────────────────────────────────────────
    if args["trace"]:
        shmem.info("Trace warmup (compiling traced kernel variant)...")
        C.zero_()
        workspace.locks.zero_()
        shmem.barrier()
        all_gather_matmul_col_parallel(
            shmem, C, A_sharded, B_local,
            config=config, async_op=False, workspace=workspace,
            num_fetch_sms=num_fetch_sms, k_per_flag=k_per_flag,
            num_warps=num_warps, num_stages=num_stages,
            num_fetch_stages=num_fetch_stages,
            first_stage_fetch_sms=first_stage_fetch_sms,
            fetch_pipe_depth=fetch_pipe_depth,
            trace=True,
            split_kernels=False,  # trace requires fused kernel
            gemm_wgs=gemm_wgs,
        )
        torch.cuda.synchronize()
        shmem.barrier()

        shmem.info("Running single traced iteration...")
        C.zero_()
        workspace.locks.zero_()
        shmem.barrier()

        all_gather_matmul_col_parallel(
            shmem, C, A_sharded, B_local,
            config=config, async_op=False, workspace=workspace,
            num_fetch_sms=num_fetch_sms, k_per_flag=k_per_flag,
            num_warps=num_warps, num_stages=num_stages,
            num_fetch_stages=num_fetch_stages,
            first_stage_fetch_sms=first_stage_fetch_sms,
            fetch_pipe_depth=fetch_pipe_depth,
            trace=True,
            split_kernels=False,  # trace requires fused kernel
            gemm_wgs=gemm_wgs,
        )
        torch.cuda.synchronize()
        shmem.barrier()

        # Save per-rank trace
        if hasattr(workspace, "trace_data"):
            trace_base = args.get("trace_output", "trace_col_parallel.png")
            trace_stem = trace_base.rsplit(".", 1)[0] if "." in trace_base else trace_base
            trace_ext = trace_base.rsplit(".", 1)[1] if "." in trace_base else "png"
            per_rank_path = f"{trace_stem}_rank{rank}.{trace_ext}"
            try:
                _plot_trace(workspace.trace_data, per_rank_path, rank, M, N_local, K, num_fetch_sms)
            except ImportError:
                print(f"  [Rank {rank}] (matplotlib not available -- skipping trace plot)")
            except Exception as e:
                print(f"  [Rank {rank}] (Trace plot failed: {e})")

        # Gather trace data from all ranks to rank 0 for combined plot
        shmem.barrier()

        if args.get("multi_gpu_trace", True) and hasattr(workspace, "trace_data"):
            # Serialize trace_data to a tensor for all_gather
            import pickle
            trace_bytes = pickle.dumps(workspace.trace_data)
            trace_tensor = torch.tensor(list(trace_bytes), dtype=torch.uint8, device=f"cuda:{rank}")
            # Gather sizes first so we can pad
            size_tensor = torch.tensor([len(trace_bytes)], dtype=torch.int64, device=f"cuda:{rank}")
            all_sizes = [torch.zeros(1, dtype=torch.int64, device=f"cuda:{rank}") for _ in range(world_size)]
            dist.all_gather(all_sizes, size_tensor)
            max_size = max(s.item() for s in all_sizes)
            # Pad to max_size
            padded = torch.zeros(max_size, dtype=torch.uint8, device=f"cuda:{rank}")
            padded[:len(trace_bytes)] = trace_tensor
            all_padded = [torch.zeros(max_size, dtype=torch.uint8, device=f"cuda:{rank}") for _ in range(world_size)]
            dist.all_gather(all_padded, padded)

            if rank == 0:
                all_trace_data = {}
                for r in range(world_size):
                    sz = all_sizes[r].item()
                    raw = bytes(all_padded[r][:sz].cpu().numpy().tolist())
                    all_trace_data[r] = pickle.loads(raw)
                multi_gpu_path = f"{trace_stem}_multi_gpu.{trace_ext}"
                try:
                    _plot_multi_gpu_trace(all_trace_data, multi_gpu_path, M, N_local, K)
                except ImportError:
                    print("  (matplotlib not available -- skipping multi-GPU trace plot)")
                except Exception as e:
                    print(f"  (Multi-GPU trace plot failed: {e})")

        shmem.barrier()

    # ── PyTorch baseline ─────────────────────────────────────────────────
    if args["benchmark_pytorch"]:
        shmem.info("Benchmarking PyTorch (all_gather_into_tensor + matmul)...")

        pt_A = torch.randn(M_local, K, dtype=datatype, device=f"cuda:{rank}")
        pt_B = torch.randn(K, N_local, dtype=datatype, device=f"cuda:{rank}")
        pt_Ag = torch.zeros(M, K, dtype=datatype, device=f"cuda:{rank}")

        for _ in range(10):
            dist.all_gather_into_tensor(pt_Ag, pt_A)
            _ = torch.matmul(pt_Ag, pt_B)
        torch.cuda.synchronize()
        dist.barrier()

        def run_pt():
            dist.all_gather_into_tensor(pt_Ag, pt_A)
            _ = torch.matmul(pt_Ag, pt_B)

        per_gpu_flops = 2 * M * N_local * K
        element_size = torch.tensor([], dtype=datatype).element_size()
        ag_bytes = M_local * K * element_size * (world_size - 1)

        pt_ms = iris.do_bench(run_pt, dist.barrier)
        pt_tflops = (per_gpu_flops * 1e-12) / (pt_ms * 1e-3) if pt_ms > 0 else 0
        pt_bw = (ag_bytes / (1024**3)) / (pt_ms * 1e-3) if pt_ms > 0 else 0

        shmem.info(
            f"PyTorch (M={M}, M_local={M_local}, K={K}, N_local={N_local}, ws={world_size}, "
            f"dtype={args['datatype']}): "
            f"{pt_ms:.3f} ms, {pt_tflops:.3f} TFLOPS, AG_BW={pt_bw:.3f} GB/s"
        )

        if args["benchmark"]:
            avg_ms = total_ms / num_experiments if num_experiments > 0 else 0
            iris_tflops = (per_gpu_flops * 1e-12) / (avg_ms * 1e-3) if avg_ms > 0 else 0
            speedup = iris_tflops / pt_tflops if pt_tflops > 0 else 0
            shmem.info(f"Speedup (Col-Parallel HBM-Buffer / PyTorch): {speedup:.2f}x")

        shmem.barrier()

    shmem.barrier()
    dist.destroy_process_group()


def main():
    print("Starting col-parallel HBM-buffer AG+GEMM benchmark...")
    args = parse_args()
    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        _worker(args)
    else:
        print(
            "Please run with torchrun:\n"
            "  torchrun --nproc_per_node=N "
            "benchmark/ops/all_gather_matmul/benchmark_col_parallel_hbm.py [OPTIONS]"
        )


if __name__ == "__main__":
    main()
