#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""K-810 phase-decomposition benchmark.

Compares per-iter latency of:
    * iris baseline two-shot all-reduce              (one launch per call)
    * iris persistent burst two-shot, no barrier     (one launch for ``iters``
                                                      calls; CONSTANT-INPUT
                                                      MICROBENCH ONLY)
    * iris persistent burst two-shot, with barrier   (correct general-purpose
                                                      configuration — pays a
                                                      per-iter cross-rank
                                                      counter barrier)
    * RCCL all-reduce                                (torch.distributed)

across several payload sizes.  Mirrors the K-782 harness contract:
    * 200 warmup iters
    * 1000 timed iters
    * outputs JSON to ``output/persistent_bench_ws<W>_<RUN>.json``

The ``no-barrier`` variant exposes the raw launch-overhead amortisation but is
ONLY correct when the peer inputs are constant across iterations (this
benchmark fills the input once and reuses it).  Production callers should use
the with-barrier variant; both numbers are reported so the trade-off is
explicit.

Run with:
    torchrun --nproc_per_node=<W> --standalone bench_persistent.py \
        --output output/persistent_bench_ws<W>_<RUN>.json
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

import iris
from iris.ccl import Config


SIZES = [
    ("1KB", 256),  # 256 fp32 = 1KB
    ("4KB", 1024),
    ("16KB", 4096),
    ("64KB", 16384),
    ("256KB", 65536),
    ("1MB", 262144),
]


def dev_us():
    """Return a CUDA-synced perf-counter time in microseconds."""
    torch.cuda.synchronize()
    return time.perf_counter() * 1e6


def time_iris_baseline(ctx, output, input, config, iters):
    """Per-call all_reduce with explicit cross-rank barrier per iter."""
    workspace = ctx.ccl.all_reduce_preamble(output, input, config=config)
    ctx.barrier()
    torch.cuda.synchronize()
    t0 = dev_us()
    for _ in range(iters):
        ctx.ccl.all_reduce(output, input, config=config, workspace=workspace, async_op=True)
    torch.cuda.synchronize()
    ctx.barrier()
    t1 = dev_us()
    return (t1 - t0) / iters


def time_iris_persistent_burst(ctx, output, input, config, iters, use_barrier):
    """Single launch — ``iters`` reductions back-to-back.

    ``use_barrier`` mirrors the same kwarg in the public API: if False, the
    kernel skips the per-iter cross-rank flag barrier (only safe when input is
    constant across iters — fine for this latency microbench).  This is the
    apples-to-apples comparison vs the per-call baseline (which also skips
    the per-iter ``ctx.barrier()`` via ``async_op=True``).
    """
    ctx.barrier()
    torch.cuda.synchronize()
    t0 = dev_us()
    ctx.ccl.all_reduce_persistent_burst(
        output,
        input,
        num_iters=iters,
        config=config,
        async_op=True,
        use_barrier=use_barrier,
    )
    torch.cuda.synchronize()
    ctx.barrier()
    t1 = dev_us()
    return (t1 - t0) / iters


def time_rccl(output, input, iters):
    dist.barrier()
    torch.cuda.synchronize()
    t0 = dev_us()
    for _ in range(iters):
        dist.all_reduce(output, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()
    dist.barrier()
    t1 = dev_us()
    return (t1 - t0) / iters


def measure_one_size(ctx, label, n_elems, world_size, warmup, iters):
    """Return per-iter latency in µs for each backend at this payload size."""
    M = 1
    N = n_elems
    dtype = torch.float32

    iris_in = ctx.zeros((M, N), dtype=dtype)
    iris_in.fill_(float(ctx.get_rank() + 1))
    iris_out = ctx.zeros((M, N), dtype=dtype)

    rccl_in = torch.empty((M, N), dtype=dtype, device=f"cuda:{ctx.get_rank()}")
    rccl_in.copy_(iris_in)
    rccl_out = rccl_in.clone()

    config = Config(
        all_reduce_variant="two_shot",
        block_size_m=1,
        block_size_n=min(64, N),
    )

    # ---- warmup --------------------------------------------------------
    workspace = ctx.ccl.all_reduce_preamble(iris_out, iris_in, config=config)
    ctx.barrier()
    for _ in range(warmup):
        ctx.ccl.all_reduce(iris_out, iris_in, config=config, workspace=workspace, async_op=True)
    torch.cuda.synchronize()
    ctx.barrier()

    # warmup persistent burst (compile)
    ctx.ccl.all_reduce_persistent_burst(iris_out, iris_in, num_iters=min(8, warmup), config=config, async_op=True)
    torch.cuda.synchronize()

    # warmup rccl
    for _ in range(warmup):
        dist.all_reduce(rccl_out, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()

    # ---- measurements --------------------------------------------------
    iris_baseline_us = time_iris_baseline(ctx, iris_out, iris_in, config, iters)
    iris_burst_nobar_us = time_iris_persistent_burst(ctx, iris_out, iris_in, config, iters, use_barrier=False)
    iris_burst_bar_us = time_iris_persistent_burst(ctx, iris_out, iris_in, config, iters, use_barrier=True)
    rccl_us = time_rccl(rccl_out, rccl_in, iters)

    bytes_per_elem = dtype.itemsize if hasattr(dtype, "itemsize") else 4
    return {
        "size": label,
        "elems": n_elems,
        "bytes": n_elems * bytes_per_elem,
        "world_size": world_size,
        "iris_baseline_us": iris_baseline_us,
        # CONSTANT-INPUT MICROBENCH ONLY — exposes the raw launch-overhead
        # amortisation by skipping the per-iter cross-rank barrier.  Not
        # safe when peer inputs change between iters.
        "iris_persistent_burst_no_barrier_us": iris_burst_nobar_us,
        # General-purpose configuration — per-iter cross-rank barrier inside
        # the kernel.  Use this number to compare against RCCL for the
        # "shippable" persistent fast-path.
        "iris_persistent_burst_with_barrier_us": iris_burst_bar_us,
        "rccl_us": rccl_us,
        # Headline speedup uses the with-barrier (general-purpose) variant.
        "burst_speedup_vs_baseline": (iris_baseline_us / iris_burst_bar_us if iris_burst_bar_us > 0 else 0),
        # Microbench-only speedup, included for completeness.
        "burst_no_barrier_speedup_vs_baseline": (
            iris_baseline_us / iris_burst_nobar_us if iris_burst_nobar_us > 0 else 0
        ),
        "burst_gap_us_vs_rccl": iris_burst_bar_us - rccl_us,
        "baseline_gap_us_vs_rccl": iris_baseline_us - rccl_us,
        "burst_gap_closure_pct": (
            100.0 * (iris_baseline_us - iris_burst_bar_us) / max(iris_baseline_us - rccl_us, 1e-6)
            if iris_baseline_us > rccl_us
            else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--heap_size", type=int, default=1 << 31)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    ctx = iris.iris(heap_size=args.heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    rows = []
    for label, n_elems in SIZES:
        try:
            row = measure_one_size(ctx, label, n_elems, world_size, args.warmup, args.iters)
        except Exception as e:  # noqa: BLE001
            row = {
                "size": label,
                "elems": n_elems,
                "world_size": world_size,
                "error": str(e),
            }
        if rank == 0:
            print(json.dumps(row))
        rows.append(row)

    if rank == 0:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"world_size": world_size, "results": rows}, f, indent=2)

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
