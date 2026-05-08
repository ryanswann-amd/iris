#!/usr/bin/env python3
"""K-679 — rocprofv3 --kernel-trace harness.

The PRD asked for rocprofv3 PC sampling for bucket attribution. The
rocprofiler-sdk shipped with the K-679 baseline image
(rocm/pytorch:rocm7_ubuntu24.04_py3.12_pytorch_release_2.10.0,
rocprofv3 1.1.0 / sdk fc0010cf) does NOT expose --pc-sampling. The
closest empirical attribution available is --kernel-trace, which gives
exact GPU-side kernel start/end timestamps. We use that to:

  - measure the empirical AR-kernel duration (the (c)+(d) lump on the
    GPU side: xgmi_transfer + local_reduction execute concurrently
    inside the same Triton/RCCL kernel and cannot be split without PC
    sampling or ATT)
  - measure the device_barrier kernel duration (iris path) -> validates
    bucket (b) GPU side
  - cross-check the cudaEvent.elapsed_time() the wall-clock loop uses

This script runs ONE measured cell per (variant, size) at small iter
count (--iters 50) under rocprofv3, so the trace is small and fast.

Usage (within docker exec on a granted node):
  torchrun --nproc_per_node=8 \
    rocprofv3 --kernel-trace --output-format csv \
              --output-file kernels --output-directory <DIR>/<cell> \
              -- python3 bench_kernel_trace.py \
                 --variant one_shot --size 1024 --iters 50

But rocprofv3 must wrap each rank, so we use a shim: the LAUNCHER
(scripts/_rocprof_run.sh) wraps the python invocation per rank.
"""
import argparse
import datetime as dt
import json
import os
import time

import torch
import torch.distributed as dist


def now():
    return dt.datetime.utcnow().isoformat(timespec="seconds")


def init_dist():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["rccl", "one_shot", "two_shot"])
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--heap_gb", type=int, default=4)
    args = ap.parse_args()

    rank, world = init_dist()
    if rank == 0:
        print(f"[K-679 ktrace {now()}] variant={args.variant} size={args.size} "
              f"warmup={args.warmup} iters={args.iters}", flush=True)

    dtype = torch.bfloat16
    elem_size = torch.tensor([], dtype=dtype).element_size()
    total = args.size // elem_size
    if total % world:
        total = ((total // world) + 1) * world
    M, N = world, total // world

    if args.variant in ("one_shot", "two_shot"):
        import iris
        from iris.ccl import Config
        from iris.ccl.utils import extract_group_info
        from iris.ccl.triton.all_reduce import launch as launch_kernel
        from iris.ccl.all_reduce import all_reduce_preamble
        ctx = iris.iris(heap_size=args.heap_gb << 30)
        cfg = Config(
            block_size_m=32, block_size_n=64,
            all_reduce_variant=args.variant,
            all_reduce_distribution=1,
            comm_sms=64, num_warps=8, num_stages=1,
        )
        inp = ctx.zeros((M, N), dtype=dtype); inp.fill_(float(rank + 1))
        out = ctx.zeros((M, N), dtype=dtype)
        ws = all_reduce_preamble(out, inp, ctx, config=cfg, workspace=None)
        ws.prepared = True

        def call():
            nonlocal ws
            rg, rgb, ws_, rs, rst = extract_group_info(None, ctx)
            ws = launch_kernel(out, inp, ctx, rg, rgb, ws_, rs, rst, cfg, ws, group=None)
            ctx.device_barrier(group=None)
    else:
        inp = torch.full((M, N), float(rank + 1), device='cuda', dtype=dtype)

        def call():
            dist.all_reduce(inp, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()

    for _ in range(args.warmup):
        call()
    dist.barrier()
    torch.cuda.synchronize()

    # Measured kernels — rocprofv3 will trace these.
    t_start = time.perf_counter_ns()
    for _ in range(args.iters):
        call()
    torch.cuda.synchronize()
    t_end = time.perf_counter_ns()

    if rank == 0:
        wall_per_iter = (t_end - t_start) / args.iters
        print(f"[K-679 ktrace {now()}] DONE wall/iter={wall_per_iter:.0f}ns "
              f"({args.iters} iters)", flush=True)

    dist.barrier()


if __name__ == "__main__":
    main()
