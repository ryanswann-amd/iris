#!/usr/bin/env python3
"""K-679 v2 bench — 5-bucket latency decomposition for iris all_reduce
(one_shot / two_shot) and RCCL ncclAllReduce on 8x MI300X (c42), at
1KB / 4KB / 16KB bf16.

Changes from v1:
  - amdsmi snapshot path REMOVED (firmware-poll cadence undersamples
    sub-second bursts; the value was always dead). xGMI bucket is set
    purely from the analytical per-link floor (msg_bytes / div / 64GB/s)
    and is computed inside the aggregator, not at bench time.
  - per-rank JSONL output is PRE-AGGREGATED: emit (median, p50, p90,
    p99, mean, stdev, n_samples, clamp_count) per (variant, size, rank,
    bucket), instead of 2000 raw per-iter ns values per bucket.
    Cuts JSONL volume ~100x.
  - track local_reduction clamp_count (number of iters where the model
    overshoots and we clip to 0). Reviewer ask.
  - rocprofv3 empirical kernel duration is collected separately by
    bench_kernel_trace.py (sibling script) since rocprofv3 wraps the
    whole process and 2000-iter cells generate a few-hundred-MB trace
    we don't want for the wall-clock measurement loop.

Bucket definitions (per iter, ns), reconciled to the per-iter wall:
  (a) host_launch_ns      CPU `perf_counter` around the
                          launch_kernel(...) (iris) or dist.all_reduce(...)
                          (RCCL) call. Returns once enqueued.
  (b) device_barrier_ns   CPU `perf_counter` around ctx.device_barrier
                          (iris, K-402 atomic on-device barrier) or
                          torch.cuda.synchronize() (RCCL). Wall time the
                          host blocks until the GPU pipe drains.
  (c) xgmi_transfer_ns    Analytical floor only (computed in aggregator):
                          per-link bytes / 64 GB/s SOL.
                          (See K-664 for the per-(src,dst) byte counts;
                          we use those + iris source for per-link.)
  (d) local_reduction_ns  cudaEvent.elapsed_time(launch) - device_barrier_ns
                          - xgmi_transfer_ns (clipped >=0). The
                          AR-kernel GPU compute portion that overlaps
                          with device_barrier.
                          clamp_count tracks iters where this clipped.
  (e) epilogue_sync_ns    wall - (a) - (b) - (c) - (d), clipped >=0.
                          Captures any extra unaccounted sync.

Critical-path identity: wall_ns = host_launch + device_barrier (CPU iter
time). (c) and (d) execute on the GPU during the device_barrier wait and
are bounded by it; they are NOT additive on the wall axis.
"""
import argparse
import datetime as dt
import json
import os
import statistics
import sys
import time

import numpy as np
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


def make_shape_per_rank(per_rank_bytes, world, dtype):
    elem_size = torch.tensor([], dtype=dtype).element_size()
    total_elems = per_rank_bytes // elem_size
    if total_elems % world:
        total_elems = ((total_elems // world) + 1) * world
    M = world
    N = total_elems // world
    return M, N


XGMI_LINK_BW_GBPS = 64.0  # MI300X xGMI4 unidirectional


def analytical_xgmi_ns(variant, bytes_msg):
    """Per-iter modeled xGMI transfer floor on UBB MI300X (8 GPUs,
    7-peer mesh per GPU). The dominant per-link byte volume divided by
    64 GB/s SOL.

    one_shot : push msg_bytes to each peer over its dedicated link
               -> per-link write = msg_bytes (one-way).
    two_shot : reduce-scatter (msg/8 per peer) + all-gather (msg/8 per
               peer) over each link = msg/4.
    rccl     : LL128 / ring-style for small messages -> per-link
               ~= msg/8 (rough scattering bound).
    """
    if variant == "one_shot":
        bpl = float(bytes_msg)
    elif variant == "two_shot":
        bpl = float(bytes_msg) / 4.0
    elif variant == "rccl":
        bpl = float(bytes_msg) / 8.0
    else:
        bpl = float(bytes_msg)
    return bpl / (XGMI_LINK_BW_GBPS * 1e9) * 1e9


def percentiles(arr_np):
    """Return (med, p90, p99, mean, std, n)."""
    if arr_np.size == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0)
    return (
        float(np.median(arr_np)),
        float(np.quantile(arr_np, 0.90)),
        float(np.quantile(arr_np, 0.99)),
        float(np.mean(arr_np)),
        float(np.std(arr_np)),
        int(arr_np.size),
    )


def stat_dict(arr_np, prefix):
    med, p90, p99, mean, std, n = percentiles(arr_np)
    return {
        f"{prefix}_med": med,
        f"{prefix}_p90": p90,
        f"{prefix}_p99": p99,
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_n": n,
    }


def run_one_cell(rank, world, variant, size_bytes, warmup, iters,
                 ctx, group_handle, dtype=torch.bfloat16):
    """Run a single (variant, size) cell. Return per-rank summary dict
    with pre-aggregated bucket statistics (no per-iter arrays)."""
    M, N = make_shape_per_rank(size_bytes, world, dtype)
    elem_size = torch.tensor([], dtype=dtype).element_size()
    actual_bytes = M * N * elem_size

    if variant in ("one_shot", "two_shot"):
        from iris.ccl import Config
        from iris.ccl.utils import ReduceOp, extract_group_info
        from iris.ccl.triton.all_reduce import launch as launch_kernel
        cfg = Config(
            block_size_m=32, block_size_n=64,
            all_reduce_variant=variant,
            all_reduce_distribution=1,
            comm_sms=64, num_warps=8, num_stages=1,
        )
        inp = ctx.zeros((M, N), dtype=dtype); inp.fill_(float(rank + 1))
        out = ctx.zeros((M, N), dtype=dtype)
        workspace = None
        # Pre-prepare arena (K-482) once.
        from iris.ccl.all_reduce import all_reduce_preamble
        workspace = all_reduce_preamble(out, inp, ctx, config=cfg, workspace=workspace)
        workspace.prepared = True

        def call(rec_t):
            nonlocal workspace
            t0 = time.perf_counter_ns()
            rg, rgb, ws_, rs, rst = extract_group_info(None, ctx)
            workspace = launch_kernel(out, inp, ctx, rg, rgb, ws_, rs, rst,
                                      cfg, workspace, group=None)
            t1 = time.perf_counter_ns()
            ctx.device_barrier(group=None)
            t2 = time.perf_counter_ns()
            rec_t[0] = t0; rec_t[1] = t1; rec_t[2] = t2

    elif variant == "rccl":
        inp = torch.full((M, N), float(rank + 1), device='cuda', dtype=dtype)

        def call(rec_t):
            t0 = time.perf_counter_ns()
            dist.all_reduce(inp, op=dist.ReduceOp.SUM, group=group_handle)
            t1 = time.perf_counter_ns()
            torch.cuda.synchronize()
            t2 = time.perf_counter_ns()
            rec_t[0] = t0; rec_t[1] = t1; rec_t[2] = t2
    else:
        raise ValueError(f"unknown variant {variant}")

    # ---- warmup ----
    scratch = [0, 0, 0]
    for _ in range(warmup):
        call(scratch)

    # ---- measured loop ----
    # numpy arrays for per-iter timings
    t0_arr = np.zeros(iters, dtype=np.int64)
    t1_arr = np.zeros(iters, dtype=np.int64)
    t2_arr = np.zeros(iters, dtype=np.int64)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    rec_t = [0, 0, 0]
    for i in range(iters):
        starts[i].record()
        call(rec_t)
        stops[i].record()
        t0_arr[i] = rec_t[0]; t1_arr[i] = rec_t[1]; t2_arr[i] = rec_t[2]
    torch.cuda.synchronize()

    # cudaEvent kernel time (covers launch + device_barrier kernel for iris;
    # for RCCL covers the full ncclAllReduce kernel; sync runs after).
    event_us = np.array(
        [s.elapsed_time(e) * 1e3 for s, e in zip(starts, stops)],
        dtype=np.float64,
    )
    event_ns = (event_us * 1e3).astype(np.int64)

    # ---- correctness check (single synced call) ----
    expected = float(world * (world + 1) / 2.0)
    if variant == "rccl":
        inp.fill_(float(rank + 1))
        torch.cuda.synchronize()
        dist.all_reduce(inp, op=dist.ReduceOp.SUM, group=group_handle)
        torch.cuda.synchronize()
        actual = float(inp.flatten()[0].cpu().item())
    else:
        inp.fill_(float(rank + 1))
        out.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        call(rec_t)
        torch.cuda.synchronize()
        actual = float(out.flatten()[0].cpu().item())
    correct = abs(actual - expected) < 1e-3 * expected

    # ---- decompose per-iter ----
    host_launch = (t1_arr - t0_arr).astype(np.int64)
    device_barrier = (t2_arr - t1_arr).astype(np.int64)
    wall = host_launch + device_barrier

    xgmi_floor_ns = analytical_xgmi_ns(variant, actual_bytes)
    # local_reduction estimate: cudaEvent kernel total - device_barrier_ns
    # - xgmi_floor.
    raw_red = event_ns.astype(np.float64) - device_barrier.astype(np.float64) - xgmi_floor_ns
    clamp_count = int(np.sum(raw_red < 0))
    local_reduction = np.clip(raw_red, 0, None)

    # epilogue/sync residual
    epilogue_signed = (
        wall.astype(np.float64) - host_launch.astype(np.float64)
        - device_barrier.astype(np.float64) - xgmi_floor_ns - local_reduction
    )
    epilogue = np.clip(epilogue_signed, 0, None)

    rec = {
        "rank": rank, "world": world,
        "variant": variant, "bytes": int(actual_bytes),
        "M": int(M), "N": int(N),
        "warmup": warmup, "iters": iters,
        "correct": bool(correct),
        "expected": expected, "actual": actual,
        "xgmi_floor_ns": xgmi_floor_ns,
        "local_reduction_clamp_count": clamp_count,
        "local_reduction_clamp_pct": (clamp_count / max(1, iters)) * 100.0,
        "epilogue_negative_count": int(np.sum(epilogue_signed < 0)),
    }
    rec.update(stat_dict(host_launch, "host_launch_ns"))
    rec.update(stat_dict(device_barrier, "device_barrier_ns"))
    rec.update(stat_dict(wall, "wall_ns"))
    rec.update(stat_dict(event_ns, "event_total_ns"))
    rec.update(stat_dict(local_reduction.astype(np.int64), "local_reduction_ns"))
    rec.update(stat_dict(epilogue.astype(np.int64), "epilogue_sync_ns"))
    rec.update(stat_dict(epilogue_signed.astype(np.int64), "epilogue_signed_ns"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--sizes", default="1024,4096,16384")
    ap.add_argument("--variants", default="rccl,one_shot,two_shot")
    ap.add_argument("--heap_gb", type=int, default=4)
    args = ap.parse_args()

    rank, world = init_dist()
    os.makedirs(args.out_dir, exist_ok=True)
    if rank == 0:
        print(f"[K-679 v2 {now()}] world={world} run_id={args.run_id} "
              f"warmup={args.warmup} iters={args.iters}", flush=True)

    import iris
    ctx = iris.iris(heap_size=args.heap_gb << 30)
    group_handle = None

    sizes = [int(x) for x in args.sizes.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]

    out_path = os.path.join(args.out_dir, f"rank{rank}_{args.run_id}.jsonl")
    with open(out_path, "w") as f:
        for variant in variants:
            for size in sizes:
                if rank == 0:
                    print(f"[K-679 v2 {now()}] cell variant={variant} bytes={size}", flush=True)
                dist.barrier()
                rec = run_one_cell(rank, world, variant, size,
                                   args.warmup, args.iters, ctx, group_handle)
                rec["run_id"] = args.run_id
                rec["timestamp"] = now()
                f.write(json.dumps(rec) + "\n")
                f.flush()

    if rank == 0:
        print(f"[K-679 v2 {now()}] DONE rank0 -> {out_path}", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
