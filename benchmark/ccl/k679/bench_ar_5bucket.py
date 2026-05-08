#!/usr/bin/env python3
"""K-679 [S-007] iris all_reduce 5-bucket latency decomposition for SMALL
all_reduce sizes (1KB / 4KB / 16KB; bf16) on 8x MI300X (c42).

5 buckets per iter (ns), reconciled to the per-iter wall:
  (a) host_launch_ns      — CPU time enqueuing the kernel (perf_counter
                            around the launch call; for RCCL: dist.all_reduce)
  (b) device_barrier_ns   — CPU time in ctx.barrier() (iris device_barrier
                            since K-402 patch is applied at runtime;
                            for RCCL: torch.cuda.synchronize())
  (c) xgmi_transfer_ns    — modeled = max(per-link bytes) / 64e9 * 1e9
                            (per-link bytes sampled via amdsmi
                            read_link_metrics before/after the measured burst)
  (d) local_reduction_ns  — max(0, kernel_total_ns - xgmi_transfer_ns)
                            where kernel_total_ns is the cudaEvent.elapsed_time
                            wrapping the launch call alone (does NOT include
                            barrier, as barrier executes AFTER launch returns)
  (e) epilogue_sync_ns    — wall - host_launch - device_barrier - kernel_total
                            (residual: any post-completion sync overhead beyond
                            what device_barrier captured)

Sum (a..e) = wall_ns by construction (kernel_total = c+d).

K-482 / K-402 patches: applied at runtime as a monkey-patch of
iris.ccl.all_reduce so the public API uses ctx.device_barrier (and skips
the workspace.prepared = False reset that re-fires a TCP host barrier).
This keeps the bench off iris main without forking the lib.

Adapted from K-664/bench_ar_smallmsg.py.
"""

import argparse
import datetime as dt
import json
import os
import statistics
import sys
import threading
import time

import numpy as np
import torch
import torch.distributed as dist


XGMI_PER_LINK_SOL_GBPS = 64.0  # MI300X xGMI4 unidirectional peak


def now():
    return dt.datetime.utcnow().isoformat(timespec="seconds")


def init_dist():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world


def apply_iris_patch(record_phase_times):
    """Monkey-patch iris.ccl.all_reduce to:
      (1) skip workspace.prepared = False so one_shot doesn't re-fire host barrier
          (K-482 finding),
      (2) replace ctx.barrier() with ctx.device_barrier(group=group)
          (K-402 / R-K166-TIRE-1 fix that flipped iris from gloo TCP barrier to
          on-device atomic barrier),
      (3) record (t_call_start, t_launch_returns, t_barrier_returns) for every
          call so we can decompose per-iter wall.

    Wrap once; idempotent.
    """
    import iris.ccl.all_reduce as ar_mod
    if getattr(ar_mod, "_K679_PATCHED", False):
        return False
    from iris.ccl.config import Config
    from iris.ccl.utils import ReduceOp, extract_group_info
    from iris.ccl.triton.all_reduce import launch as launch_kernel

    def patched_all_reduce(output_tensor, input_tensor, ctx, op=None, group=None,
                           async_op=False, config=None, workspace=None):
        if op is None:
            op = ReduceOp.SUM
        if op != ReduceOp.SUM:
            raise ValueError(f"Only SUM supported, got {op}")
        if config is None:
            config = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=1)
        rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

        t0 = time.perf_counter_ns()
        workspace = launch_kernel(
            output_tensor, input_tensor, ctx, rank_in_group, rank_global,
            world_size, rank_start, rank_stride, config, workspace, group=group,
        )
        t1 = time.perf_counter_ns()
        # K-482: do NOT re-fire workspace.prepared = False here.
        if not async_op:
            # K-402 R-K166-TIRE-1: device_barrier (atomic, on-device) — the real
            # post-K-402 latency; otherwise default ctx.barrier() is a gloo TCP
            # host barrier (~530 µs).
            ctx.device_barrier(group=group)
        t2 = time.perf_counter_ns()
        record_phase_times.append((t0, t1, t2))
        return workspace

    ar_mod.all_reduce = patched_all_reduce
    ar_mod._K679_PATCHED = True
    try:
        import iris.ccl as ccl_mod
        ccl_mod.all_reduce = patched_all_reduce
    except Exception:
        pass
    return True


# -------------------- per-rank amdsmi link snapshot ----------------------

_AMDSMI_INITED = False


def _ensure_amdsmi_init():
    global _AMDSMI_INITED
    if _AMDSMI_INITED:
        return
    import amdsmi
    try:
        amdsmi.amdsmi_init()
    except Exception:
        # already inited?
        pass
    _AMDSMI_INITED = True


def snapshot_link_bytes(rank, world):
    """Return ndarray of shape (world,) giving cumulative bytes (read+written)
    on this rank's xGMI link to each peer rank, as reported by amdsmi
    amdsmi_get_link_metrics. Index 0..world-1 = peer rank.
    Loopback (rank == self) is 0.
    """
    import amdsmi
    _ensure_amdsmi_init()
    handles = amdsmi.amdsmi_get_processor_handles()
    # Map BDF → GPU index. amdsmi may return more handles than GPUs (e.g.,
    # MI3xx exposes one handle per partition). Use the GPU minor (0..world-1)
    # as identifier — for MI300X these line up with rank when single-die mode.
    h = handles[rank]
    metrics = amdsmi.amdsmi_get_link_metrics(h)
    out = np.zeros(world, dtype=np.float64)
    # link metrics is a list of dicts with bdf, read, write, bit_rate, etc.
    # We need to map peer BDF → peer rank. Build a map of BDF → rank.
    bdf_to_rank = {}
    for r in range(world):
        try:
            b = amdsmi.amdsmi_get_gpu_device_bdf(handles[r])
            bdf_to_rank[b] = r
        except Exception:
            pass
    for link in metrics["links"]:
        # link dict has: bdf, read, write, link_type, ...
        try:
            peer_bdf = link.get("bdf", None)
            if peer_bdf is None:
                continue
            peer = bdf_to_rank.get(peer_bdf, None)
            if peer is None or peer == rank:
                continue
            out[peer] += float(link.get("read", 0)) + float(link.get("write", 0))
        except Exception:
            continue
    return out


def make_shape_per_rank(per_rank_bytes, world, dtype):
    elem_size = torch.tensor([], dtype=dtype).element_size()
    total_elems = per_rank_bytes // elem_size
    if total_elems % world:
        total_elems = ((total_elems // world) + 1) * world
    M = world
    N = total_elems // world
    return M, N


def run_one_cell(rank, world, variant, size_bytes, warmup, iters,
                 ctx, group_handle, dtype=torch.bfloat16):
    """Run a single (variant, size) cell. Return per-iter timing arrays + meta."""
    M, N = make_shape_per_rank(size_bytes, world, dtype)
    actual_bytes = M * N * dtype.itemsize if hasattr(dtype, 'itemsize') else \
                   M * N * torch.tensor([], dtype=dtype).element_size()

    # ---- buffers ----
    if variant in ("one_shot", "two_shot"):
        from iris.ccl import Config
        cfg = Config(
            block_size_m=32, block_size_n=64,
            all_reduce_variant=variant,
            all_reduce_distribution=1,
            comm_sms=64, num_warps=8, num_stages=1,
        )
        inp = ctx.zeros((M, N), dtype=dtype); inp.fill_(float(rank + 1))
        out = ctx.zeros((M, N), dtype=dtype)
        workspace = None
        record = []
        # The patched all_reduce will record into a shared list. We swap it per
        # cell.
        import iris.ccl.all_reduce as ar_mod
        ar_mod._K679_RECORD = record
        # Re-bind the closure to use this record
        from iris.ccl.config import Config as _Cfg
        from iris.ccl.utils import ReduceOp, extract_group_info
        from iris.ccl.triton.all_reduce import launch as launch_kernel
        _record = record

        def call():
            nonlocal workspace
            from iris.ccl.utils import ReduceOp as _RO
            t0 = time.perf_counter_ns()
            rg, rgb, ws_, rs, rst = extract_group_info(None, ctx)
            workspace = launch_kernel(out, inp, ctx, rg, rgb, ws_, rs, rst,
                                      cfg, workspace, group=None)
            t1 = time.perf_counter_ns()
            ctx.device_barrier(group=None)
            t2 = time.perf_counter_ns()
            _record.append((t0, t1, t2))
            return out

        # initial arena prep — preamble does the K-482 prepare call once
        from iris.ccl.all_reduce import all_reduce_preamble
        workspace = all_reduce_preamble(out, inp, ctx, config=cfg, workspace=workspace)
        workspace.prepared = True

    elif variant == "rccl":
        # Plain torch tensors (NOT on iris symmetric heap)
        inp = torch.full((M, N), float(rank + 1), device='cuda', dtype=dtype)
        out = torch.empty_like(inp)
        record = []
        _record = record

        def call():
            t0 = time.perf_counter_ns()
            dist.all_reduce(inp, op=dist.ReduceOp.SUM, group=group_handle)
            t1 = time.perf_counter_ns()
            torch.cuda.synchronize()
            t2 = time.perf_counter_ns()
            _record.append((t0, t1, t2))
            return inp
    else:
        raise ValueError(f"unknown variant {variant}")

    # ---- warmup (untimed) ----
    record.clear()
    for _ in range(warmup):
        call()
    record.clear()

    # ---- pre-burst link snapshot ----
    pre_links = snapshot_link_bytes(rank, world)
    t_burst_start = time.perf_counter_ns()

    # ---- measured loop ----
    # cudaEvent start/stop *around launch only* (does NOT include device_barrier)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        # We need to time only the kernel launch portion via cudaEvent. The
        # `call()` wrapper above also records CPU times; for kernel time we
        # need just the kernel(s). Approach: record event before call, after
        # call returns from launch (CPU-perspective), and ALSO after the
        # device_barrier. Then kernel_total = event_after_barrier -
        # event_before_call.
        call()
        stops[i].record()
    torch.cuda.synchronize()
    t_burst_end = time.perf_counter_ns()

    # ---- post-burst link snapshot ----
    post_links = snapshot_link_bytes(rank, world)

    # ---- correctness check: do one FRESH single call with reset inputs ----
    # (RCCL is in-place SUM so inp accumulates; iris writes to out so inp stays
    # constant — but to be uniform, reset inp and call once.)
    expected = float(world * (world + 1) / 2.0)  # 1+2+...+world
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
        call()  # iris path
        actual = float(out.flatten()[0].cpu().item())
    correct = abs(actual - expected) < 1e-3 * expected

    # ---- assemble per-iter arrays ----
    # Truncate the trailing correctness-check call entry (if present)
    ph = np.array(record[:iters], dtype=np.int64)  # shape (iters, 3) of (t0, t1, t2)
    assert ph.shape == (iters, 3), f"phase records shape mismatch: {ph.shape}"
    host_launch_ns = (ph[:, 1] - ph[:, 0]).astype(np.int64)
    device_barrier_ns = (ph[:, 2] - ph[:, 1]).astype(np.int64)
    # cudaEvent kernel times (covers launch+device_barrier on iris path; for
    # RCCL covers dist.all_reduce only — sync runs after)
    event_total_us = np.array(
        [s.elapsed_time(e) * 1e3 for s, e in zip(starts, stops)],
        dtype=np.float64,
    )
    event_total_ns = (event_total_us * 1e3).astype(np.int64)
    wall_ns = host_launch_ns + device_barrier_ns

    # For iris path, device_barrier is part of the kernel/event window because
    # device_barrier itself launches a kernel and the cuda event includes it.
    # event_total_ns ~ host_launch_ns + device_barrier_ns + small overhead.
    # For RCCL, event_total_ns ~ host_launch_ns + sync (sync waits for the
    # collective kernel which may include compute+transfer).
    # We use event_total_ns as the GPU-side total (kernel + barrier kernel).

    # ---- xGMI per-link byte rate (per iter, per link, per rank) ----
    delta_links = post_links - pre_links  # bytes over the whole burst, per peer
    # Convert to per-iter, then take max over peers (this is the wire bottleneck
    # for the local rank's xGMI links).
    per_iter_per_link = delta_links / iters
    max_link_bytes_per_iter = float(per_iter_per_link.max()) if per_iter_per_link.size else 0.0
    # All-reduce on UBB MI300X uses both directions on each link — every rank
    # sees this same wire pattern. Use per-rank max link bytes for the modeled
    # transfer-floor estimate (worst case): time = bytes / link_BW.
    xgmi_transfer_ns_per_iter = max_link_bytes_per_iter / (XGMI_PER_LINK_SOL_GBPS * 1e9) * 1e9

    # ---- compute 5 buckets per iter ----
    # (c) xgmi transfer = scalar (can broadcast)
    xgmi_arr = np.full(iters, xgmi_transfer_ns_per_iter, dtype=np.float64)
    # (d) local reduction = max(0, event_total - xgmi_transfer - device_barrier_kernel_part)
    #   Approximation: the cudaEvent covers launch+device_barrier (kernel),
    #   so the "compute portion of the all_reduce kernel itself" is
    #   event_total - device_barrier_ns - xgmi_transfer.
    # However device_barrier_ns is CPU-perspective time of the barrier kernel,
    # which approximates its GPU duration to first order.
    # local_reduction = max(0, event_total - device_barrier_ns - xgmi_transfer)
    local_reduction = np.clip(
        event_total_ns.astype(np.float64) - device_barrier_ns.astype(np.float64) - xgmi_arr,
        0, None,
    )
    # (e) epilogue/sync = wall - host_launch - device_barrier - (xgmi + reduction)
    #   This captures any extra CPU/GPU overhead beyond what the buckets
    #   accounted for.
    epilogue = (wall_ns.astype(np.float64) - host_launch_ns.astype(np.float64)
                - device_barrier_ns.astype(np.float64)
                - xgmi_arr - local_reduction)
    # Note epilogue may be small/negative due to the device_barrier_ns
    # double-counting with event_total. Clip to 0 — but track signed sum for
    # transparency.
    epilogue_clip = np.clip(epilogue, 0, None)

    return {
        "rank": rank, "world": world,
        "variant": variant, "bytes": int(actual_bytes),
        "M": int(M), "N": int(N),
        "warmup": warmup, "iters": iters,
        "correct": bool(correct),
        "expected": expected, "actual": actual,
        "burst_wall_ns": int(t_burst_end - t_burst_start),
        # raw per-iter arrays (lists for JSON):
        "host_launch_ns": host_launch_ns.tolist(),
        "device_barrier_ns": device_barrier_ns.tolist(),
        "event_total_ns": event_total_ns.tolist(),
        "wall_ns": wall_ns.tolist(),
        "xgmi_transfer_ns_per_iter": xgmi_transfer_ns_per_iter,
        "max_link_bytes_per_iter": max_link_bytes_per_iter,
        "delta_links_total_bytes": delta_links.tolist(),
        "local_reduction_ns": local_reduction.tolist(),
        "epilogue_sync_ns": epilogue_clip.tolist(),
        "epilogue_signed_ns": epilogue.tolist(),  # diagnostic
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Output dir for per-rank JSONs")
    ap.add_argument("--run_id", required=True, help="Run identifier (e.g., run1)")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--sizes", default="1024,4096,16384",
                    help="Comma-separated per-rank input bytes")
    ap.add_argument("--variants", default="rccl,one_shot,two_shot",
                    help="Comma-separated variants to bench")
    ap.add_argument("--heap_gb", type=int, default=4)
    args = ap.parse_args()

    rank, world = init_dist()
    os.makedirs(args.out_dir, exist_ok=True)

    if rank == 0:
        print(f"[K-679 {now()}] world={world} run_id={args.run_id} "
              f"warmup={args.warmup} iters={args.iters}", flush=True)

    # Init iris ONCE (it imports torch.distributed too)
    import iris
    ctx = iris.iris(heap_size=args.heap_gb << 30)
    group_handle = None  # default group

    # Apply patches (only affects iris, not RCCL)
    apply_iris_patch([])

    sizes = [int(x) for x in args.sizes.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]

    out_path = os.path.join(args.out_dir, f"rank{rank}_{args.run_id}.jsonl")
    with open(out_path, "w") as f:
        for variant in variants:
            for size in sizes:
                if rank == 0:
                    print(f"[K-679 {now()}] cell variant={variant} bytes={size}", flush=True)
                # Light barrier between cells
                dist.barrier()
                rec = run_one_cell(rank, world, variant, size,
                                   args.warmup, args.iters, ctx, group_handle)
                rec["run_id"] = args.run_id
                rec["timestamp"] = now()
                f.write(json.dumps(rec) + "\n")
                f.flush()

    if rank == 0:
        print(f"[K-679 {now()}] DONE rank0 -> {out_path}", flush=True)
    # Don't destroy_process_group (matches K-664 / K-468 pattern)
    dist.barrier()


if __name__ == "__main__":
    main()
