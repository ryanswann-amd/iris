# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""K-599 regression test for `iris.ccl.all_gather` (`persistent` variant).

Two distinct regressions guarded:

1. **>=1 GB no-crash + bit-exact vs RCCL.** Pre-fix
   ``persistent_all_gather`` SIGABRTs on rank 4 with
   ``hipErrorIllegalAddress`` at 1 GB bf16 per-rank input on world=8 because
   ``output_offset = rm_output * stride_out_m`` was evaluated at int32 and
   wrapped negative at byte offset ~ 2^32. The K-599 fix casts ``rm_output``
   and ``rn`` to ``tl.int64`` before the multiplication. The kernel must
   complete and produce a bit-identical output to ``dist.all_gather_into_tensor``
   (bf16 with no reduction => byte-equal).

2. **Wall-time within 1.30x of RCCL at 1 GB.** A bound that the kernel's
   per-PID destination-loop rotation has not regressed to the original
   serial-by-rank ordering (which historically queued all 64 COMM_SMS PIDs
   on the same outgoing xGMI peer link). Empirical post-fix is ~1.03x of
   RCCL; 1.30x is generous against iter-to-iter jitter on a shared cluster
   but tight enough to fail a serial-by-rank regression.

Requires ``world_size == 8`` (a typical 8x MI300X node). Other world sizes
are skipped (the int32 overflow only triggers at >=1 GB output buffers,
which only exists at world=8 with bf16 1 GB per-rank input).
"""

import os

import pytest
import torch
import torch.distributed as dist

import iris
from iris.ccl import Config


SIZE_MB_REGRESSION = 1024  # 1 GB per-rank input — the size that pre-fix SIGABRTs
N_COL = 2048
WARMUP = 50
ITERS = 100
WALL_RATIO_MAX = 1.30


def _require_world_8():
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    if dist.get_world_size() != 8:
        pytest.skip(
            f"K-599 regression requires world_size=8 to reproduce >=1 GB"
            f" output (got world_size={dist.get_world_size()})")


def _make_buffers(shmem, rank, world, size_mb):
    size_bytes = size_mb * (1 << 20)
    elem_bytes = 2  # bf16
    total_elems = size_bytes // elem_bytes
    M = total_elems // N_COL
    assert M * N_COL == total_elems, (
        f"size {size_mb}MB not divisible by N={N_COL}")

    in_t = shmem.zeros((M, N_COL), dtype=torch.bfloat16,
                       device=f"cuda:{rank}")
    out_t = shmem.zeros((world * M, N_COL), dtype=torch.bfloat16,
                        device=f"cuda:{rank}")
    in_t.fill_(rank + 1)
    rccl_in = torch.full((M, N_COL), rank + 1, dtype=torch.bfloat16,
                         device=f"cuda:{rank}")
    rccl_out = torch.empty((world * M, N_COL), dtype=torch.bfloat16,
                           device=f"cuda:{rank}")
    return in_t, out_t, rccl_in, rccl_out


def test_persistent_all_gather_1gb_no_crash_and_bit_exact():
    """K-599 #1: 1 GB per-rank bf16 must not SIGABRT and must match RCCL."""
    _require_world_8()
    rank = dist.get_rank()
    world = dist.get_world_size()

    shmem = iris.iris(2 ** 34)  # 16 GB heap (need room for 8x1GB)
    try:
        cfg = Config(block_size_m=32, block_size_n=64,
                     all_gather_variant="persistent", comm_sms=64)
        in_t, out_t, rccl_in, rccl_out = _make_buffers(
            shmem, rank, world, SIZE_MB_REGRESSION)

        # Pre-fix this SIGABRTs on rank 4 with hipErrorIllegalAddress.
        out_t.fill_(-1.0)
        shmem.ccl.all_gather(out_t, in_t, async_op=True, config=cfg)
        shmem.barrier()
        torch.cuda.synchronize()
        dist.barrier()

        rccl_out.fill_(-1.0)
        dist.all_gather_into_tensor(rccl_out, rccl_in)
        torch.cuda.synchronize()

        assert torch.equal(out_t, rccl_out), (
            f"K-599 regression on rank {rank}: iris.ccl.all_gather output"
            f" differs from dist.all_gather_into_tensor at"
            f" {SIZE_MB_REGRESSION} MB / world={world}. The int64 offset"
            f" cast in persistent_all_gather may have been reverted,"
            f" or a new int32-overflow was introduced.")
    finally:
        shmem.barrier()
        del shmem
        import gc
        gc.collect()


def test_persistent_all_gather_1gb_wall_time_within_1p3x_of_rccl():
    """K-599 #2: wall-time at 1 GB stays within 1.30x of RCCL.

    Pre-fix kernel cannot run at 1 GB (SIGABRT). Post-fix is ~1.03x of
    RCCL. A regression that reverts the per-PID destination-loop rotation
    to serial-by-rank ordering would push the ratio well above 1.30x by
    queueing all PIDs on a single outgoing xGMI peer link."""
    _require_world_8()
    rank = dist.get_rank()
    world = dist.get_world_size()

    shmem = iris.iris(2 ** 34)
    try:
        cfg = Config(block_size_m=32, block_size_n=64,
                     all_gather_variant="persistent", comm_sms=64)
        in_t, out_t, rccl_in, rccl_out = _make_buffers(
            shmem, rank, world, SIZE_MB_REGRESSION)

        def iris_fn():
            shmem.ccl.all_gather(out_t, in_t, async_op=True, config=cfg)

        def rccl_fn():
            dist.all_gather_into_tensor(rccl_out, rccl_in)

        def time_one(fn):
            for _ in range(WARMUP):
                fn()
            torch.cuda.synchronize()
            dist.barrier()
            ts = []
            for _ in range(ITERS):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                dist.barrier()
                s.record()
                fn()
                e.record()
                torch.cuda.synchronize()
                ts.append(s.elapsed_time(e) * 1e3)
            ts.sort()
            return ts[len(ts) // 2]

        iris_med = time_one(iris_fn)
        rccl_med = time_one(rccl_fn)

        t = torch.tensor([iris_med, rccl_med], dtype=torch.float64,
                         device=f"cuda:{rank}")
        gathered = [torch.zeros_like(t) for _ in range(world)]
        dist.all_gather(gathered, t)
        if rank == 0:
            iris_max = max(g[0].item() for g in gathered)
            rccl_max = max(g[1].item() for g in gathered)
            ratio = iris_max / rccl_max
            print(f"K-599 wall ratio @ {SIZE_MB_REGRESSION}MB:"
                  f" iris={iris_max:.1f}us rccl={rccl_max:.1f}us"
                  f" ratio={ratio:.3f}x", flush=True)
            assert ratio <= WALL_RATIO_MAX, (
                f"K-599 regression: iris/rccl wall-time ratio at"
                f" {SIZE_MB_REGRESSION} MB = {ratio:.3f}x exceeds"
                f" {WALL_RATIO_MAX}x. The per-PID destination-loop"
                f" rotation may have reverted to serial-by-rank xGMI"
                f" ordering.")
    finally:
        shmem.barrier()
        del shmem
        import gc
        gc.collect()
