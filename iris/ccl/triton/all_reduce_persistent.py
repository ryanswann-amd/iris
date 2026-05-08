# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Persistent / resident-kernel fast-path for the iris ``all_reduce`` collective.

Motivation
----------
``K-796`` decomposed the iris small-message launch envelope on MI300X and
showed that ~50 µs of every ``all_reduce`` call is host-side launch overhead
(~18 µs Triton wrapper + ~20 µs MES dispatch + ~3 µs HIP submit + plumbing).
RCCL's persistent-kernel model amortises most of that cost across iterations.

This module prototypes the equivalent for iris with
``persistent_all_reduce_two_shot_burst`` — a single launch that performs
``NUM_ITERS`` back-to-back reductions on the *same* input/output buffers and
uses a symmetric counter-based barrier between iterations.  Per-iter cost
becomes (kernel body + cross-rank flag barrier) with the host-side launch
envelope amortised across all iterations.

Only the ``two_shot`` variant is exposed because K-685/K-782 showed it has
the largest absolute launch_us gap vs RCCL.

All remote ``iris.load`` calls use ``cache_modifier='.cg'`` per the project
rules in ``CLAUDE.md`` to bypass the CU L1 cache (avoiding stale data on
re-read in the persistent loop).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import triton
import triton.language as tl
import torch
import iris

from iris.host.tracing.kernel_artifacts import iris_launch


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


@dataclass
class PersistentAllReduceWorkspace:
    """Reusable workspace for the persistent two-shot kernel.

    Attributes:
        shape:              (M, N) of the tensor the persistent kernel was
                            specialised for.
        dtype:              torch dtype used for the kernel body.
        max_iters:          number of iteration slots provisioned in the
                            ``iter_barrier`` array.
        comm_sms:           number of CTAs the persistent kernel was launched
                            with (used to size the per-iter barrier counter).
        iter_barrier:       symmetric int32 array of shape (max_iters,) — CTAs
                            atomically count themselves at the end of each iter
                            so the next iter can wait until all peer ranks
                            finished writing the previous result.
        prepared:           true once the workspace flags are zeroed and ready.
    """

    shape: Tuple[int, int] = ()
    dtype: Optional[torch.dtype] = None
    max_iters: int = 0
    comm_sms: int = 0
    iter_barrier: Optional[torch.Tensor] = None
    prepared: bool = False


def persistent_all_reduce_preamble(
    output_tensor,
    input_tensor,
    ctx,
    config,
    max_iters: int,
    workspace: Optional[PersistentAllReduceWorkspace] = None,
):
    """Allocate / re-zero the per-iter barrier flag array.

    The flag array is placed in the iris symmetric heap so any rank can
    atomically read or write any peer's slot via ``iris.atomic_add`` /
    ``iris.atomic_cas``.
    """
    if max_iters <= 0:
        raise ValueError(f"max_iters must be positive, got {max_iters}")

    M, N = input_tensor.shape[:2]
    dtype = input_tensor.dtype

    if workspace is None:
        workspace = PersistentAllReduceWorkspace()

    needs_alloc = (
        workspace.iter_barrier is None
        or workspace.max_iters < max_iters
        or workspace.comm_sms != config.comm_sms
    )

    if needs_alloc:
        workspace.iter_barrier = ctx.zeros((max_iters,), dtype=torch.int32)
    else:
        workspace.iter_barrier.zero_()

    workspace.shape = (M, N)
    workspace.dtype = dtype
    workspace.max_iters = max_iters
    workspace.comm_sms = config.comm_sms

    # Cross-rank barrier so all ranks see freshly-zeroed flags before the
    # persistent kernel is launched.
    ctx.barrier()
    workspace.prepared = True
    return workspace


# ---------------------------------------------------------------------------
# Triton helpers
# ---------------------------------------------------------------------------


@triton.jit
def _two_shot_iter_body(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases,
    pid,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
):
    """Inlined two-shot iteration body for the persistent burst kernel.

    Mirrors ``persistent_all_reduce_two_shot`` in ``all_reduce.py`` but
    annotates remote loads with ``cache_modifier='.cg'`` so successive
    persistent iterations always observe peer writes (the CU/L1 line is
    bypassed; data is fetched through L2 / LLC).
    """
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    tiles_per_rank = tl.cdiv(total_tiles, world_size)
    if DISTRIBUTION == 0:
        start_tile = group_rank
        stride = world_size
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.cdiv(remaining, stride)
    else:
        start_tile = group_rank * tiles_per_rank
        stride = 1
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.minimum(tiles_per_rank, remaining)

    for tile_offset in range(pid, max_tile_offset, COMM_SMS):
        tile_id = start_tile + tile_offset * stride

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N

        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)

        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        base_ptr = input_ptr + input_offset
        out_ptr = output_ptr + output_offset

        if is_full:
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(
                base_ptr,
                iris_rank,
                start_rank_global,
                heap_bases,
                cache_modifier=".cg",
            ).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(
                    base_ptr,
                    iris_rank,
                    remote_rank,
                    heap_bases,
                    cache_modifier=".cg",
                ).to(acc_dtype)

            reduced = acc.to(output_ptr.type.element_ty)

            tl.store(out_ptr, reduced, cache_modifier=".wt")

            for i in tl.static_range(0, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                if remote_rank_idx != group_rank:
                    iris.store(
                        out_ptr,
                        reduced,
                        iris_rank,
                        remote_rank,
                        heap_bases,
                        hint=(1, BLOCK_SIZE_N),
                    )

        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(
                base_ptr,
                iris_rank,
                start_rank_global,
                heap_bases,
                mask=mask,
                cache_modifier=".cg",
            ).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(
                    base_ptr,
                    iris_rank,
                    remote_rank,
                    heap_bases,
                    mask=mask,
                    cache_modifier=".cg",
                ).to(acc_dtype)

            reduced = acc.to(output_ptr.type.element_ty)

            tl.store(out_ptr, reduced, mask=mask, cache_modifier=".wt")

            for i in tl.static_range(0, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                if remote_rank_idx != group_rank:
                    iris.store(
                        out_ptr,
                        reduced,
                        iris_rank,
                        remote_rank,
                        heap_bases,
                        mask=mask,
                        hint=(1, BLOCK_SIZE_N),
                    )


@triton.jit
def _cross_rank_iter_barrier(
    iter_barrier_ptr,
    iter_id,
    heap_bases,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    EXPECTED: tl.constexpr,
):
    """Counter-based cross-rank barrier.

    Each CTA increments ``iter_barrier_ptr[iter_id]`` on every peer rank
    (release semantics) and then spins until its local counter reaches
    ``EXPECTED = COMM_SMS * world_size`` (acquire semantics).

    By the time the wait returns, every CTA on every rank has finished iter
    ``iter_id`` — so reads of peer outputs / inputs in the next iter are safe.
    """
    for i in range(world_size):
        target = rank_start + i * rank_stride
        iris.atomic_add(
            iter_barrier_ptr + iter_id,
            1,
            iris_rank,
            target,
            heap_bases,
            sem="release",
            scope="sys",
        )
    # Spin on local counter (acquire — make peer writes visible).  Use
    # cmp==val==EXPECTED pattern: when local count < EXPECTED, the CAS is a
    # no-op load returning the current value.  Once it hits EXPECTED the swap
    # is a no-op write of the same value.
    seen = tl.atomic_cas(iter_barrier_ptr + iter_id, EXPECTED, EXPECTED, sem="acquire", scope="sys")
    while seen != EXPECTED:
        seen = tl.atomic_cas(iter_barrier_ptr + iter_id, EXPECTED, EXPECTED, sem="acquire", scope="sys")


# ---------------------------------------------------------------------------
# Burst kernel: single launch runs NUM_ITERS reductions back-to-back.
# ---------------------------------------------------------------------------


@triton.jit
def persistent_all_reduce_two_shot_burst(
    input_ptr,
    output_ptr,
    iter_barrier_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,  # signature parity with non-persistent kernel
    CHUNK_SIZE: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
    NUM_ITERS: tl.constexpr,
    USE_BARRIER: tl.constexpr,
):
    """Single launch that performs ``NUM_ITERS`` two-shot all-reduces.

    Between iterations a counter-based cross-rank barrier guarantees every
    rank has finished writing its assigned output tiles before the next
    iteration's remote reads/writes begin.  Set ``USE_BARRIER=False`` only
    in microbenchmarks where input data is constant across iterations and
    the per-iter inter-rank ordering is irrelevant.
    """
    pid = tl.program_id(0)
    expected = COMM_SMS * world_size

    for iter_id in range(NUM_ITERS):
        _two_shot_iter_body(
            input_ptr,
            output_ptr,
            M,
            N,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
            heap_bases,
            pid,
            group_rank,
            iris_rank,
            world_size,
            rank_start,
            rank_stride,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GROUP_SIZE_M,
            COMM_SMS,
            DISTRIBUTION,
        )

        if USE_BARRIER:
            _cross_rank_iter_barrier(
                iter_barrier_ptr,
                iter_id,
                heap_bases,
                iris_rank,
                world_size,
                rank_start,
                rank_stride,
                expected,
            )


# ---------------------------------------------------------------------------
# Host-side launch wrappers.
# ---------------------------------------------------------------------------


def launch_persistent_burst(
    output_tensor,
    input_tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    config,
    workspace: PersistentAllReduceWorkspace,
    num_iters: int,
    use_barrier: bool = True,
):
    """Single launch that runs ``num_iters`` all-reduces in one kernel."""
    if num_iters <= 0:
        raise ValueError(f"num_iters must be positive, got {num_iters}")
    if num_iters > workspace.max_iters:
        raise ValueError(f"num_iters={num_iters} exceeds workspace.max_iters={workspace.max_iters}")
    M, N = input_tensor.shape[:2]
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)
    heap_bases = ctx.get_heap_bases()

    iris_launch(
        persistent_all_reduce_two_shot_burst,
        (config.comm_sms,),
        input_tensor,
        output_tensor,
        workspace.iter_barrier,
        M,
        N,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        heap_bases,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        config.block_size_m,
        config.block_size_n,
        config.swizzle_size,
        config.comm_sms,
        config.num_xcds,
        config.chunk_size,
        config.all_reduce_distribution,
        num_iters,
        use_barrier,
        num_warps=8,
        num_stages=1,
        waves_per_eu=1,
        algorithm="all_reduce",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
    return workspace


