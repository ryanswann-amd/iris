# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernel for reduce-scatter collective communication.
Uses the two-shot approach: reduce assigned tiles and store only to own rank.
"""

import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from ..utils import chiplet_transform_chunked


# Number of parallel reduction accumulators used inside the per-tile remote-load
# loop. Splitting the single `acc` chain into N independent chains breaks the
# serial `flat_load -> s_waitcnt vmcnt(0) -> v_add_f32 acc, acc, v` dependency
# the original kernel produced and lets the AMD backend issue all `flat_load`s
# before any wait, then drain them with a single trailing `s_waitcnt`. This is
# the K-630 RS sync-bucket fix (R-RS-WAITCNT-FUSION) — same shape as K-370 /
# K-371 propose for all_gather / all_to_all RC-2'. 4 chains is chosen to match
# the typical world_size=8 single-node MI300X configuration (depth-2 per chain).
_NUM_PARALLEL_ACC: tl.constexpr = 4


@triton.jit()
def persistent_reduce_scatter_two_shot(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
):
    """
    Reduce-scatter using two-shot approach.

    Each rank reduces its assigned tiles from all ranks and stores the result
    only to its own output (no broadcast to other ranks).
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

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

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N

        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

        # Build indices (used by both paths)
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)

        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        base_ptr = input_ptr + input_offset
        out_ptr = output_ptr + output_offset

        # Per-PID rotation (already applied — K-630 verified per-link is balanced).
        start_rank_idx = pid % world_size

        # Fast path: NO MASKS (full tiles)
        # The masking is problem size dependent, and the compiler does not recognize it can have two paths
        # (one with masks and one without). Separate unmasked paths allow the compiler to generate
        # more efficient vectorized instructions.
        if is_full:
            # K-642 / R-RS-WAITCNT-FUSION: split the single accumulator into 4
            # independent chains so the AMD backend can issue all world_size
            # `flat_load` instructions before any `s_waitcnt`, instead of
            # serialising them as `flat_load -> s_waitcnt -> v_add` per peer.
            # `cache_modifier=".cg"` on remote loads bypasses the L1 (project
            # rule for iris CCL kernels — remote data is single-use, L1 caching
            # only adds eviction pressure).
            acc0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), acc_dtype)
            acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), acc_dtype)
            acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), acc_dtype)
            acc3 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), acc_dtype)

            for i in tl.static_range(0, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                v = iris.load(
                    base_ptr,
                    iris_rank,
                    remote_rank,
                    heap_bases,
                    hint=(1, BLOCK_SIZE_N),
                    cache_modifier=".cg",
                ).to(acc_dtype)
                # Distribute across 4 parallel chains (constexpr branch — i is
                # constexpr from tl.static_range, so this is selected at compile
                # time without runtime divergence).
                if i % 4 == 0:
                    acc0 += v
                elif i % 4 == 1:
                    acc1 += v
                elif i % 4 == 2:
                    acc2 += v
                else:
                    acc3 += v

            # Tree fan-in (depth = log2(4) = 2) instead of linear chain.
            acc = (acc0 + acc1) + (acc2 + acc3)

            reduced = acc.to(output_ptr.type.element_ty)

            # Store only to own rank (no broadcast)
            tl.store(out_ptr, reduced, cache_modifier=".wt")

        # Slow path: MASKED (only boundary tiles land here)
        # This path handles tiles at tensor boundaries where not all elements are valid.
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            acc0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), acc_dtype)
            acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), acc_dtype)
            acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), acc_dtype)
            acc3 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), acc_dtype)

            for i in tl.static_range(0, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                v = iris.load(
                    base_ptr,
                    iris_rank,
                    remote_rank,
                    heap_bases,
                    mask=mask,
                    hint=(1, BLOCK_SIZE_N),
                    cache_modifier=".cg",
                ).to(acc_dtype)
                if i % 4 == 0:
                    acc0 += v
                elif i % 4 == 1:
                    acc1 += v
                elif i % 4 == 2:
                    acc2 += v
                else:
                    acc3 += v

            acc = (acc0 + acc1) + (acc2 + acc3)

            reduced = acc.to(output_ptr.type.element_ty)

            # Store only to own rank (no broadcast)
            tl.store(out_ptr, reduced, mask=mask, cache_modifier=".wt")


def launch(
    output_tensor,
    input_tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    config,
):
    """Launch the Triton reduce-scatter kernel."""
    M, N = input_tensor.shape[:2]
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    heap_bases = ctx.get_heap_bases()
    distribution = config.all_reduce_distribution

    iris_launch(
        persistent_reduce_scatter_two_shot,
        (config.comm_sms,),
        input_tensor,
        output_tensor,
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
        distribution,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="reduce_scatter",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
