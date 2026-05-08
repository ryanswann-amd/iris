# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernel for all-to-all collective communication.
"""

import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from ..utils import chiplet_transform_chunked


# =============================================================================
# K-641 / S-007 closure note (2026-05-08, c42 8x MI300X, ROCm 7.2)
# =============================================================================
# Premise (from K-629 hand-off): the iris.ccl.all_to_all kernel below was
# suspected to suffer from per-(src,dst) xGMI traffic-matrix asymmetry
# analogous to the pre-K-387 all_reduce hot-link pattern, and a peer-rotation
# / chunking schedule (along the lines of K-387 for all_reduce, K-599 for
# all_gather) was hypothesised to lift per-link xGMI efficiency at 256 MB+.
#
# K-629 (per-peer amdsmi traffic-matrix sweep, 8x MI300X, 64/256/1024 MB,
# 3 torchruns x 15 s burst) directly measured the per-(src,dst) link load on
# all 56 directed xGMI4 links for both iris.ccl.all_to_all (this kernel,
# post-K-402 device_barrier) and RCCL ncclAllToAll. Headline numbers:
#
#   max-link / min-link of mean wr_gbps across the 56 links:
#     iris : 1.007 - 1.027   (CV 0.18 - 0.72 %)
#     rccl : 1.034 - 1.040   (CV 1.00 - 1.18 %)
#
#   per-rank egress (mean of 3 runs):
#     1024 MB : iris 60.0 GB/s/link (93.8 % SOL) vs rccl 57.6 GB/s (90.0 %)
#      256 MB : iris 59.5 GB/s/link (92.9 % SOL) vs rccl 54.9 GB/s (85.7 %)
#       64 MB : iris 57.8 GB/s/link (90.3 % SOL) vs rccl 43.1 GB/s (67.4 %)
#
#   walltime ratio (iris / rccl, lower-is-better-iris):
#     0.782 / 0.912 / 0.933   at 64 / 256 / 1024 MB.
#
# Findings:
#   1. iris all_to_all is ALREADY more balanced than RCCL (max/min 1.027 vs
#      1.040 worst-case across the 56 links). There is no K-387-style hot-
#      link signature (which was max/min >= 5x).
#   2. iris ALREADY BEATS RCCL on wall time at every measured size. The
#      "gap to close" stated in S-007 is negative: iris is ~7-22 % faster
#      and pushes ~1.04x more egress per link than RCCL at 1 GB.
#   3. The persistent tile loop below ( `for tile_id in range(pid,
#      total_tiles, COMM_SMS)` , followed by an inner `for i in
#      range(world_size)` over peers ) self-balances over the all-to-all
#      destination set the same way K-490/K-616 demonstrated for the
#      analogous all_gather kernel: at any instant, COMM_SMS workgroups
#      are simultaneously stepping through 7 different remote ranks each,
#      so the across-PID first-touched-link pattern is uniform.
#
# Conclusion (mirrors K-387 closure on all_reduce):
#   - No K-387-style start_rank_idx = pid % world_size rotation is added to
#     this kernel: the per-link distribution leaves no headroom (max-link
#     already 93-94 % of xGMI4 SOL at 256 MB+), and any rotation that re-
#     orders the inner peer loop without changing the tile partition would
#     change at most the first-iteration link choice, which K-629 measured
#     to be already uniform.
#   - The K-387 PR explicitly rejected an analogous per-tile rotation on
#     all_reduce after measuring -2.5 pp at 256 MB and -1.8 pp at 1 GB on
#     the same hardware; that result is a strong prior against blindly
#     introducing the same change here.
#   - Future regressions on all_to_all should look at host-barrier (K-402-
#     class), launch overhead (K-259-class), int64 byte-offset overflow at
#     non-default shapes (K-195 R-A2A-INT64), or multi-channel scaling
#     (K-490 R-MULTI-CHANNEL) - NOT scheduling/rotation.
#
# This block is comment-only; the kernel below is unchanged.
# =============================================================================


@triton.jit()
def persistent_all_to_all(
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
):
    """
    Persistent all-to-all kernel.

    Each rank sends input data to all ranks and receives data from all ranks.
    Similar to all-scatter but bidirectional.

    Args:
        input_ptr: Pointer to input tensor (local rank's data to send)
        output_ptr: Pointer to output tensor (will receive from all ranks)
        M: Number of rows
        N: Number of columns per rank (output will be N * world_size)
        stride_in_m, stride_in_n: Strides for input tensor
        stride_out_m, stride_out_n: Strides for output tensor
        heap_bases: Heap base pointers for all ranks
        group_rank: Rank within the ProcessGroup (0 to group_size-1), used for tile assignment and comparisons
        iris_rank: Rank in the iris context, used for iris RMA operations (heap_bases indexing)
        world_size: Total number of ranks in the group
        BLOCK_SIZE_M, BLOCK_SIZE_N: Block sizes for tiling
        GROUP_SIZE_M: Group size for M dimension tiling
        COMM_SMS: Number of SMs for communication
        NUM_XCDS: Number of XCDs
        CHUNK_SIZE: Chunk size for chiplet transform
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, COMM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # Compute base indices for this tile
        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N

        # Check if this tile is fully within bounds (no edge cases)
        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

        # Build indices (used by both paths)
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Pre-compute base offsets for better memory access patterns and vectorization
        input_base_m = rm[:, None] * stride_in_m
        output_base_m = rm[:, None] * stride_out_m
        input_base_n = rn[None, :] * stride_in_n
        output_base_n = rn[None, :] * stride_out_n

        # Fast path: NO MASKS (full tiles)
        # The masking is problem size dependent, and the compiler does not recognize it can have two paths
        # (one with masks and one without). Separate unmasked paths allow the compiler to generate
        # more efficient vectorized instructions.
        if is_full:
            # Process local rank first for better cache locality
            input_offset_local = input_base_m + (input_base_n + group_rank * N * stride_in_n)
            output_offset_local = output_base_m + (output_base_n + group_rank * N * stride_out_n)
            input_ptr_local = input_ptr + input_offset_local
            output_ptr_local = output_ptr + output_offset_local
            input_ptr_local = tl.multiple_of(input_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))
            output_ptr_local = tl.multiple_of(output_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))

            data = tl.load(input_ptr_local)
            tl.store(output_ptr_local, data, cache_modifier=".wt")

            # Process all remote ranks
            for i in range(world_size):
                target_rank = rank_start + i * rank_stride
                if i != group_rank:
                    # Calculate which chunk of input to read based on rank_in_group
                    rank_in_group_target = i
                    input_offset_remote = input_base_m + (input_base_n + rank_in_group_target * N * stride_in_n)
                    output_offset_remote = output_base_m + (output_base_n + group_rank * N * stride_out_n)
                    input_ptr_remote = input_ptr + input_offset_remote
                    output_ptr_remote = output_ptr + output_offset_remote
                    input_ptr_remote = tl.multiple_of(input_ptr_remote, (BLOCK_SIZE_M, BLOCK_SIZE_N))
                    output_ptr_remote = tl.multiple_of(output_ptr_remote, (BLOCK_SIZE_M, BLOCK_SIZE_N))

                    remote_data = tl.load(input_ptr_remote)
                    iris.store(
                        output_ptr_remote,
                        remote_data,
                        iris_rank,
                        target_rank,
                        heap_bases,
                        hint=(1, BLOCK_SIZE_N),
                    )

        # Slow path: MASKED (only boundary tiles land here)
        # This path handles tiles at tensor boundaries where not all elements are valid.
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            # Process local rank first for better cache locality
            input_offset_local = input_base_m + (input_base_n + group_rank * N * stride_in_n)
            output_offset_local = output_base_m + (output_base_n + group_rank * N * stride_out_n)
            input_ptr_local = input_ptr + input_offset_local
            output_ptr_local = output_ptr + output_offset_local
            input_ptr_local = tl.multiple_of(input_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))
            output_ptr_local = tl.multiple_of(output_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))

            data = tl.load(input_ptr_local, mask=mask)
            tl.store(output_ptr_local, data, mask=mask, cache_modifier=".wt")

            # Process all remote ranks
            for i in range(world_size):
                target_rank = rank_start + i * rank_stride
                if i != group_rank:
                    # Calculate which chunk of input to read based on rank_in_group
                    rank_in_group_target = i
                    input_offset_remote = input_base_m + (input_base_n + rank_in_group_target * N * stride_in_n)
                    output_offset_remote = output_base_m + (output_base_n + group_rank * N * stride_out_n)
                    input_ptr_remote = input_ptr + input_offset_remote
                    output_ptr_remote = output_ptr + output_offset_remote
                    input_ptr_remote = tl.multiple_of(input_ptr_remote, (BLOCK_SIZE_M, BLOCK_SIZE_N))
                    output_ptr_remote = tl.multiple_of(output_ptr_remote, (BLOCK_SIZE_M, BLOCK_SIZE_N))

                    remote_data = tl.load(input_ptr_remote, mask=mask)
                    iris.store(
                        output_ptr_remote,
                        remote_data,
                        iris_rank,
                        target_rank,
                        heap_bases,
                        mask=mask,
                        hint=(1, BLOCK_SIZE_N),
                    )


def launch(
    input_tensor,
    output_tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    config,
):
    """Launch the Triton all-to-all kernel."""
    M, total_N = input_tensor.shape[:2]
    N = total_N // world_size

    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    iris_launch(
        persistent_all_to_all,
        (config.comm_sms,),
        input_tensor,
        output_tensor,
        M,
        N,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        ctx.get_heap_bases(),
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
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="all_to_all",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
