#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Iris MoE V2 - Ported from Triton's exact reference
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris

# Using local copy of Triton kernels for standalone example
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from triton_kernels.distributed import make_expt_dict_uniform, make_expt_assignment, symm_mem_pool
from triton_kernels.reduce import reduce
from triton_kernels.topk import topk
from triton_kernels.matmul import matmul
from triton_kernels.tensor import make_ragged_tensor_metadata, remap_ragged_tensor_metadata


def _convert_launch_metadata(grid, kernel, args):
    """
    Launch metadata for profiling - same signature as Triton reference
    For now, return empty dict (can add profiling metrics later)
    """
    return {}


@triton.jit(launch_metadata=_convert_launch_metadata)
def _convert_dp_to_ep_iris(
    dst_ptr,
    dst_stride_m,
    src_ptr,
    src_stride_m,
    src_shape_n,
    expt_filter_ptr,
    expt_filter_stride_m,
    expt_indx_ptr,
    expt_indx_stride_m,
    dst_row_indx_ptr,
    dst_row_indx_stride_m,
    n_tokens_local,
    heap_bases,
    SRC_RANK: tl.constexpr,
    N_EXPT_ACT: tl.constexpr,
    N_RANKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_m_global = pid_m + n_tokens_local * SRC_RANK
    off_m_local = pid_m

    # Routing logic
    offs_r = tl.arange(0, N_RANKS)
    offs_e = tl.arange(0, N_EXPT_ACT)
    offs_n = tl.arange(0, BLOCK)
    dst_row_indx = tl.load(dst_row_indx_ptr + off_m_global * dst_row_indx_stride_m + offs_e)
    expt_indx = tl.load(expt_indx_ptr + off_m_global * expt_indx_stride_m + offs_e)
    expt_filter_ptr_rows = expt_filter_ptr + offs_r[:, None] * expt_filter_stride_m
    expt_filter = (tl.load(expt_filter_ptr_rows + (expt_indx // 32)[None, :]) >> (expt_indx % 32)) & 1
    expt_ranks = tl.sum(offs_r[:, None] * expt_filter, axis=0)

    # This is where Triton example generate better code
    dst_row_offsets = dst_row_indx * dst_stride_m
    dst_offsets = dst_row_offsets[:, None] + offs_n[None, :]
    src_ptrs = src_ptr + off_m_local * src_stride_m + offs_n

    for start_n in range(0, src_shape_n, BLOCK):
        mask_n = start_n + offs_n < src_shape_n
        src = tl.load(src_ptrs, mask=mask_n, other=0.0)

        # Write to each rank
        for r in tl.static_range(N_RANKS):
            rank_mask = expt_ranks == r
            dst_ptrs = dst_ptr + dst_offsets + start_n
            full_mask = rank_mask[:, None] & mask_n[None, :]
            iris.store(dst_ptrs, src[None, :], SRC_RANK, r, heap_bases, mask=full_mask)

        src_ptrs += BLOCK


def convert_dp_to_ep_iris(src, expt_assignment, expt_indx, gate_indx, shmem, dst_buffer):
    """
    Iris version of convert_dp_to_ep
    Uses Iris symmetric memory - dst_buffer must be pre-allocated with shmem.zeros()
    """
    expt_bitmask = expt_assignment.expt_bitmask
    rank = dist.get_rank()
    n_ranks = dist.get_world_size()
    device = src.device
    n_tokens_local, d_model = src.shape
    n_tokens_global, n_expt_act = expt_indx.shape

    # Validate
    assert n_ranks == expt_bitmask.size(0)
    assert all(t.device == device for t in [expt_bitmask, expt_indx, gate_indx])
    assert expt_bitmask.dtype == torch.int32
    assert n_tokens_local * n_ranks <= n_tokens_global

    # Get Iris heap bases
    heap_bases = shmem.get_heap_bases()

    # Launch kernel
    BLOCK = 512
    grid = (n_tokens_local,)
    _convert_dp_to_ep_iris[grid](
        dst_buffer,
        dst_buffer.stride(0),
        src,
        src.stride(0),
        src.shape[1],
        expt_bitmask,
        expt_bitmask.stride(0),
        expt_indx,
        expt_indx.stride(0),
        gate_indx,
        gate_indx.stride(0),
        n_tokens_local,
        heap_bases,
        SRC_RANK=rank,
        N_EXPT_ACT=n_expt_act,
        N_RANKS=n_ranks,
        BLOCK=BLOCK,
    )

    # Iris barrier
    shmem.barrier()
    return dst_buffer


@triton.jit(launch_metadata=_convert_launch_metadata)
def _convert_ep_to_dp_iris(
    dst_ptr,
    dst_stride_m,
    src_ptr,
    src_stride_m,
    src_shape_n,
    expt_filter_ptr,
    expt_filter_stride_m,
    expt_indx_ptr,
    dst_row_indx_ptr,
    n_tokens_local,
    heap_bases,  # Iris heap bases pointer
    BLOCK: tl.constexpr,
    SRC_RANK: tl.constexpr,
    N_RANKS: tl.constexpr,
):
    pid_m = tl.program_id(0)

    # Determine destination rank and index
    dst_indx_global = tl.load(dst_row_indx_ptr + pid_m)
    dst_rank = dst_indx_global // n_tokens_local

    # Check if this rank owns the destination expert
    dst_expt_indx = tl.load(expt_indx_ptr + dst_indx_global)
    expt_filter_ptr_local = expt_filter_ptr + SRC_RANK * expt_filter_stride_m
    has_dst_expt = (tl.load(expt_filter_ptr_local + dst_expt_indx // 32) >> (dst_expt_indx % 32)) & 1

    if not has_dst_expt.to(tl.int1):
        return

    dst_indx_local = dst_indx_global - dst_rank * n_tokens_local

    # Load and write to destination rank
    offs_n = tl.arange(0, BLOCK)
    src_ptrs = src_ptr + pid_m * src_stride_m + offs_n

    for start_n in range(0, src_shape_n, BLOCK):
        mask_n = start_n + offs_n < src_shape_n
        src = tl.load(src_ptrs, mask=mask_n, other=0.0)

        dst_offset = dst_indx_local * dst_stride_m + start_n + offs_n
        dst_ptrs = dst_ptr + dst_offset

        for r in tl.static_range(N_RANKS):
            if dst_rank == r:
                iris.store(dst_ptrs, src, SRC_RANK, r, heap_bases, mask=mask_n)

        src_ptrs += BLOCK


def convert_ep_to_dp_iris(src, expt_assignment, expt_indx, topk_indx, shmem, dst_buffer):
    """
    Iris version of convert_ep_to_dp
    Uses Iris symmetric memory - dst_buffer must be pre-allocated with shmem.zeros()
    """
    expt_bitmask = expt_assignment.expt_bitmask
    rank = dist.get_rank()
    n_ranks = dist.get_world_size()
    n_tokens_global, d_model = src.shape
    n_tokens_local = n_tokens_global // n_ranks

    # Get Iris heap bases
    heap_bases = shmem.get_heap_bases()

    # Launch kernel
    BLOCK = 512
    grid = (n_tokens_global,)
    _convert_ep_to_dp_iris[grid](
        dst_buffer,
        dst_buffer.stride(0),
        src,
        src.stride(0),
        src.shape[1],
        expt_bitmask,
        expt_bitmask.stride(0),
        expt_indx,
        topk_indx,
        n_tokens_local,
        heap_bases,
        BLOCK=BLOCK,
        SRC_RANK=rank,
        N_RANKS=n_ranks,
    )

    # Iris barrier
    shmem.barrier()
    return dst_buffer


def mixture_of_expt_iris(x_dp_local, l_dp_local, w_ep_local, b_ep_local, expt_assignment, n_expts_act, shmem):
    """
    Complete Iris MoE implementation
    Uses Iris for communication, Triton's matmul for compute
    All buffers allocated with Iris symmetric memory
    """
    rank = dist.get_rank()
    n_ranks = dist.get_world_size()
    expt_map = expt_assignment.expt_map[rank, :]
    n_tokens_local, d_model = x_dp_local.shape
    n_tokens_global = n_tokens_local * n_ranks

    # Use Triton for routing
    l_global_active = topk(l_dp_local, n_expts_act, apply_softmax=True, all_gather=True, y_indx=None)
    active_indx = l_global_active.indx
    expt_sizes = l_global_active.mask_metadata.col_sum
    dispatch_indx = l_global_active.mask_metadata.row_sorted_indx
    combine_indx = l_global_active.mask_metadata.col_sorted_indx
    x_global_metadata = make_ragged_tensor_metadata(expt_sizes, dispatch_indx.shape[0])

    # Allocate symmetric memory buffers
    dp_to_ep_buf = shmem.zeros((n_tokens_global * n_expts_act, d_model), dtype=x_dp_local.dtype)
    ep_to_dp_buf = shmem.zeros((n_tokens_local, d_model), dtype=x_dp_local.dtype)

    # Convert DP → EP
    y_ep_local = convert_dp_to_ep_iris(x_dp_local, expt_assignment, active_indx, dispatch_indx, shmem, dp_to_ep_buf)
    y_ep_local_metadata = remap_ragged_tensor_metadata(x_global_metadata, expt_map)

    # Use Triton's optimized matmul (tl.dot) - operates in-place
    y_ep_local = matmul(y_ep_local, w_ep_local, b_ep_local, a_ragged_metadata=y_ep_local_metadata)

    # Convert EP → DP
    y_dp_local = convert_ep_to_dp_iris(y_ep_local, expt_assignment, active_indx, combine_indx, shmem, ep_to_dp_buf)

    # Weighted average
    y_dp_local = y_dp_local.view(-1, n_expts_act, y_dp_local.shape[-1])
    z_dp_local, _ = reduce(y_dp_local, dim=1)

    return z_dp_local


def moe_iris_v2(x_dp_local, l_dp_local, w_ep_local, b_ep_local, expt_assignment, n_expts_act, shmem):
    """
    Iris MoE V2 - Ported from Triton reference
    """
    return mixture_of_expt_iris(x_dp_local, l_dp_local, w_ep_local, b_ep_local, expt_assignment, n_expts_act, shmem)


# ==============================================================================
# Export for testing
# ==============================================================================
__all__ = ["moe_iris_v2", "convert_dp_to_ep_iris", "convert_ep_to_dp_iris"]
