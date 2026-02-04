# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for tile-level all-reduce primitives.
"""

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris
import iris.x



pytestmark = pytest.mark.multi_rank_required

@triton.jit
def x_all_reduce_atomic_kernel(
    input_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_in_m: tl.constexpr,
    stride_in_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_n: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Kernel that iterates over tiles and calls all_reduce_atomic for each."""
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, grid_size):  # Stride by grid size to avoid overlap
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        # Load local tile data
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        src_ptr = input_ptr + rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        local_data = tl.load(src_ptr, mask=mask, other=0.0)

        # Create Tile with loaded data and views
        tile = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, local_data)
        dst_view = iris.x.TensorView(output_ptr, M, N, stride_out_m, stride_out_n)
        ctx = iris.x.DeviceContext(cur_rank, world_size, heap_bases)

        iris.x.all_reduce_atomic(tile, dst_view, ctx)


@triton.jit
def x_all_reduce_one_shot_kernel(
    input_ptr,
    temp_buffer,
    output_ptr,
    locks,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_in_m: tl.constexpr,
    stride_in_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_n: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Kernel that iterates over tiles and calls all_reduce_one_shot for each."""
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, grid_size):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        # Load local tile data from input
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        src_ptr = input_ptr + rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        local_data = tl.load(src_ptr, mask=mask, other=0.0)

        # Store to temp_buffer (avoid race condition) and signal ready
        temp_ptr = temp_buffer + rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        tl.store(temp_ptr, local_data, mask=mask, cache_modifier=".wt")
        tl.debug_barrier()  # Ensures all stores are visible before the atomic_xchg
        tl.atomic_xchg(locks + tile_id, 1, sem="release", scope="gpu")  # Release ensures prior stores visible

        # Create Tile with data and views
        tile = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, local_data)
        src_view = iris.x.TensorView(temp_buffer, M, N, stride_in_m, stride_in_n)
        dst_view = iris.x.TensorView(output_ptr, M, N, stride_out_m, stride_out_n)
        ctx = iris.x.DeviceContext(cur_rank, world_size, heap_bases)

        iris.x.all_reduce_one_shot(tile, src_view, dst_view, locks, ctx)


@triton.jit
def x_all_reduce_two_shot_kernel(
    input_ptr,
    temp_buffer,
    output_ptr,
    locks,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_in_m: tl.constexpr,
    stride_in_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_n: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Kernel that iterates over tiles and calls all_reduce_two_shot for each."""
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, grid_size):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        # Load local tile data from input
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        src_ptr = input_ptr + rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        local_data = tl.load(src_ptr, mask=mask, other=0.0)

        # Store to temp_buffer (avoid race condition) and signal ready
        temp_ptr = temp_buffer + rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        tl.store(temp_ptr, local_data, mask=mask, cache_modifier=".wt")
        tl.debug_barrier()  # Ensures all stores are visible before the atomic_xchg
        tl.atomic_xchg(locks + tile_id, 1, sem="release", scope="gpu")  # Release ensures prior stores visible

        # Create Tile with data and views
        tile = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, local_data)
        src_view = iris.x.TensorView(temp_buffer, M, N, stride_in_m, stride_in_n)
        dst_view = iris.x.TensorView(output_ptr, M, N, stride_out_m, stride_out_n)
        ctx = iris.x.DeviceContext(cur_rank, world_size, heap_bases)

        iris.x.all_reduce_two_shot(tile, src_view, dst_view, locks, cur_rank, world_size, ctx)


@triton.jit
def x_all_reduce_spinlock_kernel(
    input_ptr,
    output_ptr,
    locks_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_in_m: tl.constexpr,
    stride_in_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_n: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Kernel that iterates over tiles and calls all_reduce_spinlock for each."""
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, grid_size):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        # Load local tile data
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        src_ptr = input_ptr + rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        local_data = tl.load(src_ptr, mask=mask, other=0.0)

        # Create Tile with loaded data and views
        tile = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, local_data)
        dst_view = iris.x.TensorView(output_ptr, M, N, stride_out_m, stride_out_n)
        ctx = iris.x.DeviceContext(cur_rank, world_size, heap_bases)

        iris.x.all_reduce_spinlock(tile, dst_view, locks_ptr, ctx)



pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(