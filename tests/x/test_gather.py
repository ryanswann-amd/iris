#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Tests for iris.x.gather primitive (single-rank gather)."""

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris
import iris.x



pytestmark = pytest.mark.multi_rank_required

@triton.jit
def gather_kernel(
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
    source_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Test kernel that uses gather to pull a single tile from source_rank."""
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, grid_size):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        # Create tile and views
        tile = iris.x.TileView(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
        src_view = iris.x.TensorView(input_ptr, M, N, stride_in_m, stride_in_n)
        ctx = iris.x.DeviceContext(cur_rank, world_size, heap_bases)

        # Use gather to pull tile from source_rank
        data = iris.x.gather(tile, src_view, source_rank, ctx)

        # Store to output
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_m = rm < M
        mask_n = rn < N
        mask = mask_m[:, None] & mask_n[None, :]
        out_ptr = output_ptr + rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        tl.store(out_ptr, data, mask=mask)



pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(