# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris



pytestmark = pytest.mark.multi_rank_required

@triton.jit
def copy_get_kernel(
    data,
    results,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    """GET: cur_rank == to_rank (pull from remote)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    for target_rank in range(num_ranks):
        src_data = data + BLOCK_SIZE * cur_rank
        dest_data = results + BLOCK_SIZE * target_rank
        iris.copy(src_data + offsets, dest_data + offsets, target_rank, cur_rank, cur_rank, heap_bases, mask)


@triton.jit
def copy_put_kernel(
    data,
    results,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    """PUT: cur_rank == from_rank (push to remote)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    for target_rank in range(num_ranks):
        src_data = data + BLOCK_SIZE * cur_rank
        dest_data = results + BLOCK_SIZE * cur_rank
        iris.copy(src_data + offsets, dest_data + offsets, cur_rank, target_rank, cur_rank, heap_bases, mask)


@triton.jit
def copy_local_kernel(
    data,
    results,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    """LOCAL: from_rank == to_rank == cur_rank"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    for i in range(num_ranks):
        src_data = data + BLOCK_SIZE * i
        dest_data = results + BLOCK_SIZE * i
        iris.copy(src_data + offsets, dest_data + offsets, cur_rank, cur_rank, cur_rank, heap_bases, mask)



pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(