# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris


# TODO: Separate this kernel out in the following categories:
# 1. for local put.
# 2. for remote put with one other rank.
# 3. for remote put with more than one rank (if num_ranks > 2).
@triton.jit
def put_kernel(
    data,
    results,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    # Put data in all ranks
    # Doesn't matter which rank stores at the end, the data should all be the same at the end.
    for target_rank in range(num_ranks):
        iris.put(data + offsets, results + offsets, cur_rank, target_rank, heap_bases, mask=mask)



pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(