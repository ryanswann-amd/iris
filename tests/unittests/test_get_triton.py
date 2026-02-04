# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris


# TODO: Separate this kernel out in the following categories:
# 1. for local get.
# 2. for remote get with one other rank.
# 3. for remote get with more than one rank (if num_ranks > 2).
@triton.jit
def get_kernel(
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

    acc = tl.zeros([BLOCK_SIZE], dtype=data.type.element_ty)

    # Loop over all ranks, get the stored data.
    # load to local register, accumulate.
    for target_rank in range(num_ranks):
        iris.get(data + offsets, results + offsets, cur_rank, target_rank, heap_bases, mask=mask)
        acc += tl.load(results + offsets, mask=mask)

    # Store the accumulated value back to the output.
    tl.store(results + offsets, acc, mask=mask)



pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(