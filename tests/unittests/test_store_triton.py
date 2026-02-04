# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris


@triton.jit
def store_kernel(
    data,
    results,
    destination_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < BLOCK_SIZE

    # Load the data from src for this block
    value = tl.load(data + offsets, mask=mask)

    # Store data to all ranks
    # Doesn't matter which rank stores at the end, the data should all be the same at the end.
    for dst_rank in range(num_ranks):
        iris.store(results + offsets, value, destination_rank, dst_rank, heap_bases, mask=mask)



pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(