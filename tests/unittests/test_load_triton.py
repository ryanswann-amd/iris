# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris


@triton.jit
def load_kernel(
    data,
    results,
    source_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    pid = tl.program_id(0)

    partner = int((source_rank + num_ranks // 2) % num_ranks)
    # Compute start index of this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Guard for out-of-bounds accesses
    mask = offsets < BLOCK_SIZE
    result = iris.load(data + offsets, source_rank, partner, heap_bases, mask=mask)
    tl.store(results + offsets, result, mask=mask)



pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(