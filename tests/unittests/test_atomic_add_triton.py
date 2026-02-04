# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris



pytestmark = pytest.mark.multi_rank_required

@triton.jit
def atomic_add_kernel(
    results,
    sem: tl.constexpr,
    scope: tl.constexpr,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    acc = tl.full([BLOCK_SIZE], 1, dtype=results.type.element_ty)

    # Loop over all ranks, get the stored data.
    # atomic_add acc into results.
    for target_rank in range(num_ranks):
        iris.atomic_add(
            results + offsets,
            acc,
            cur_rank,
            target_rank,
            heap_bases,
            mask,
            sem=sem,
            scope=scope,
        )



pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(