# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris


@triton.jit
def atomic_cas_kernel(
    results,
    sem: tl.constexpr,
    scope: tl.constexpr,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    heap_bases: tl.tensor,
):
    # Cast constants to match results.dtype
    dtype = results.dtype.element_ty
    cmp = tl.full((), 0, dtype=dtype)  # scalar 0
    val = tl.full((), num_ranks, dtype=dtype)  # scalar num_ranks

    for target_rank in range(num_ranks):
        iris.atomic_cas(results, cmp, val, cur_rank, target_rank, heap_bases, sem=sem, scope=scope)



pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(