# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import iris.experimental.iris_gluon as iris_gl


# TODO: Separate this kernel out in the following categories:
# 1. for local put.
# 2. for remote put with one other rank.
# 3. for remote put with more than one rank (if num_ranks > 2).

pytestmark = pytest.mark.multi_rank_required

@gluon.jit
def put_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    data,
    results,
    cur_rank: gl.constexpr,
    num_ranks: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
):
    ctx = IrisDeviceCtx.initialize(context_tensor)
    pid = gl.program_id(0)
    block_start = pid * BLOCK_SIZE
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offsets = block_start + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < BLOCK_SIZE

    # Put data in all ranks
    # Doesn't matter which rank stores at the end, the data should all be the same at the end.
    for target_rank in range(num_ranks):
        ctx.put(data + offsets, results + offsets, target_rank, mask=mask)



pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(