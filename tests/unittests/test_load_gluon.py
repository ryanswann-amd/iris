# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import iris.experimental.iris_gluon as iris_gl


@gluon.jit
def load_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    data,
    results,
    source_rank: gl.constexpr,
    num_ranks: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
):
    ctx = IrisDeviceCtx.initialize(context_tensor)
    pid = gl.program_id(0)

    partner = int((source_rank + num_ranks // 2) % num_ranks)
    # Compute start index of this block
    block_start = pid * BLOCK_SIZE
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offsets = block_start + gl.arange(0, BLOCK_SIZE, layout=layout)

    # Guard for out-of-bounds accesses
    mask = offsets < BLOCK_SIZE
    result = ctx.load(data + offsets, partner, mask=mask)
    gl.store(results + offsets, result, mask=mask)



pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(