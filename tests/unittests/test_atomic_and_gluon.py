# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import iris.experimental.iris_gluon as iris_gl


@gluon.jit
def atomic_and_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    results,
    sem: gl.constexpr,
    scope: gl.constexpr,
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

    bit = (cur_rank // 32) % 2
    val = bit << (cur_rank % results.dtype.element_ty.primitive_bitwidth)
    acc = gl.full([BLOCK_SIZE], val, results.type.element_ty, layout)

    for target_rank in range(num_ranks):
        ctx.atomic_and(results + offsets, acc, target_rank, mask=mask, sem=sem, scope=scope)



pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(