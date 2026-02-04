# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import iris.experimental.iris_gluon as iris_gl



pytestmark = pytest.mark.multi_rank_required

@gluon.jit
def atomic_xchg_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    results,
    val_ptr,
    sem: gl.constexpr,
    scope: gl.constexpr,
    cur_rank: gl.constexpr,
    num_ranks: gl.constexpr,
):
    ctx = IrisDeviceCtx.initialize(context_tensor)
    # Load value from single-element tensor passed from host using ctx.load
    # This is a workaround for Gluon's lack of 0D tensor support
    # Use ctx.load which handles the translation, loading from current rank (cur_rank)
    val = ctx.load(val_ptr, cur_rank)

    for target_rank in range(num_ranks):
        ctx.atomic_xchg(results, val, target_rank, mask=None, sem=sem, scope=scope)



pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(