# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-gather collective operation — public API.

Routes to triton/ or gluon/ based on config.use_gluon.
"""

from iris.ccl.utils import extract_group_info
from iris.ccl import launch_cache as _launch_cache


def all_gather(output_tensor, input_tensor, ctx, group=None, async_op=False, config=None):
    """
    All-gather: each rank sends its input to all ranks.

    Output is (world_size * M, N) — inputs concatenated along dim 0.

    Args:
        output_tensor: Shape (world_size * M, N)
        input_tensor: Shape (M, N)
        ctx: Iris instance
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
    """
    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=64)

    # ----- K-820/K-861 fastpath: per-Config cached fused launch -----
    rank_global_for_key = ctx.get_rank()
    cache = getattr(config, "_iris_launch_cache", None)
    if cache is not None:
        _probe_key = (
            "all_gather",
            tuple(input_tensor.shape),
            tuple(output_tensor.shape),
            input_tensor.dtype,
            output_tensor.dtype,
            rank_global_for_key,
            id(group),
            id(config),
        )
        cached = cache.get(_probe_key)
        if cached is not None:
            _launch_cache.record_hit()
            cached(input_tensor, output_tensor, ctx)
            if not async_op:
                ctx.barrier()
            return
    _launch_cache.record_miss()

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    M, N = input_tensor.shape[:2]
    expected_output_shape = (world_size * M, N)
    if output_tensor.shape[:2] != expected_output_shape:
        raise ValueError(
            f"Output tensor shape {output_tensor.shape[:2]} does not match expected shape "
            f"{expected_output_shape}. Expected (world_size * M, N) = ({world_size * M}, {N})"
        )

    if config.use_gluon:
        from iris.ccl.gluon.all_gather import launch
    else:
        from iris.ccl.triton.all_gather import launch

    launch(
        input_tensor,
        output_tensor,
        ctx,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        config,
    )

    # Populate the per-Config fastpath closure now that the JIT cache is hot.
    _key = (
        "all_gather",
        tuple(input_tensor.shape),
        tuple(output_tensor.shape),
        input_tensor.dtype,
        output_tensor.dtype,
        rank_global,
        id(group),
        id(config),
    )
    _cfg = config
    _ri, _rg, _ws, _rs, _rstr = rank_in_group, rank_global, world_size, rank_start, rank_stride
    _launch = launch

    def _fast(_in, _out, _ctx, _l=_launch, _cf=_cfg, _a=_ri, _b=_rg, _c=_ws, _d=_rs, _e=_rstr):
        _l(_in, _out, _ctx, _a, _b, _c, _d, _e, _cf)

    _launch_cache.store(_cfg, _key, _fast)

    if not async_op:
        ctx.barrier()
