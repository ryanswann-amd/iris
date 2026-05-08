# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-to-all collective operation — public API.

Routes to triton/ or gluon/ based on config.use_gluon.
"""

from iris.ccl.utils import extract_group_info
from iris.ccl import launch_cache as _launch_cache


def all_to_all(output_tensor, input_tensor, ctx, group=None, async_op=False, config=None):
    """
    All-to-all: each rank sends a chunk to every other rank.

    Input/output shape: (M, N * world_size).

    Args:
        output_tensor: Shape (M, N * world_size)
        input_tensor: Shape (M, N * world_size)
        ctx: Iris instance
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
    """
    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=128)

    # ----- K-820/K-861 fastpath: per-Config cached fused launch -----
    # Only valid when caller supplies a stable Config object across calls.
    # Falls through to the cold path on first call or on any
    # observable change to shape/dtype/world/Config field.
    rank_global_for_key = ctx.get_rank()
    # world_size is part of the cache key but is also needed for the
    # cold path; resolve it lazily via extract_group_info only on miss
    # to keep the hit path under R-PY-WRAPPER-FUSE budget.
    cache = getattr(config, "_iris_launch_cache", None)
    if cache is not None:
        # Use a fast probe key built from cheap fields; world_size enters
        # via the cached closure's identity (it is captured at cold-path
        # time and re-validated on misses).
        _probe_key = (
            "all_to_all",
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

    if config.use_gluon:
        from iris.ccl.gluon.all_to_all import launch
    else:
        from iris.ccl.triton.all_to_all import launch

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
        "all_to_all",
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
