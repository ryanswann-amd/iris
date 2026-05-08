# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce-scatter collective operation — public API.

Triton only (no gluon support).
"""

from iris.ccl.utils import extract_group_info
from iris.ccl import launch_cache as _launch_cache


def reduce_scatter(output_tensor, input_tensor, ctx, op=None, group=None, async_op=False, config=None):
    """
    Reduce-scatter: each rank reduces its assigned tiles, stores locally.

    Args:
        output_tensor: Shape (M, N)
        input_tensor: Shape (M, N)
        ctx: Iris instance
        op: ReduceOp (only SUM supported)
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
    """
    from iris.ccl.config import Config
    from iris.ccl.utils import ReduceOp

    if op is None:
        op = ReduceOp.SUM
    if op != ReduceOp.SUM:
        raise ValueError(
            f"Only ReduceOp.SUM is currently supported, got {op}. "
            "Support for other operations will be added in a future release."
        )
    if config is None:
        config = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=1)
    if config.use_gluon:
        raise ValueError(
            "reduce_scatter does not support use_gluon=True. "
            "Gluon implementation is not available for reduce_scatter. "
            "Use default config (use_gluon=False)."
        )

    variant = getattr(config, "reduce_scatter_variant", "two_shot")
    if variant != "two_shot":
        raise ValueError(f"reduce_scatter only supports variant='two_shot', got '{variant}'.")

    # ----- K-820/K-861 fastpath: per-Config cached fused launch -----
    rank_global_for_key = ctx.get_rank()
    cache = getattr(config, "_iris_launch_cache", None)
    if cache is not None:
        _probe_key = (
            "reduce_scatter",
            variant,
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

    if output_tensor.shape[:2] != (M, N):
        raise ValueError(
            f"Output tensor shape {output_tensor.shape[:2]} does not match input shape {(M, N)}. "
            f"For reduce-scatter, output should have the same shape as input."
        )

    from iris.ccl.triton.reduce_scatter import launch

    launch(
        output_tensor,
        input_tensor,
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
        "reduce_scatter",
        variant,
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
        # Note: triton/reduce_scatter.launch signature is (output, input, ctx, ...)
        _l(_out, _in, _ctx, _a, _b, _c, _d, _e, _cf)

    _launch_cache.store(_cfg, _key, _fast)

    if not async_op:
        ctx.barrier()
