# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-reduce collective operation — public API.

Triton only (no gluon support).
"""

from iris.ccl.utils import extract_group_info
from iris.ccl import launch_cache as _launch_cache


def all_reduce_preamble(output_tensor, input_tensor, ctx, config=None, workspace=None):
    """Prepare reusable workspace for all-reduce."""
    from iris.ccl.triton.all_reduce import all_reduce_preamble as _preamble

    return _preamble(output_tensor, input_tensor, ctx, config=config, workspace=workspace)


def all_reduce(output_tensor, input_tensor, ctx, op=None, group=None, async_op=False, config=None, workspace=None):
    """
    All-reduce: sum inputs across all ranks, result on every rank.

    Args:
        output_tensor: Shape (M, N)
        input_tensor: Shape (M, N)
        ctx: Iris instance
        op: ReduceOp (only SUM supported)
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
        workspace: Reusable workspace from all_reduce_preamble
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
            "all_reduce does not support use_gluon=True. "
            "Gluon implementation is not available for all_reduce. "
            "Use default config (use_gluon=False)."
        )

    variant = config.all_reduce_variant.lower()
    valid_variants = ["atomic", "spinlock", "ring", "two_shot", "one_shot"]
    if variant not in valid_variants:
        raise ValueError(f"Invalid all_reduce_variant: {variant}. Must be one of: {', '.join(valid_variants)}")

    # ----- K-820/K-861 fastpath: per-Config cached fused launch -----
    # The fastpath captures the (kernel_fn, group_info, frozen kwargs) tuple
    # for the given (M, N, dtype, world, variant). It is correctness-safe
    # for variants that do NOT require per-call workspace mutation:
    #   - two_shot: stateless, safe
    #   - one_shot: needs preamble (output.zero_() + ctx.barrier()) — the
    #     fastpath still includes that preamble inside the closure so the
    #     contract is preserved
    #   - ring/spinlock/atomic: stateful workspaces; we conservatively fall
    #     through to the cold path (workspace pointer changes break the
    #     cached arg list)
    #
    # The probe key intentionally includes ``id(workspace)`` so that the
    # fastpath becomes invalid the moment the user supplies a freshly
    # allocated workspace.
    fastpath_eligible = variant in ("two_shot", "one_shot") and workspace is None
    rank_global_for_key = ctx.get_rank()
    cache = getattr(config, "_iris_launch_cache", None)
    if fastpath_eligible and cache is not None:
        _probe_key = (
            "all_reduce",
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
            ws = cached(input_tensor, output_tensor, ctx)
            if not async_op:
                ctx.barrier()
            return ws
    if fastpath_eligible:
        _launch_cache.record_miss()

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    from iris.ccl.triton.all_reduce import launch

    workspace = launch(
        output_tensor,
        input_tensor,
        ctx,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        config,
        workspace,
        group=group,
    )

    if workspace is not None:
        workspace.prepared = False

    # Populate the per-Config fastpath closure for the eligible variants.
    if fastpath_eligible:
        _key = (
            "all_reduce",
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
        _group = group

        def _fast(_in, _out, _ctx, _l=_launch, _cf=_cfg, _a=_ri, _b=_rg, _c=_ws, _d=_rs, _e=_rstr, _g=_group):
            ws_local = _l(_out, _in, _ctx, _a, _b, _c, _d, _e, _cf, None, group=_g)
            if ws_local is not None:
                ws_local.prepared = False
            return ws_local

        _launch_cache.store(_cfg, _key, _fast)

    if not async_op:
        ctx.barrier()

    return workspace
