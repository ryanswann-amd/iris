# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce-scatter collective operation — public API.

Triton only (no gluon support).
"""

# Hoist imports out of the per-call hot path. These previously sat inside
# ``reduce_scatter()`` and were re-resolved every iteration -- contributing
# to the per-call Python wrapper overhead that K-786 v2 measured at
# ~17.5us mean across non-AR-one_shot collectives.
from iris.ccl.utils import extract_group_info, ReduceOp
from iris.ccl.config import Config
from iris.ccl.triton.reduce_scatter import (
    launch as _triton_launch,
    capture_reduce_scatter_descriptor as _capture_descriptor,
)
from iris.ccl.triton._fused_launch_cache import try_fused_fastpath


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

    Notes:
        K-871 fused-launch fastpath: when ``config.fused_launch=True``
        (or env ``IRIS_CCL_FUSED_LAUNCH=1``) and variant is ``two_shot``
        (the only supported variant), steady-state calls bypass iris-side
        dispatch (op validation, extract_group_info, output-shape check,
        heap_bases lookup, kwargs construction). Targets the top-2 launch
        sub-phases identified by K-786 v2.
    """
    if try_fused_fastpath(
        collective_name="reduce_scatter",
        config=config,
        input_tensor=input_tensor,
        output_tensor=output_tensor,
        ctx=ctx,
        group=group,
        async_op=async_op,
        slow_path=lambda: _slow_path_reduce_scatter(output_tensor, input_tensor, ctx, op, group, async_op, config),
        capture=lambda: _capture_descriptor(
            output_tensor, input_tensor, ctx,
            *extract_group_info(group, ctx),
            config,
        ),
        extra_guard=(config is not None and getattr(config, "reduce_scatter_variant", "two_shot") == "two_shot"),
    ):
        return

    _slow_path_reduce_scatter(output_tensor, input_tensor, ctx, op, group, async_op, config)


def _slow_path_reduce_scatter(output_tensor, input_tensor, ctx, op, group, async_op, config):
    """The original reduce_scatter implementation, factored out so the
    fastpath stanza in ``reduce_scatter`` can stay tight."""
    if op is None:
        op = ReduceOp.SUM
    elif op != ReduceOp.SUM:
        raise ValueError(
            f"Only ReduceOp.SUM is currently supported, got {op}. "
            "Support for other operations will be added in a future release."
        )
    if config is None:
        config = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=1)
    elif config.use_gluon:
        raise ValueError(
            "reduce_scatter does not support use_gluon=True. "
            "Gluon implementation is not available for reduce_scatter. "
            "Use default config (use_gluon=False)."
        )

    variant = getattr(config, "reduce_scatter_variant", "two_shot")
    if variant != "two_shot":
        raise ValueError(f"reduce_scatter only supports variant='two_shot', got '{variant}'.")

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)
    M, N = input_tensor.shape[:2]

    if output_tensor.shape[:2] != (M, N):
        raise ValueError(
            f"Output tensor shape {output_tensor.shape[:2]} does not match input shape {(M, N)}. "
            f"For reduce-scatter, output should have the same shape as input."
        )

    _triton_launch(
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

    if not async_op:
        ctx.barrier()
