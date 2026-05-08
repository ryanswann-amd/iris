# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-to-all collective operation — public API.

Routes to triton/ or gluon/ based on config.use_gluon.
"""

# Hoist imports out of the per-call hot path. These previously sat inside
# ``all_to_all()`` and were re-resolved every iteration -- contributing to the
# per-call Python wrapper overhead that K-786 v2 measured at ~17.5us mean
# across non-AR-one_shot collectives.
from iris.ccl.utils import extract_group_info
from iris.ccl.config import Config
from iris.ccl.triton.all_to_all import (
    launch as _triton_launch,
    capture_all_to_all_descriptor as _capture_descriptor,
)
from iris.ccl.triton._fused_launch_cache import (
    fused_launch_enabled,
    get_or_build_cache,
)


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

    Notes:
        K-871 fused-launch fastpath: when ``config.fused_launch=True``
        (or env ``IRIS_CCL_FUSED_LAUNCH=1``) and the gluon backend is not
        in use, steady-state calls bypass iris-side dispatch wrappers.
        Targets the top-2 launch sub-phases identified by K-786 v2
        (py_wrapper + cache_lookup).
    """
    # ---- Fastpath: triton + fused_launch enabled ---------------------
    if (
        config is not None
        and (getattr(config, "fused_launch", False) or fused_launch_enabled())
        and not config.use_gluon
        and group is None  # group != None case rarely benchmarked; falls back
    ):
        cache = get_or_build_cache(config)
        shape = input_tensor.shape
        key = ("all_to_all", shape[0], shape[1], input_tensor.dtype)
        desc = cache.get(key)
        if desc is not None:
            desc.invoke(input_tensor, output_tensor)
            if not async_op:
                ctx.barrier()
            return

        # Cold path: full slow path AND descriptor capture.
        _slow_path_all_to_all(output_tensor, input_tensor, ctx, group, async_op, config)
        rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)
        cache[key] = _capture_descriptor(
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
        return

    _slow_path_all_to_all(output_tensor, input_tensor, ctx, group, async_op, config)


def _slow_path_all_to_all(output_tensor, input_tensor, ctx, group, async_op, config):
    """The original all_to_all implementation, factored out so the
    fastpath stanza in ``all_to_all`` can stay tight."""
    if config is None:
        config = Config(block_size_m=32, block_size_n=128)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    if config.use_gluon:
        from iris.ccl.gluon.all_to_all import launch
    else:
        launch = _triton_launch

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

    if not async_op:
        ctx.barrier()
