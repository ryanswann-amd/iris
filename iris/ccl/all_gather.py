# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-gather collective operation — public API.

Routes to triton/ or gluon/ based on config.use_gluon.
"""

# Hoist imports out of the per-call hot path. These previously sat inside
# ``all_gather()`` and were re-resolved every iteration -- contributing to the
# per-call Python wrapper overhead that K-786 v2 measured at ~17.5us mean
# across non-AR-one_shot collectives.
from iris.ccl.utils import extract_group_info
from iris.ccl.config import Config
from iris.ccl.triton.all_gather import (
    launch as _triton_launch,
    capture_all_gather_descriptor as _capture_descriptor,
)
from iris.ccl.triton._fused_launch_cache import (
    fused_launch_enabled,
    get_or_build_cache,
)


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

    Notes:
        K-871 fused-launch fastpath: when ``config.fused_launch=True``
        (or env ``IRIS_CCL_FUSED_LAUNCH=1``) and the gluon backend is not
        in use, steady-state calls bypass the iris-side dispatch wrappers
        (extract_group_info, output-shape validation, variant if/elif,
        heap_bases lookup, kernel kwargs construction) and invoke the
        cached Triton kernel directly. Targets the top-2 launch sub-phases
        identified by K-786 v2 (py_wrapper + cache_lookup).
    """
    # ---- Fastpath: triton + fused_launch enabled ---------------------
    # Keep this stanza VERY short to minimise warm-path Python cost.
    # The first call falls through to the slow path (which also captures
    # the descriptor); subsequent calls return after this block.
    if (
        config is not None
        and (getattr(config, "fused_launch", False) or fused_launch_enabled())
        and not config.use_gluon
        and group is None  # group != None case rarely benchmarked; falls back to slow path
    ):
        cache = get_or_build_cache(config)
        shape = input_tensor.shape
        # Tag with collective name so a single Config can host multiple
        # collectives without key collisions.
        key = ("all_gather", shape[0], shape[1], input_tensor.dtype)
        desc = cache.get(key)
        if desc is not None:
            desc.invoke(input_tensor, output_tensor)
            if not async_op:
                ctx.barrier()
            return

        # Cold path: run the full slow path AND capture a descriptor for
        # subsequent warm-path calls.
        _slow_path_all_gather(output_tensor, input_tensor, ctx, group, async_op, config)
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

    _slow_path_all_gather(output_tensor, input_tensor, ctx, group, async_op, config)


def _slow_path_all_gather(output_tensor, input_tensor, ctx, group, async_op, config):
    """The original all_gather implementation, factored out so the
    fastpath stanza in ``all_gather`` can stay tight."""
    if config is None:
        config = Config(block_size_m=32, block_size_n=64)

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
