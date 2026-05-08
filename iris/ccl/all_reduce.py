# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-reduce collective operation — public API.

Triton only (no gluon support).
"""

# Hoist imports out of the per-call hot path. These previously sat inside
# ``all_reduce()`` and were re-resolved every iteration -- contributing to the
# per-call Python wrapper overhead that K-786 v2 measured at ~17.5us mean
# across non-AR-one_shot collectives.
from iris.ccl.utils import extract_group_info, ReduceOp
from iris.ccl.config import Config
from iris.ccl.triton.all_reduce import (
    all_reduce_preamble as _preamble,
    launch as _triton_launch,
    capture_two_shot_descriptor as _capture_two_shot_descriptor,
)
from iris.ccl.triton._fused_launch_cache import (
    fused_launch_enabled,
    get_or_build_cache,
)


# Frozen tuple-based set; faster membership check than re-creating a list each
# call, and avoids the prior ``valid_variants = [...]`` allocation.
_VALID_AR_VARIANTS = frozenset({"atomic", "spinlock", "ring", "two_shot", "one_shot"})

# Default Config for the no-config caller path. Cached at module scope so we
# don't allocate a fresh Config each call (cheap, but the per-call wrapper
# overhead matters at <=16KB where launch_us dominates total latency).
_DEFAULT_CONFIG_NONE = None  # lazy: built on first use, then frozen


def all_reduce_preamble(output_tensor, input_tensor, ctx, config=None, workspace=None):
    """Prepare reusable workspace for all-reduce."""
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

    Notes:
        K-820 fused-launch fastpath: when ``config.fused_launch=True`` (or env
        ``IRIS_CCL_FUSED_LAUNCH=1``) and variant is ``two_shot``, steady-state
        calls bypass the iris-side dispatch wrappers (extract_group_info,
        variant if/elif, heap_bases lookup, kernel_kwargs construction) and
        invoke the cached Triton kernel directly. Targets the top-2 launch
        sub-phases identified by K-786 v2 (py_wrapper + cache_lookup ~ 63 %
        of two_shot launch_us).
    """
    # ---- Fastpath: two_shot + fused_launch enabled --------------------
    # We keep this stanza VERY short to minimize the warm-path Python cost.
    # The first call falls through to the slow path (which also captures
    # the descriptor); subsequent calls return after this block.
    if (
        config is not None
        and (getattr(config, "fused_launch", False) or fused_launch_enabled())
        and config.all_reduce_variant == "two_shot"
        and group is None  # group != None case rarely benchmarked; falls back to slow path
    ):
        cache = get_or_build_cache(config)
        # Tiny key: (M, N, dtype). World/group/block sizes are config-bound
        # and constant across all entries in this cache. If the user changes
        # block sizes mid-stream, build a new Config.
        shape = input_tensor.shape
        key = (shape[0], shape[1], input_tensor.dtype)
        desc = cache.get(key)
        if desc is not None:
            desc.invoke(input_tensor, output_tensor)
            if not async_op:
                ctx.barrier()
            return None

        # Cold path: run the full slow path AND capture a descriptor for
        # subsequent warm-path calls.
        ws = _slow_path_all_reduce(
            output_tensor, input_tensor, ctx, op, group, async_op, config, workspace
        )
        cache[key] = _capture_two_shot_descriptor(
            output_tensor, input_tensor, ctx, config, ctx.barrier
        )
        return ws

    return _slow_path_all_reduce(
        output_tensor, input_tensor, ctx, op, group, async_op, config, workspace
    )


def _slow_path_all_reduce(
    output_tensor, input_tensor, ctx, op, group, async_op, config, workspace
):
    """The original (HEAD) all_reduce implementation, factored out so the
    fastpath stanza in ``all_reduce`` can stay tight."""
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
            "all_reduce does not support use_gluon=True. "
            "Gluon implementation is not available for all_reduce. "
            "Use default config (use_gluon=False)."
        )

    variant = config.all_reduce_variant.lower()
    if variant not in _VALID_AR_VARIANTS:
        raise ValueError(
            f"Invalid all_reduce_variant: {variant}. Must be one of: "
            f"{', '.join(sorted(_VALID_AR_VARIANTS))}"
        )

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    workspace = _triton_launch(
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

    if not async_op:
        ctx.barrier()

    return workspace
