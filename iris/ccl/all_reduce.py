# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-reduce collective operation — public API.

Triton only (no gluon support).
"""

# Hoist imports out of the per-call hot path. These previously sat inside
# ``all_reduce()`` and were re-resolved every iteration -- contributing to
# Python wrapper overhead that dominates launch_us at small message sizes
# (K-786 v2: ~63 % of two_shot launch_us is iris-side dispatch + Triton
# binder lookup).
from iris.ccl.utils import extract_group_info, ReduceOp
from iris.ccl.config import Config
from iris.ccl.triton.all_reduce import (
    all_reduce_preamble as _preamble,
    launch as _triton_launch,
    persistent_all_reduce_two_shot as _two_shot_kernel,
)


_VALID_AR_VARIANTS = ("atomic", "spinlock", "ring", "two_shot", "one_shot")


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
        K-820 fused-launch fastpath: when ``config.fused_launch`` is True and
        variant is ``two_shot`` with ``group=None``, steady-state calls bypass
        the iris-side dispatch chain (extract_group_info, variant if/elif,
        heap_bases lookup, kernel_kwargs construction) and replay a cached
        ``kernel[grid](*args)`` call. The first call falls through to the
        full slow path and captures a per-(M,N,dtype) args tuple in
        ``config._fused_cache``; subsequent calls re-use it directly.
    """
    # ---- Fastpath probe (two_shot + fused_launch enabled) -------------
    # The presence of ``config._fused_cache`` is itself the activation gate
    # (only fused_launch=True + two_shot + group=None ever populates it), so
    # the warm path skips per-call lookup of ``fused_launch`` and
    # ``all_reduce_variant``. The bound ``kernel[grid]`` launcher is captured
    # once so warm calls also avoid Triton's ``__getitem__`` allocation.
    if config is not None:
        cache = config.__dict__.get("_fused_cache")
        if cache is not None and group is None:
            shape = input_tensor.shape
            cached = cache.get((shape[0], shape[1], input_tensor.dtype))
            if cached is not None:
                # Cached tuple: (bound kernel[grid] launcher, args, kwargs).
                # Tuple unpack + *args call beats a closure (closure adds a
                # Python frame); measured ~0.5us better on MI300X bf16.
                launcher, args, kwargs = cached
                launcher(input_tensor, output_tensor, *args, **kwargs)
                if not async_op:
                    ctx.barrier()
                return None

        # Cold-path capture: only when user opted in AND variant is two_shot
        # (and the (M, N, dtype) cell was a miss above, or cache didn't exist).
        if (
            group is None
            and config.fused_launch
            and config.all_reduce_variant == "two_shot"
        ):
            ws = _slow_path(output_tensor, input_tensor, ctx, op, group, async_op, config, workspace)
            if cache is None:
                cache = {}
                config._fused_cache = cache
            M, N = input_tensor.shape[:2]
            rank_global = ctx.get_rank()
            cache[(M, N, input_tensor.dtype)] = (
                _two_shot_kernel[(config.comm_sms,)],
                (
                    M,
                    N,
                    input_tensor.stride(0),
                    input_tensor.stride(1),
                    output_tensor.stride(0),
                    output_tensor.stride(1),
                    ctx.get_heap_bases(),
                    rank_global,
                    rank_global,
                    ctx.get_num_ranks(),
                    0,
                    1,
                    config.block_size_m,
                    config.block_size_n,
                    config.swizzle_size,
                    config.comm_sms,
                    config.num_xcds,
                    config.chunk_size,
                    config.all_reduce_distribution,
                ),
                {"num_warps": 8, "num_stages": 1, "waves_per_eu": 1},
            )
            return ws

    return _slow_path(output_tensor, input_tensor, ctx, op, group, async_op, config, workspace)


def _slow_path(output_tensor, input_tensor, ctx, op, group, async_op, config, workspace):
    """The original (HEAD) all_reduce implementation."""
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
    if variant not in _VALID_AR_VARIANTS:
        raise ValueError(
            f"Invalid all_reduce_variant: {variant}. Must be one of: "
            f"{', '.join(_VALID_AR_VARIANTS)}"
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
