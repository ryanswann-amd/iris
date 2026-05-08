# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Persistent / resident all-reduce — public API.

See :mod:`iris.ccl.triton.all_reduce_persistent` for design notes.
"""

from iris.ccl.utils import extract_group_info


def all_reduce_persistent_preamble(output_tensor, input_tensor, ctx, max_iters, config=None, workspace=None):
    """Allocate the iter-barrier workspace for the persistent burst kernel.

    Args:
        output_tensor:  Symmetric output tensor (M, N) — will receive the sum.
        input_tensor:   Symmetric input tensor (M, N) — local rank's data.
        ctx:            Iris context.
        max_iters:      Number of iteration slots to provision.
        config:         Optional :class:`iris.ccl.Config`.
        workspace:      Existing workspace to reuse / re-zero.

    Returns:
        :class:`iris.ccl.triton.all_reduce_persistent.PersistentAllReduceWorkspace`.
    """
    from iris.ccl.config import Config
    from iris.ccl.triton.all_reduce_persistent import (
        persistent_all_reduce_preamble as _preamble,
    )

    if config is None:
        config = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=0)
    return _preamble(
        output_tensor,
        input_tensor,
        ctx,
        config=config,
        max_iters=max_iters,
        workspace=workspace,
    )


def all_reduce_persistent_burst(
    output_tensor,
    input_tensor,
    ctx,
    num_iters,
    config=None,
    workspace=None,
    group=None,
    async_op=False,
    use_barrier=True,
):
    """Run ``num_iters`` two-shot all-reduces in a single persistent kernel launch.

    The launch envelope (~50 µs on MI300X per K-796) is paid **once** for
    the whole window of ``num_iters`` iterations rather than once per call.

    Inputs/outputs must be the same on every iteration (the kernel re-reads
    the same input pointers each iter).  This is the natural fast-path for
    workloads that re-use the same input tensor — most notably benchmark
    sweeps and tightly-batched inference.

    Args:
        output_tensor:  Symmetric output tensor (M, N).
        input_tensor:   Symmetric input tensor (M, N).
        ctx:            Iris context.
        num_iters:      How many iterations to fuse.
        config:         Optional :class:`iris.ccl.Config`.  Only the
                        ``two_shot`` variant is supported.
        workspace:      Workspace from :func:`all_reduce_persistent_preamble`
                        (auto-allocated if None).
        group:          Optional process group.
        async_op:       If True, skip the trailing ``ctx.barrier()``.
        use_barrier:    If True (default), insert a counter-based cross-rank
                        barrier between iterations.  This is the only safe
                        setting for general use — the barrier guarantees
                        iter K+1 only starts reading peer outputs after every
                        rank has finished writing iter K.  Disable ONLY if
                        the input is provably constant across iterations
                        (e.g. a latency microbenchmark that reuses the same
                        input buffer); the resulting numbers expose the raw
                        launch-overhead reduction but are not safe when peer
                        inputs change between iters.

    Returns:
        The persistent workspace (reusable across calls with identical
        ``num_iters`` and tensor shapes).
    """
    from iris.ccl.config import Config
    from iris.ccl.triton.all_reduce_persistent import launch_persistent_burst

    if config is None:
        config = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=0)
    if config.use_gluon:
        raise ValueError("all_reduce_persistent does not support use_gluon=True.")

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    if workspace is None or not getattr(workspace, "prepared", False) or workspace.max_iters < num_iters:
        workspace = all_reduce_persistent_preamble(
            output_tensor, input_tensor, ctx, max_iters=num_iters, config=config, workspace=workspace
        )

    workspace = launch_persistent_burst(
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
        num_iters=num_iters,
        use_barrier=use_barrier,
    )

    if not async_op:
        ctx.barrier()
    return workspace
