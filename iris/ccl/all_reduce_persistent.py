# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Persistent / resident all-reduce — public API.

See :mod:`iris.ccl.triton.all_reduce_persistent` for design notes.
"""

from iris.ccl.utils import extract_group_info


def all_reduce_persistent_preamble(
    output_tensor, input_tensor, ctx, max_iters, config=None, workspace=None
):
    """Allocate the doorbell / done / iter-barrier workspace.

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
                        barrier between iterations.  Disable ONLY if the
                        input is constant across iterations (eg microbench
                        sweep) — otherwise iter K+1 may see iter K's
                        partially-written tiles on peer ranks.

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


def all_reduce_persistent_doorbell_start(
    output_tensor, input_tensor, ctx, max_iters, config=None, workspace=None, group=None
):
    """Launch the doorbell-driven persistent kernel.

    Returns *immediately* — the kernel keeps running in the background.
    Drive it with :func:`all_reduce_persistent_doorbell_step` and shut it
    down with :func:`all_reduce_persistent_doorbell_stop`.
    """
    from iris.ccl.config import Config
    from iris.ccl.triton.all_reduce_persistent import launch_persistent_doorbell

    if config is None:
        config = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=0)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)
    workspace = all_reduce_persistent_preamble(
        output_tensor, input_tensor, ctx, max_iters=max_iters, config=config, workspace=workspace
    )
    return launch_persistent_doorbell(
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
    )


def all_reduce_persistent_doorbell_step(workspace):
    """Trigger one doorbell-driven iteration and wait for completion.

    Writes ``1`` into the next doorbell slot (releasing the kernel for one
    iter) and busy-waits on the corresponding ``done`` slot.

    Returns the iteration index that was just consumed.
    """
    import torch

    from iris.ccl.triton.all_reduce_persistent import PersistentAllReduceWorkspace

    assert isinstance(workspace, PersistentAllReduceWorkspace)
    if workspace.next_iter >= workspace.max_iters:
        raise RuntimeError(
            f"persistent doorbell exhausted ({workspace.next_iter} == max_iters)"
        )
    i = workspace.next_iter
    # Always issue host-side writes on the device default stream — using a
    # NCCL-tagged or test-wrapped current stream can deadlock against the
    # persistent kernel's tight polling loop on AMD hardware (observed
    # empirically: fill_ never completes when there's a concurrent .cv-load
    # spin from the persistent kernel and NCCL had previously bound a
    # different stream).
    default = torch.cuda.default_stream()
    with torch.cuda.stream(default):
        workspace.doorbell[i].fill_(1)
        # Spin on the done slot.  ``.item()`` forces a host-side read each
        # loop and only synchronizes the current (default) stream — it does
        # NOT wait on the persistent stream.
        done = workspace.done
        while int(done[i].item()) != 1:
            pass
    workspace.next_iter = i + 1
    return i


def all_reduce_persistent_doorbell_stop(workspace, ctx):
    """Tell the persistent kernel to exit and join the launch."""
    import torch

    from iris.ccl.triton.all_reduce_persistent import shutdown_doorbell

    shutdown_doorbell(workspace)
    # Make sure the sentinel write has hit the device, then wait for the
    # persistent kernel's stream to drain.
    torch.cuda.current_stream().synchronize()
    if workspace.persistent_stream is not None:
        workspace.persistent_stream.synchronize()
        workspace.persistent_stream = None
    ctx.barrier()
