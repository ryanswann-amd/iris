# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-to-all collective operation — public API.

Routes to triton/ or gluon/ based on config.use_gluon.
"""

from iris.ccl.utils import extract_group_info


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
    from iris.ccl.config import _detect_arch, default_config
    from iris.ccl.validation import warn_if_unvalidated

    if config is None:
        # Per-rank input bytes drive the (arch, collective, message-size) lookup
        # in iris/ccl/config.py::_DEFAULTS_TABLE. The warn-vs-silent policy on
        # cells without on-target evidence is applied here at the call site
        # (round-10 Architect requirement) so the contract is visible per
        # collective rather than implicit in a shared helper.
        message_bytes = input_tensor.numel() * input_tensor.element_size()
        warn_if_unvalidated(_detect_arch(), "all_to_all", message_bytes)
        config = default_config("all_to_all", message_bytes)

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

    if not async_op:
        ctx.barrier()
