# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for collective operations with ProcessGroups.

These tests verify that iris.ccl operations work correctly with torch.distributed
ProcessGroups, including both consecutive and strided rank patterns.

Requires at least 4 GPUs to run (for meaningful subgroup testing).
"""

import pytest
import torch
import torch.distributed as dist
import iris
from iris.ccl import Config



pytestmark = pytest.mark.multi_rank_required

def _get_world_info():
    """Get world size and rank, skip if not enough ranks."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size < 4:
        pytest.skip(f"Process group tests require at least 4 ranks, got {world_size}")

    return world_size, rank


def _create_consecutive_groups(world_size, group_size=2):
    """
    Create consecutive (TP-like) groups.

    Example with world_size=4, group_size=2:
        Group 0: [0, 1]
        Group 1: [2, 3]

    Note: dist.new_group() is a collective operation - ALL ranks must call it,
    even if they're not part of the group being created.
    """
    groups = []
    for i in range(0, world_size, group_size):
        ranks = list(range(i, min(i + group_size, world_size)))
        if len(ranks) == group_size:
            # All ranks must call new_group collectively
            groups.append(dist.new_group(ranks))
        else:
            # For incomplete groups, still need a placeholder but all ranks
            # participated in all group creations above
            groups.append(None)
    return groups


def _create_strided_groups(world_size, num_groups=2):
    """
    Create strided (DP-like) groups.

    Example with world_size=4, num_groups=2:
        Group 0: [0, 2]  (stride=2)
        Group 1: [1, 3]  (stride=2)

    Note: dist.new_group() is a collective operation - ALL ranks must call it,
    even if they're not part of the group being created.
    """
    groups = []
    stride = num_groups

    for i in range(num_groups):
        ranks = list(range(i, world_size, stride))
        # All ranks must call new_group collectively
        groups.append(dist.new_group(ranks))

    return groups


def _get_my_group(groups, rank):
    """Find which group the current rank belongs to."""
    for i, group in enumerate(groups):
        if group is not None:
            group_ranks = dist.get_process_group_ranks(group)
            if rank in group_ranks:
                return i, group
    return None, None


# =============================================================================
# All-Reduce with Process Groups
# =============================================================================



pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(