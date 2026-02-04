# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for fused all_gather + matmul operation.

Each rank has A_sharded (M x K_local), B is replicated.
The operation gathers A from all ranks and computes C = A_gathered @ B.
"""

import pytest
import torch
import torch.distributed as dist

import iris



pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(