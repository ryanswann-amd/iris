# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for high-level matmul_reduce_scatter API.
"""

import pytest
import torch
import torch.distributed as dist
import iris
import iris.ops as ops



pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(