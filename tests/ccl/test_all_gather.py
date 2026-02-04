# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for all-gather collective operation.
"""

import pytest
import torch
import torch.distributed as dist
import iris
from iris.ccl import Config



pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(