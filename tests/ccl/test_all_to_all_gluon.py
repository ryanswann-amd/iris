# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for all-to-all collective operation using Gluon with traffic shaping.
"""

import pytest
import torch
import torch.distributed as dist

# Try to import Gluon, skip tests if not available
try:
    import iris.experimental.iris_gluon as iris_gluon
    from iris.ccl import Config
    from iris.ccl.all_to_all import all_to_all

    GLUON_AVAILABLE = True
except ImportError:
    GLUON_AVAILABLE = False


@pytest.mark.skipif(not GLUON_AVAILABLE, reason="Gluon not available")

pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(