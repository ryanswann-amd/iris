# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import iris



pytestmark = pytest.mark.single_rank

def test_arange_basic_functionality():