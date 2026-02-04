# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
import iris



pytestmark = pytest.mark.single_rank

@pytest.mark.parametrize(