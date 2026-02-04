# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import numpy as np
import pytest
import iris.experimental.iris_gluon as iris_gl



pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(