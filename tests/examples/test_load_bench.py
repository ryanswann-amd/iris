#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import numpy as np
import iris

import importlib.util
from pathlib import Path


pytestmark = pytest.mark.multi_rank_required

current_dir = Path(__file__).parent
file_path = (current_dir / "../../examples/00_load/load_bench.py").resolve()
module_name = "load_bench"
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.skip(reason="Test is inconsistent and needs debugging - tracked in issue")

pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(