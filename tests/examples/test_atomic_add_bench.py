#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import iris

import importlib.util
import sys
from pathlib import Path


pytestmark = pytest.mark.multi_rank_required

current_dir = Path(__file__).parent

# Add examples directory to sys.path so that example files can import from examples.common
# Note: Examples use "from examples.common.utils import ..." which requires examples/ in sys.path
examples_dir = (current_dir / "../..").resolve()
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

# Load utils module from file path (not package import)
# Note: We use path-based imports instead of "from examples.common.utils import ..."
# because examples/ is not included in the installed package. This allows tests to
# work with both editable install (pip install -e .) and regular install (pip install git+...).
utils_path = (current_dir / "../../examples/common/utils.py").resolve()
utils_spec = importlib.util.spec_from_file_location("utils", utils_path)
utils_module = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(utils_module)
torch_dtype_to_str = utils_module.torch_dtype_to_str

# Load benchmark module
file_path = (current_dir / "../../examples/04_atomic_add/atomic_add_bench.py").resolve()
module_name = "atomic_add_bench"
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)



pytestmark = pytest.mark.multi_rank_required

@pytest.mark.parametrize(