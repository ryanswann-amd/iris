#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import triton
import triton.language as tl
import numpy as np
import iris

import importlib.util
from pathlib import Path

current_dir = Path(__file__).parent
file_path = (current_dir / "../../examples/03_all_store/all_store_bench.py").resolve()
module_name = "all_store_bench"
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "buffer_size, heap_size",
    [
        ((1 << 32), (1 << 33)),
    ],
)
@pytest.mark.parametrize(
    "block_size",
    [
        512,
        1024,
    ],
)
def test_all_store_bench(dtype, buffer_size, heap_size, block_size):
    shmem = iris.iris(heap_size)
    num_ranks = shmem.get_num_ranks()

    element_size_bytes = torch.tensor([], dtype=dtype).element_size()
    n_elements = buffer_size // element_size_bytes
    buffer = shmem.zeros(n_elements, device="cuda", dtype=dtype)

    shmem.barrier()

    # Create arguments dict similar to what parse_args() would return
    # Using minimal required parameters for testing
    args = {
        "datatype": _torch_dtype_to_str(dtype),
        "block_size": block_size,
        "verbose": False,
        "validate": False,
        "num_experiments": 1,  # Minimal for testing
        "num_warmup": 0,      # Skip warmup for testing
        "active_ranks": min(num_ranks, 8),  # Use available ranks or 8, whichever is smaller
    }

    # Call the run_experiment function from the module
    bandwidth_gbps = module.run_experiment(shmem, args, buffer)

    # Basic validation that we got a reasonable bandwidth value
    assert bandwidth_gbps >= 0.0, f"Bandwidth should be non-negative, got {bandwidth_gbps}"
    assert bandwidth_gbps < 10000.0, f"Bandwidth seems unreasonably high: {bandwidth_gbps} GiB/s"


def _torch_dtype_to_str(dtype):
    """Helper function to convert torch dtype to string format expected by the module"""
    dtype_map = {
        torch.float16: "fp16",
        torch.float32: "fp32",
        torch.int8: "int8",
        torch.bfloat16: "bf16",
    }
    return dtype_map[dtype]
