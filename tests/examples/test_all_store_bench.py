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
    cur_rank = shmem.get_rank()

    element_size_bytes = torch.tensor([], dtype=dtype).element_size()
    n_elements = buffer_size // element_size_bytes
    buffer = shmem.zeros(n_elements, device="cuda", dtype=dtype)

    # Simple test similar to load_bench - just test the kernel functionality
    # without the complex benchmarking infrastructure
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Test all_store_kernel directly, similar to how load_bench tests the load_kernel
    if cur_rank < min(num_ranks, 8):  # Only test with a reasonable number of ranks
        module.all_store_kernel[grid](
            buffer,
            cur_rank,
            n_elements,
            num_ranks,
            block_size,
            shmem.get_heap_bases(),
        )


def _torch_dtype_to_str(dtype):
    """Helper function to convert torch dtype to string format expected by the module"""
    dtype_map = {
        torch.float16: "fp16",
        torch.float32: "fp32",
        torch.int8: "int8",
        torch.bfloat16: "bf16",
    }
    return dtype_map[dtype]
