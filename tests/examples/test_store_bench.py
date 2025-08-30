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
file_path = (current_dir / "../../examples/01_store/store_bench.py").resolve()
module_name = "store_bench"
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
def test_store_bench(dtype, buffer_size, heap_size, block_size):
    shmem = iris.iris(heap_size)
    num_ranks = shmem.get_num_ranks()

    bandwidth_matrix = np.zeros((num_ranks, num_ranks), dtype=np.float32)
    element_size_bytes = torch.tensor([], dtype=dtype).element_size()
    buffer = shmem.zeros(buffer_size // element_size_bytes, dtype=dtype)

    shmem.barrier()

    for source_rank in range(num_ranks):
        for destination_rank in range(num_ranks):
            bandwidth_gbps = module.bench_store(shmem, source_rank, destination_rank, buffer, block_size, dtype)
            bandwidth_matrix[source_rank, destination_rank] = bandwidth_gbps
            shmem.barrier()
