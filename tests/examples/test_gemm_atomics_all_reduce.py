#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import importlib.util
from pathlib import Path

import pytest
import torch
import iris
from examples.common.validation import validate_gemm

# Import the benchmark module
current_dir = Path(__file__).parent
benchmark_path = (current_dir / "../../examples/08_gemm_atomics_all_reduce/benchmark.py").resolve()
spec = importlib.util.spec_from_file_location("benchmark", benchmark_path)
benchmark_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_module)

# Test parameters
DTYPES = [torch.float16, torch.float32]
MATRIX_SIZES = [(256, 256, 256), (512, 512, 512)]
BLOCK_SIZES = [(64, 64, 32)]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m, n, k", MATRIX_SIZES)
@pytest.mark.parametrize("block_m, block_n, block_k", BLOCK_SIZES)
def test_gemm_atomics_all_reduce(dtype, m, n, k, block_m, block_n, block_k):
    # Initialize iris with appropriate heap size
    heap_size = 1 << 30  # 1GB
    shmem = iris.iris(heap_size)

    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Skip test if matrix dimensions are not divisible by world size
    if n % world_size != 0 or k % world_size != 0:
        pytest.skip(f"Matrix dimensions not divisible by world size {world_size}")

    # Create test matrices
    A = shmem.randn(m, k, device="cuda", dtype=dtype)
    B = shmem.randn(n, k, device="cuda", dtype=dtype)

    # Run the GEMM all-reduce operation using the benchmark function
    global_C, local_C = benchmark_module.run_gemm_all_reduce(
        A,
        B,
        shmem,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        gsize_m=8,
        two_tiles=True,
        num_stages=4,
        num_warps=4,
        waves_per_eu=2,
        mfma_instr_size=16,
        kpack=1,
        trace_tiles=False,
    )

    # Validate results
    success = validate_gemm(A, B, global_C, shmem, atol=1e-1)

    # Assert test passed
    assert success, "GEMM all-reduce validation failed"

    # Verify that we got a non-zero result
    assert not torch.allclose(global_C, torch.zeros_like(global_C)), "Result should not be all zeros"
