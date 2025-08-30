#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import iris

import importlib.util
import sys
from pathlib import Path

current_dir = Path(__file__).parent

# Add the repository root to Python path so relative imports work
repo_root = current_dir.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import the matmul wrapper
matmul_path = (current_dir / "../../examples/07_gemm_all_scatter/matmul_wrapper.py").resolve()
matmul_spec = importlib.util.spec_from_file_location("matmul_wrapper", matmul_path)
matmul_module = importlib.util.module_from_spec(matmul_spec)
matmul_spec.loader.exec_module(matmul_module)

# Import the validation function
validation_path = (current_dir / "../../examples/common/validation.py").resolve()
validation_spec = importlib.util.spec_from_file_location("validation", validation_path)
validation_module = importlib.util.module_from_spec(validation_spec)
validation_spec.loader.exec_module(validation_module)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "m, n, k",
    [
        (64, 64, 64),  # Very small for quick testing
        (128, 128, 128),  # Small
        (256, 256, 256),  # Medium
    ],
)
@pytest.mark.parametrize(
    "BLK_M, BLK_N, BLK_K",
    [
        (32, 32, 16),  # Small blocks
        (64, 64, 32),  # Medium blocks
    ],
)
def test_gemm_all_scatter(dtype, m, n, k, BLK_M, BLK_N, BLK_K):
    # Set up iris shared memory
    heap_size = 1 << 30  # 1GB heap size for tests
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Ensure matrix dimensions are compatible with world size
    if n % world_size != 0 or k % world_size != 0:
        pytest.skip(f"Matrix dimensions (n={n}, k={k}) not divisible by world_size={world_size}")

    # Create test matrices
    A = shmem.randn(m, k, device="cuda", dtype=dtype)
    B = shmem.randn(n, k, device="cuda", dtype=dtype).T

    # Local splitting for each rank
    local_n = n // world_size
    local_B = B[:, rank * local_n : (rank + 1) * local_n].clone()
    local_A = A

    # Allocate result matrices
    global_C = shmem.zeros((m, n), device="cuda", dtype=dtype)
    local_C = shmem.zeros((m, local_n), device="cuda", dtype=dtype)

    # Set up parameters similar to benchmark
    bias = None
    gemm_sms = min(shmem.get_cu_count(), 64)  # Use fewer SMs for testing
    gsize_m = 4  # Smaller group size for tests

    shmem.barrier()

    # Run the GEMM all-scatter kernel
    result_C = matmul_module.matmul.apply(
        local_A,
        local_B,
        local_C,
        global_C,
        bias,
        rank,
        world_size,
        gemm_sms,
        BLK_M,
        BLK_N,
        BLK_K,
        gsize_m,
        shmem.get_heap_bases(),
        "gfx942",
        False,  # COLLECT_TIMESTAMPS
        None,  # mm_begin_timestamp
        None,  # mm_end_timestamp
    )

    shmem.barrier()

    # Validate the result using the existing validation function
    success = validation_module.validate_gemm(A, B, global_C, shmem, atol=1e-2)

    # Additional assertion with detailed error message
    assert success, (
        f"GEMM all-scatter validation failed for dtype={dtype}, m={m}, n={n}, k={k}, BLK_M={BLK_M}, BLK_N={BLK_N}, BLK_K={BLK_K}"
    )


@pytest.mark.parametrize("dtype", [torch.float32])
def test_gemm_all_scatter_minimal(dtype):
    """Test with minimal dimensions to ensure basic functionality works."""
    # Set up iris shared memory
    heap_size = 1 << 28  # 256MB heap size for minimal test
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Use very small dimensions that should work with any world size
    m, n, k = 32, 32 * world_size, 32 * world_size
    BLK_M, BLK_N, BLK_K = 16, 16, 16

    # Create test matrices
    A = shmem.randn(m, k, device="cuda", dtype=dtype)
    B = shmem.randn(n, k, device="cuda", dtype=dtype).T

    # Local splitting for each rank
    local_n = n // world_size
    local_B = B[:, rank * local_n : (rank + 1) * local_n].clone()
    local_A = A

    # Allocate result matrices
    global_C = shmem.zeros((m, n), device="cuda", dtype=dtype)
    local_C = shmem.zeros((m, local_n), device="cuda", dtype=dtype)

    # Set up parameters for minimal test
    bias = None
    gemm_sms = min(shmem.get_cu_count(), 32)  # Use even fewer SMs for minimal test
    gsize_m = 2  # Small group size

    shmem.barrier()

    # Run the GEMM all-scatter kernel
    result_C = matmul_module.matmul.apply(
        local_A,
        local_B,
        local_C,
        global_C,
        bias,
        rank,
        world_size,
        gemm_sms,
        BLK_M,
        BLK_N,
        BLK_K,
        gsize_m,
        shmem.get_heap_bases(),
        "gfx942",
        False,  # COLLECT_TIMESTAMPS
        None,  # mm_begin_timestamp
        None,  # mm_end_timestamp
    )

    shmem.barrier()

    # Validate the result
    success = validation_module.validate_gemm(A, B, global_C, shmem, atol=1e-2)
    assert success, f"Minimal GEMM all-scatter validation failed for dtype={dtype}"
