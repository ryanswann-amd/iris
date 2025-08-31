#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for the 09_gemm_one_shot_all_reduce example.

This test suite provides comprehensive testing for the GEMM one-shot all-reduce
algorithm implementation. Tests expect ROCm/HIP to be available in CI environment.
"""

import pytest
import torch
import triton
import numpy as np
import sys
import os

import importlib.util
from pathlib import Path

# Add the project root to Python path to help with imports
current_dir = Path(__file__).parent
project_root = (current_dir / "../..").resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the example module
example_dir = (project_root / "examples/09_gemm_one_shot_all_reduce").resolve()
if str(example_dir) not in sys.path:
    sys.path.insert(0, str(example_dir))

# Import necessary modules
import iris
from examples.common.validation import validate_gemm

# Import the benchmark module
benchmark_file = example_dir / "benchmark.py"
spec = importlib.util.spec_from_file_location("benchmark", benchmark_file)
benchmark_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_module)


def test_gemm_one_shot_all_reduce_import():
    """Test that the benchmark module can be imported and has required functions."""
    assert hasattr(benchmark_module, "main"), "Benchmark module should have a main function"
    assert hasattr(benchmark_module, "parse_args"), "Benchmark module should have a parse_args function"
    assert hasattr(benchmark_module, "gemm_one_shot_all_reduce"), (
        "Benchmark module should have a gemm_one_shot_all_reduce function"
    )


def test_parse_args():
    """Test argument parsing functionality."""
    # Temporarily replace sys.argv to test argument parsing
    original_argv = sys.argv
    try:
        # Test with minimal arguments
        sys.argv = ["benchmark.py", "-m", "128", "-n", "128", "-k", "128", "--validate"]

        args = benchmark_module.parse_args()

        # Check that arguments are parsed correctly
        assert args["m"] == 128, f"Expected m=128, got {args['m']}"
        assert args["n"] == 128, f"Expected n=128, got {args['n']}"
        assert args["k"] == 128, f"Expected k=128, got {args['k']}"
        assert args["validate"], f"Expected validate=True, got {args['validate']}"

        # Check that defaults are set
        assert "datatype" in args, "Args should contain datatype"
        assert "BLK_M" in args, "Args should contain BLK_M"
        assert "BLK_N" in args, "Args should contain BLK_N"
        assert "BLK_K" in args, "Args should contain BLK_K"

    finally:
        sys.argv = original_argv


@pytest.mark.parametrize(
    "M, N, K, world_size",
    [
        (256, 256, 256, 2),  # Basic case with 2 ranks
        (512, 512, 512, 4),  # Larger case with 4 ranks
    ],
)
def test_matrix_dimension_divisibility(M, N, K, world_size):
    """Test that matrix dimensions are properly divisible by world size as required by the algorithm."""
    # Test the assertions that are made in the benchmark code
    assert N % world_size == 0, f"N ({N}) must be divisible by world size ({world_size})"
    assert K % world_size == 0, f"K ({K}) must be divisible by world size ({world_size})"

    # Test matrix splitting logic
    rows_per_gpu = K // world_size
    assert rows_per_gpu > 0, "Each GPU should get at least one row"
    assert rows_per_gpu * world_size == K, "Total rows should equal K"


@pytest.mark.parametrize(
    "datatype, M, N, K",
    [
        (torch.float16, 128, 128, 128),
        (torch.float32, 128, 128, 128),
        (torch.bfloat16, 128, 128, 128),
    ],
)
def test_gemm_one_shot_all_reduce_function(datatype, M, N, K):
    """Test the core gemm_one_shot_all_reduce function with different data types."""
    # Set up iris environment
    heap_size = 1 << 30  # 1GB heap
    shmem = iris.iris(heap_size)
    world_size = shmem.get_num_ranks()

    # Skip if dimensions are not divisible by world size
    if N % world_size != 0 or K % world_size != 0:
        pytest.skip(f"Matrix dimensions ({M}x{N}x{K}) not divisible by world_size ({world_size})")

    # Create input matrices
    A = shmem.randn(M, K, device="cuda", dtype=datatype)
    B = shmem.randn(N, K, device="cuda", dtype=datatype).T

    # Set up algorithm parameters
    args_dict = {
        "m": M,
        "n": N,
        "k": K,
        "BLK_M": 64,  # Smaller blocks for testing
        "BLK_N": 64,
        "BLK_K": 32,
        "gsize_m": 1,
        "two_tiles": True,
        "num_stages": 1,
        "num_warps": 4,
        "waves_per_eu": 0,
        "mfmaInstrSize": 16,
        "kpack": 1,
        "gemm_sms": min(64, 288),  # Reduced for testing
        "total_sms": 304,
        "trace_tiles": False,
    }

    # Run the GEMM one-shot all-reduce
    result_C = benchmark_module.gemm_one_shot_all_reduce(A, B, shmem, args_dict)

    # Basic shape and type checks
    assert result_C.shape == (M, N), f"Expected output shape ({M}, {N}), got {result_C.shape}"
    assert result_C.dtype == datatype, f"Expected output dtype {datatype}, got {result_C.dtype}"

    # Validate the result using the existing validation function
    success = validate_gemm(A, B, result_C, shmem, atol=2)
    assert success, "GEMM validation failed"


def test_block_size_calculations():
    """Test block size calculations used in the GEMM kernel."""
    # Test triton.cdiv functionality which is used in the benchmark
    M, N, K = 1000, 2000, 3000
    BLK_M, BLK_N, BLK_K = 256, 256, 32

    # Test ceiling division
    import math

    expected_blocks_M = math.ceil(M / BLK_M)
    expected_blocks_N = math.ceil(N / BLK_N)
    expected_blocks_K = math.ceil(K / BLK_K)

    actual_blocks_M = triton.cdiv(M, BLK_M)
    actual_blocks_N = triton.cdiv(N, BLK_N)
    actual_blocks_K = triton.cdiv(K, BLK_K)

    assert actual_blocks_M == expected_blocks_M, (
        f"Block M calculation mismatch: {actual_blocks_M} != {expected_blocks_M}"
    )
    assert actual_blocks_N == expected_blocks_N, (
        f"Block N calculation mismatch: {actual_blocks_N} != {expected_blocks_N}"
    )
    assert actual_blocks_K == expected_blocks_K, (
        f"Block K calculation mismatch: {actual_blocks_K} != {expected_blocks_K}"
    )


def test_file_structure():
    """Test that all required files exist in the example directory."""
    example_dir = Path(__file__).parent / "../../examples/09_gemm_one_shot_all_reduce"
    example_dir = example_dir.resolve()

    required_files = [
        "benchmark.py",
        "gemm_one_shot_all_reduce.py",
        "matmul_wrapper.py",
    ]

    for filename in required_files:
        file_path = example_dir / filename
        assert file_path.exists(), f"Required file {filename} not found in {example_dir}"
        assert file_path.is_file(), f"{filename} exists but is not a file"

        # Check that file is not empty
        assert file_path.stat().st_size > 0, f"{filename} is empty"


def test_algorithm_parameters_validation():
    """Test validation of algorithm parameters."""
    # Test that invalid gemm_sms configuration is caught
    args_dict = {
        "gemm_sms": 350,  # Greater than total_sms
        "total_sms": 304,
    }

    with pytest.raises(ValueError, match="Invalid number of stream-K SMs"):
        # This should raise an error
        heap_size = 1 << 30
        shmem = iris.iris(heap_size)
        A = shmem.randn(64, 64, device="cuda", dtype=torch.float16)
        B = shmem.randn(64, 64, device="cuda", dtype=torch.float16).T

        # Add required parameters
        args_dict.update(
            {
                "m": 64,
                "n": 64,
                "k": 64,
                "BLK_M": 32,
                "BLK_N": 32,
                "BLK_K": 16,
                "gsize_m": 1,
                "two_tiles": True,
                "num_stages": 1,
                "num_warps": 4,
                "waves_per_eu": 0,
                "mfmaInstrSize": 16,
                "kpack": 1,
            }
        )

        benchmark_module.gemm_one_shot_all_reduce(A, B, shmem, args_dict)
