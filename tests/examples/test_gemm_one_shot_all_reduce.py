#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import triton
import triton.language as tl
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

# Add the specific example directory to help with relative imports
example_dir = (project_root / "examples/09_gemm_one_shot_all_reduce").resolve()
if str(example_dir) not in sys.path:
    sys.path.insert(0, str(example_dir))


def test_gemm_one_shot_all_reduce_import():
    """Test that the gemm_one_shot_all_reduce module can be imported correctly."""
    current_dir = Path(__file__).parent
    file_path = (current_dir / "../../examples/09_gemm_one_shot_all_reduce/benchmark.py").resolve()
    module_name = "gemm_one_shot_all_reduce_benchmark"

    assert file_path.exists(), f"Benchmark file not found at {file_path}"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)

    # Try to import - this may fail due to missing AMD GPU libraries, which is expected
    try:
        spec.loader.exec_module(module)
        # Check that required functions exist
        assert hasattr(module, "main"), "Benchmark module should have a main function"
        assert hasattr(module, "parse_args"), "Benchmark module should have a parse_args function"
    except (OSError, ImportError) as e:
        if "libamdhip64.so" in str(e) or "HIP" in str(e) or "AMD" in str(e):
            pytest.skip(f"Skipping test due to missing AMD GPU libraries: {e}")
        else:
            raise


def test_matmul_wrapper_import():
    """Test that the matmul_wrapper module can be imported correctly."""
    current_dir = Path(__file__).parent
    file_path = (current_dir / "../../examples/09_gemm_one_shot_all_reduce/matmul_wrapper.py").resolve()
    module_name = "matmul_wrapper"

    assert file_path.exists(), f"Matmul wrapper file not found at {file_path}"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)

    # Try to import - this may fail due to missing dependencies, which is expected
    try:
        spec.loader.exec_module(module)
        # Check that required classes exist
        assert hasattr(module, "matmul"), "Matmul wrapper should have a matmul class"
    except (OSError, ImportError, ModuleNotFoundError) as e:
        if any(keyword in str(e) for keyword in ["libamdhip64.so", "HIP", "AMD", "gemm_one_shot_all_reduce"]):
            pytest.skip(f"Skipping test due to missing dependencies: {e}")
        else:
            raise


def test_gemm_kernel_import():
    """Test that the gemm_one_shot_all_reduce kernel can be imported correctly."""
    current_dir = Path(__file__).parent
    file_path = (current_dir / "../../examples/09_gemm_one_shot_all_reduce/gemm_one_shot_all_reduce.py").resolve()
    module_name = "gemm_one_shot_all_reduce"

    assert file_path.exists(), f"GEMM kernel file not found at {file_path}"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)

    # Try to import - this may fail due to missing AMD GPU libraries, which is expected
    try:
        spec.loader.exec_module(module)
        # Check that required kernel exists
        assert hasattr(module, "persistent_gemm_all_reduce"), "Module should have persistent_gemm_all_reduce kernel"
    except (OSError, ImportError) as e:
        if "libamdhip64.so" in str(e) or "HIP" in str(e) or "AMD" in str(e):
            pytest.skip(f"Skipping test due to missing AMD GPU libraries: {e}")
        else:
            raise


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


def test_block_size_calculations():
    """Test block size calculations used in the GEMM kernel."""
    # Test triton.cdiv functionality which is used in the benchmark
    M, N, K = 1000, 2000, 3000
    BLK_M, BLK_N, BLK_K = 256, 256, 32

    # Test ceiling division
    import math

    total_blocks_M = math.ceil(M / BLK_M)
    total_blocks_N = math.ceil(N / BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    iters_per_tile = math.ceil(K / BLK_K)

    assert total_blocks_M > 0, "Should have at least one block in M dimension"
    assert total_blocks_N > 0, "Should have at least one block in N dimension"
    assert total_tiles > 0, "Should have at least one tile"
    assert iters_per_tile > 0, "Should have at least one iteration per tile"

    # Test specific examples
    assert math.ceil(1000 / 256) == 4, "1000/256 should ceil to 4"
    assert math.ceil(2000 / 256) == 8, "2000/256 should ceil to 8"
    assert math.ceil(3000 / 32) == 94, "3000/32 should ceil to 94"


@pytest.mark.parametrize(
    "dtype, device",
    [
        (torch.float16, "cpu"),
        (torch.float32, "cpu"),
        (torch.bfloat16, "cpu"),
    ],
)
def test_tensor_operations_cpu(dtype, device):
    """Test basic tensor operations that mirror what the GEMM kernel does, but on CPU."""

    # Small matrices for testing
    M, N, K = 64, 64, 64

    # Create test matrices similar to benchmark.py
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(N, K, dtype=dtype, device=device).T  # Note the transpose
    C = torch.zeros(M, N, dtype=dtype, device=device)

    # Test matrix multiplication
    result = A @ B

    # Check shapes
    assert A.shape == (M, K), f"A should be {M}x{K}, got {A.shape}"
    assert B.shape == (K, N), f"B should be {K}x{N}, got {B.shape}"
    assert result.shape == (M, N), f"Result should be {M}x{N}, got {result.shape}"

    # Test that result is reasonable (not all zeros, not all same value)
    assert not torch.allclose(result, torch.zeros_like(result)), "Result should not be all zeros"

    # Test validation using the validation function
    current_dir = Path(__file__).parent
    file_path = (current_dir / "../../examples/common/validation.py").resolve()
    spec = importlib.util.spec_from_file_location("validation", file_path)
    validation_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation_module)

    # Mock shmem for validation
    class MockShmem:
        def info(self, msg):
            pass

        def error(self, msg):
            pass

    shmem = MockShmem()

    # Test validation passes for correct result
    is_valid = validation_module.validate_gemm(A, B, result, shmem, atol=1e-3)
    assert is_valid, "Validation should pass for correct GEMM computation"


def test_file_structure():
    """Test that all required files exist and have the expected structure."""
    current_dir = Path(__file__).parent
    example_dir = (current_dir / "../../examples/09_gemm_one_shot_all_reduce").resolve()

    required_files = ["benchmark.py", "gemm_one_shot_all_reduce.py", "matmul_wrapper.py"]

    for filename in required_files:
        file_path = example_dir / filename
        assert file_path.exists(), f"Required file {filename} should exist at {file_path}"
        assert file_path.is_file(), f"{filename} should be a regular file"
        assert file_path.stat().st_size > 0, f"{filename} should not be empty"

    # Test that the files contain expected content
    benchmark_content = (example_dir / "benchmark.py").read_text()
    assert "def main():" in benchmark_content, "benchmark.py should have a main function"
    assert "def parse_args():" in benchmark_content, "benchmark.py should have parse_args function"
    assert "matmul.apply" in benchmark_content, "benchmark.py should call matmul.apply"

    kernel_content = (example_dir / "gemm_one_shot_all_reduce.py").read_text()
    assert "@triton.jit" in kernel_content, "Kernel should contain Triton JIT decorators"
    assert "persistent_gemm_all_reduce" in kernel_content, "Kernel should contain main function"

    wrapper_content = (example_dir / "matmul_wrapper.py").read_text()
    assert "class matmul" in wrapper_content, "Wrapper should contain matmul class"
    assert "torch.autograd.Function" in wrapper_content, "Should inherit from autograd Function"


def test_validation_function():
    """Test the validation function from common.validation."""
    current_dir = Path(__file__).parent
    file_path = (current_dir / "../../examples/common/validation.py").resolve()
    module_name = "validation"

    assert file_path.exists(), f"Validation file not found at {file_path}"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Check that validate_gemm function exists
    assert hasattr(module, "validate_gemm"), "Validation module should have validate_gemm function"

    # Test validation function with mock shmem object
    class MockShmem:
        def info(self, msg):
            pass

        def error(self, msg):
            pass

    # Create test matrices
    A = torch.randn(32, 32, dtype=torch.float32)
    B = torch.randn(32, 32, dtype=torch.float32)
    C = A @ B  # Correct result

    shmem = MockShmem()
    result = module.validate_gemm(A, B, C, shmem, atol=1e-3)
    assert result, "Validation should pass for correct computation"

    # Test with incorrect result
    C_wrong = torch.zeros_like(C)
    result = module.validate_gemm(A, B, C_wrong, shmem, atol=1e-3)
    assert not result, "Validation should fail for incorrect computation"


@pytest.mark.parametrize(
    "datatype_str",
    [
        "fp16",
        "fp32",
        "bf16",
    ],
)
def test_datatype_parsing(datatype_str):
    """Test that datatype string parsing works correctly."""

    # Test datatype mapping
    datatype_map = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "int8": torch.int8,
        "bf16": torch.bfloat16,
    }

    if datatype_str in datatype_map:
        dtype = datatype_map[datatype_str]

        # Test that we can create tensors with this dtype
        test_tensor = torch.zeros(10, dtype=dtype)
        assert test_tensor.dtype == dtype, f"Tensor should have dtype {dtype}, got {test_tensor.dtype}"


def test_parse_args_function():
    """Test the argument parsing function from the benchmark module."""
    current_dir = Path(__file__).parent
    file_path = (current_dir / "../../examples/09_gemm_one_shot_all_reduce/benchmark.py").resolve()
    module_name = "gemm_one_shot_all_reduce_benchmark"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)

    # Temporarily replace sys.argv to test argument parsing
    original_argv = sys.argv
    try:
        # Test with minimal arguments
        sys.argv = ["benchmark.py", "-m", "128", "-n", "128", "-k", "128", "--validate"]

        # Try to import - this may fail due to missing AMD GPU libraries, which is expected
        try:
            spec.loader.exec_module(module)
            args = module.parse_args()

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

        except (OSError, ImportError) as e:
            if "libamdhip64.so" in str(e) or "HIP" in str(e) or "AMD" in str(e):
                pytest.skip(f"Skipping test due to missing AMD GPU libraries: {e}")
            else:
                raise

    finally:
        sys.argv = original_argv
