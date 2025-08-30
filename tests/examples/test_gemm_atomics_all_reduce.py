#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import importlib.util
from pathlib import Path

import pytest

# Try to import dependencies - skip test if not available
try:
    import numpy as np
    import torch
    import triton
    import triton.language as tl
    import iris
    from examples.common.utils import Timestamps
    from examples.common.validation import validate_gemm

    # Define test parameters after successful import
    DTYPES = [torch.float16, torch.float32]
    MATRIX_SIZES = [(256, 256, 256), (512, 512, 512)]
    BLOCK_SIZES = [(64, 64, 32)]

except ImportError as e:
    pytest.skip(f"Skipping gemm_atomics_all_reduce test due to missing dependencies: {e}", allow_module_level=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m, n, k", MATRIX_SIZES)
@pytest.mark.parametrize("block_m, block_n, block_k", BLOCK_SIZES)
def test_gemm_atomics_all_reduce(dtype, m, n, k, block_m, block_n, block_k):
    # Import matmul_wrapper module at test time
    try:
        current_dir = Path(__file__).parent
        matmul_wrapper_path = (current_dir / "../../examples/08_gemm_atomics_all_reduce/matmul_wrapper.py").resolve()

        matmul_spec = importlib.util.spec_from_file_location("matmul_wrapper", matmul_wrapper_path)
        matmul_module = importlib.util.module_from_spec(matmul_spec)
        matmul_spec.loader.exec_module(matmul_module)
    except (ImportError, FileNotFoundError) as e:
        pytest.skip(f"Skipping test due to import error: {e}")

    # Initialize iris with appropriate heap size
    heap_size = 1 << 30  # 1GB
    shmem = iris.iris(heap_size)

    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    cu_count = shmem.get_cu_count()

    # Skip test if matrix dimensions are not divisible by world size
    if n % world_size != 0 or k % world_size != 0:
        pytest.skip(f"Matrix dimensions not divisible by world size {world_size}")

    # Create test matrices
    A = shmem.randn(m, k, device="cuda", dtype=dtype)
    B = shmem.randn(n, k, device="cuda", dtype=dtype).T
    C = shmem.zeros((m, n), device="cuda", dtype=dtype)

    # Split matrices according to rank
    rows_per_gpu = k // world_size
    start_row = rank * rows_per_gpu
    end_row = start_row + rows_per_gpu
    local_B = B[start_row:end_row, :]
    local_A = A[:, start_row:end_row]

    # Create output matrices
    global_C = shmem.zeros((m, n), device="cuda", dtype=dtype)
    local_C = shmem.zeros((m, n), device="cuda", dtype=dtype)

    # Setup parameters
    total_blocks_M = triton.cdiv(m, block_m)
    total_blocks_N = triton.cdiv(n, block_n)
    total_tiles = total_blocks_M * total_blocks_N

    # Use conservative number of SMs
    gemm_sms = min(cu_count // 2, 64)  # Use half of available CUs, max 64

    # Create required tensors
    tile_completed = shmem.zeros((total_tiles,), device="cuda", dtype=torch.int32)
    locks = shmem.zeros((gemm_sms,), device="cuda", dtype=torch.int32)
    P = shmem.zeros(
        (gemm_sms, block_m * block_n),
        device="cuda",
        dtype=torch.float32,
    )
    bias = None

    # Setup timestamps
    timestamps = Timestamps(num_tiles=total_tiles)

    # Synchronize before test
    shmem.barrier()

    # Reset tile_completed
    iris.memset_tensor(tile_completed, 0)
    shmem.barrier()

    # Run the GEMM all-reduce operation
    matmul_module.matmul.set_debug(False)

    result_C = matmul_module.matmul.apply(
        local_A,
        local_B,
        local_C,
        global_C,
        bias,
        P,
        locks,
        tile_completed,
        rank,
        world_size,
        gemm_sms,
        block_m,
        block_n,
        block_k,
        8,  # gsize_m
        True,  # two_tiles
        4,  # num_stages
        4,  # num_warps
        2,  # waves_per_eu
        16,  # mfmaInstrSize
        1,  # kpack
        shmem.get_heap_bases(),
        cu_count,
        False,  # trace_tiles
        timestamps.mm_begin_timestamp,
        timestamps.mm_end_timestamp,
    )

    # Synchronize after computation
    shmem.barrier()

    # Validate results
    success = validate_gemm(A, B, global_C, shmem, atol=1e-1)

    # Assert test passed
    assert success, "GEMM all-reduce validation failed"

    # Verify that we got a non-zero result
    assert not torch.allclose(global_C, torch.zeros_like(global_C)), "Result should not be all zeros"
