# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for reduce-scatter collective operation.
"""

import gc

import pytest
import torch
import torch.distributed as dist
import iris
from iris.ccl import Config


@pytest.mark.parametrize(
    "distribution",
    [
        0,  # striding
        1,  # block
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "M, N",
    [
        (128, 64),  # Small
        (1024, 256),  # Medium
        (8192, 8192),  # Large
    ],
)
def test_reduce_scatter(distribution, dtype, M, N):
    """Test reduce-scatter functionality.

    Each rank reduces its assigned tiles from all ranks' inputs. The tile partition
    is complete: summing the outputs across all ranks (all_reduce) should yield the
    element-wise sum of all inputs at every position.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Each rank fills its input with (rank + 1)
    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.fill_(float(rank + 1))

    iris_output_tensor = shmem.zeros((M, N), dtype=dtype)

    # Run Iris reduce_scatter
    shmem.barrier()
    config = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=distribution)
    shmem.ccl.reduce_scatter(iris_output_tensor, iris_input_tensor, config=config)
    torch.cuda.synchronize()

    # Validate: tiles are partitioned across ranks, so summing outputs from all ranks
    # (via all_reduce) should give the element-wise sum of all inputs at every position.
    expected = float(world_size * (world_size + 1) // 2)
    aggregated = iris_output_tensor.clone()
    dist.all_reduce(aggregated, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    atol = 1e-3 if dtype == torch.float16 else 1e-5
    max_diff = torch.abs(aggregated - expected).max().item()

    try:
        assert torch.allclose(aggregated, torch.full_like(aggregated, expected), atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: aggregated reduce-scatter outputs don't match expected sum "
            f"(distribution={distribution})"
        )
    finally:
        shmem.barrier()
        del shmem
        gc.collect()
