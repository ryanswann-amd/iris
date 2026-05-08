# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the persistent / resident-kernel all-reduce fast-path (K-810)."""

import gc

import pytest
import torch
import torch.distributed as dist

import iris
from iris.ccl import Config


@pytest.mark.parametrize("M, N", [(128, 64), (256, 256), (1024, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_iters", [1, 4, 16])
def test_all_reduce_persistent_burst(M, N, dtype, num_iters):
    """Burst persistent kernel produces the same result as torch.distributed."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()

    pytorch_input = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_input.fill_(float(rank + 1))

    pytorch_output = pytorch_input.clone()
    shmem.barrier()
    dist.all_reduce(pytorch_output, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    iris_input = shmem.zeros((M, N), dtype=dtype)
    iris_input.copy_(pytorch_input)
    iris_output = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()
    config = Config(all_reduce_variant="two_shot", block_size_m=32, block_size_n=64)
    workspace = shmem.ccl.all_reduce_persistent_burst(iris_output, iris_input, num_iters=num_iters, config=config)
    torch.cuda.synchronize()

    atol = 1e-3 if dtype == torch.float16 else 1e-5
    max_diff = torch.abs(iris_output - pytorch_output).max().item()
    try:
        assert torch.allclose(iris_output, pytorch_output, atol=atol), (
            f"Max diff {max_diff} exceeds tol {atol} on rank {rank} "
            f"(dtype={dtype}, M={M}, N={N}, num_iters={num_iters})"
        )
        # Workspace should be re-usable across calls.
        assert workspace is not None
        assert workspace.prepared
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


@pytest.mark.parametrize("M, N", [(128, 64), (1024, 256)])
@pytest.mark.parametrize("num_iters", [4, 16])
def test_all_reduce_persistent_burst_with_barrier(M, N, num_iters):
    """Burst persistent kernel with the per-iter cross-rank barrier enabled.

    This is the only configuration safe for general use (i.e. when peer
    inputs may change between iterations).  The barrier-disabled fast-path
    is exercised by ``test_all_reduce_persistent_burst`` above with
    constant inputs.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    dtype = torch.float32

    pytorch_input = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_input.fill_(float(rank + 1))

    pytorch_output = pytorch_input.clone()
    shmem.barrier()
    dist.all_reduce(pytorch_output, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    iris_input = shmem.zeros((M, N), dtype=dtype)
    iris_input.copy_(pytorch_input)
    iris_output = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()
    config = Config(all_reduce_variant="two_shot", block_size_m=32, block_size_n=64)
    workspace = shmem.ccl.all_reduce_persistent_burst(
        iris_output,
        iris_input,
        num_iters=num_iters,
        config=config,
        use_barrier=True,
    )
    torch.cuda.synchronize()

    try:
        atol = 1e-5
        max_diff = torch.abs(iris_output - pytorch_output).max().item()
        assert torch.allclose(iris_output, pytorch_output, atol=atol), (
            f"Max diff {max_diff} exceeds tol {atol} on rank {rank} "
            f"(M={M}, N={N}, num_iters={num_iters})"
        )
        assert workspace.prepared
    finally:
        shmem.barrier()
        del shmem
        gc.collect()
