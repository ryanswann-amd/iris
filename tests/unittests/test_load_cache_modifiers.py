# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris


@triton.jit
def load_kernel_default(
    data,
    results,
    source_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
    cache_modifier: tl.constexpr,
    volatile: tl.constexpr,
):
    pid = tl.program_id(0)

    partner = int((source_rank + num_ranks // 2) % num_ranks)
    # Compute start index of this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Guard for out-of-bounds accesses
    mask = offsets < BLOCK_SIZE
    result = iris.load(
        data + offsets, source_rank, partner, heap_bases, mask=mask, cache_modifier=cache_modifier, volatile=volatile
    )
    tl.store(results + offsets, result, mask=mask)


@triton.jit
def load_kernel_writeback(
    data,
    results,
    source_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
    volatile: tl.constexpr,
):
    pid = tl.program_id(0)

    partner = int((source_rank + num_ranks // 2) % num_ranks)
    # Compute start index of this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Guard for out-of-bounds accesses
    mask = offsets < BLOCK_SIZE
    result = iris.load(
        data + offsets, source_rank, partner, heap_bases, mask=mask, cache_modifier=".wb", volatile=volatile
    )
    tl.store(results + offsets, result, mask=mask)


@triton.jit
def load_kernel_cache_global(
    data,
    results,
    source_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
    volatile: tl.constexpr,
):
    pid = tl.program_id(0)

    partner = int((source_rank + num_ranks // 2) % num_ranks)
    # Compute start index of this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Guard for out-of-bounds accesses
    mask = offsets < BLOCK_SIZE
    result = iris.load(
        data + offsets, source_rank, partner, heap_bases, mask=mask, cache_modifier=".cg", volatile=volatile
    )
    tl.store(results + offsets, result, mask=mask)


@triton.jit
def load_kernel_cache_streaming(
    data,
    results,
    source_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
    volatile: tl.constexpr,
):
    pid = tl.program_id(0)

    partner = int((source_rank + num_ranks // 2) % num_ranks)
    # Compute start index of this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Guard for out-of-bounds accesses
    mask = offsets < BLOCK_SIZE
    result = iris.load(
        data + offsets, source_rank, partner, heap_bases, mask=mask, cache_modifier=".cs", volatile=volatile
    )
    tl.store(results + offsets, result, mask=mask)


@triton.jit
def load_kernel_write_through(
    data,
    results,
    source_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
    volatile: tl.constexpr,
):
    pid = tl.program_id(0)

    partner = int((source_rank + num_ranks // 2) % num_ranks)
    # Compute start index of this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Guard for out-of-bounds accesses
    mask = offsets < BLOCK_SIZE
    result = iris.load(
        data + offsets, source_rank, partner, heap_bases, mask=mask, cache_modifier=".wt", volatile=volatile
    )
    tl.store(results + offsets, result, mask=mask)


def test_load_default_cache():
    """Test load with default cache behavior."""
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    source_rank = shmem.get_rank()
    partner = int((source_rank + num_ranks // 2) % num_ranks)

    BLOCK_SIZE = 16
    data = shmem.full((BLOCK_SIZE,), source_rank, dtype=torch.float32)
    results = shmem.zeros_like(data)

    shmem.barrier()

    grid = lambda meta: (1,)
    load_kernel_default[grid](
        data,
        results,
        source_rank,
        num_ranks,
        BLOCK_SIZE,
        heap_bases,
        iris.cache_default,  # cache_modifier=cache_default (3)
        False,  # volatile=False
    )
    shmem.barrier()

    # Verify the result
    expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda") * partner
    torch.testing.assert_close(results, expected, rtol=0, atol=0)


def test_load_writeback_cache():
    """Test load with write-back cache behavior."""
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    source_rank = shmem.get_rank()
    partner = int((source_rank + num_ranks // 2) % num_ranks)

    BLOCK_SIZE = 16
    data = shmem.full((BLOCK_SIZE,), source_rank, dtype=torch.float32)
    results = shmem.zeros_like(data)

    shmem.barrier()

    grid = lambda meta: (1,)
    load_kernel_writeback[grid](
        data,
        results,
        source_rank,
        num_ranks,
        BLOCK_SIZE,
        heap_bases,
        False,  # volatile=False
    )
    shmem.barrier()

    # Verify the result
    expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda") * partner
    torch.testing.assert_close(results, expected, rtol=0, atol=0)


def test_load_cache_global():
    """Test load with cache global behavior."""
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    source_rank = shmem.get_rank()
    partner = int((source_rank + num_ranks // 2) % num_ranks)

    BLOCK_SIZE = 16
    data = shmem.full((BLOCK_SIZE,), source_rank, dtype=torch.float32)
    results = shmem.zeros_like(data)

    shmem.barrier()

    grid = lambda meta: (1,)
    load_kernel_cache_global[grid](
        data,
        results,
        source_rank,
        num_ranks,
        BLOCK_SIZE,
        heap_bases,
        False,  # volatile=False
    )
    shmem.barrier()

    # Verify the result
    expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda") * partner
    torch.testing.assert_close(results, expected, rtol=0, atol=0)


def test_load_cache_streaming():
    """Test load with cache streaming behavior."""
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    source_rank = shmem.get_rank()
    partner = int((source_rank + num_ranks // 2) % num_ranks)

    BLOCK_SIZE = 16
    data = shmem.full((BLOCK_SIZE,), source_rank, dtype=torch.float32)
    results = shmem.zeros_like(data)

    shmem.barrier()

    grid = lambda meta: (1,)
    load_kernel_cache_streaming[grid](
        data,
        results,
        source_rank,
        num_ranks,
        BLOCK_SIZE,
        heap_bases,
        False,  # volatile=False
    )
    shmem.barrier()

    # Verify the result
    expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda") * partner
    torch.testing.assert_close(results, expected, rtol=0, atol=0)


def test_load_write_through():
    """Test load with write-through behavior."""
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    source_rank = shmem.get_rank()
    partner = int((source_rank + num_ranks // 2) % num_ranks)

    BLOCK_SIZE = 16
    data = shmem.full((BLOCK_SIZE,), source_rank, dtype=torch.float32)
    results = shmem.zeros_like(data)

    shmem.barrier()

    grid = lambda meta: (1,)
    load_kernel_write_through[grid](
        data,
        results,
        source_rank,
        num_ranks,
        BLOCK_SIZE,
        heap_bases,
        False,  # volatile=False
    )
    shmem.barrier()

    # Verify the result
    expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda") * partner
    torch.testing.assert_close(results, expected, rtol=0, atol=0)


def test_load_volatile():
    """Test load with volatile=True."""
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    source_rank = shmem.get_rank()
    partner = int((source_rank + num_ranks // 2) % num_ranks)

    BLOCK_SIZE = 16
    data = shmem.full((BLOCK_SIZE,), source_rank, dtype=torch.float32)
    results = shmem.zeros_like(data)

    shmem.barrier()

    grid = lambda meta: (1,)
    load_kernel_default[grid](
        data,
        results,
        source_rank,
        num_ranks,
        BLOCK_SIZE,
        heap_bases,
        True,  # volatile=True
    )
    shmem.barrier()

    # Verify the result
    expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda") * partner
    torch.testing.assert_close(results, expected, rtol=0, atol=0)


if __name__ == "__main__":
    test_load_default_cache()
    # test_load_writeback_cache()
    # test_load_cache_global()
    # test_load_cache_streaming()
    # test_load_write_through()
    # test_load_volatile()
