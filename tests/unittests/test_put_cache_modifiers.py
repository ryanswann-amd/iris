# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest
import iris
from itertools import product


@triton.jit
def put_kernel(
    data,
    results,
    source_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
    load_cache_modifier: tl.constexpr,
    store_cache_modifier: tl.constexpr,
):
    pid = tl.program_id(0)

    partner = int((source_rank + num_ranks // 2) % num_ranks)
    # Compute start index of this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Guard for out-of-bounds accesses
    mask = offsets < BLOCK_SIZE

    # Load the data from local memory
    value = tl.load(data + offsets, mask=mask)

    # Copy data to partner rank using put with cache modifiers
    # We test default values set by the function when parameters are None
    if load_cache_modifier is None and store_cache_modifier is None:
        iris.put(data + offsets, results + offsets, source_rank, partner, heap_bases, mask=mask)
    elif load_cache_modifier is None:
        iris.put(
            data + offsets,
            results + offsets,
            source_rank,
            partner,
            heap_bases,
            mask=mask,
            store_cache_modifier=store_cache_modifier,
        )
    elif store_cache_modifier is None:
        iris.put(
            data + offsets,
            results + offsets,
            source_rank,
            partner,
            heap_bases,
            mask=mask,
            load_cache_modifier=load_cache_modifier,
        )
    else:
        iris.put(
            data + offsets,
            results + offsets,
            source_rank,
            partner,
            heap_bases,
            mask=mask,
            load_cache_modifier=load_cache_modifier,
            store_cache_modifier=store_cache_modifier,
        )


# Define cache modifiers for load and store operations
LOAD_CACHE_MODIFIERS = [None, "", ".ca", ".cg", ".cv"]
STORE_CACHE_MODIFIERS = [None, "", ".wb", ".cg", ".cs", ".wt"]


@pytest.mark.parametrize(
    "load_cache_modifier,store_cache_modifier", list(product(LOAD_CACHE_MODIFIERS, STORE_CACHE_MODIFIERS))
)
def test_put_cache_modifiers(load_cache_modifier, store_cache_modifier):
    """Test put (copy to other rank) with various cache modifiers."""
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
    put_kernel[grid](
        data, results, source_rank, num_ranks, BLOCK_SIZE, heap_bases, load_cache_modifier, store_cache_modifier
    )
    shmem.barrier()

    # Verify the result - each rank should have its own data
    expected = torch.ones(BLOCK_SIZE, dtype=torch.float32, device="cuda") * source_rank

    try:
        torch.testing.assert_close(results, expected, rtol=0, atol=0)
    except AssertionError as e:
        print(
            f"PUT test failed with load_cache_modifier={load_cache_modifier}, store_cache_modifier={store_cache_modifier}"
        )
        print(e)
        print("Expected:", expected)
        print("Actual:", results)
        raise
