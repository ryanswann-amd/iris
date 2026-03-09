# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for the power-of-two VMem allocator (VMemPow2Allocator).
"""

import gc

import pytest
import torch

import iris
from iris.allocators.vmem_pow2_allocator import _next_pow2


# ---------------------------------------------------------------------------
# Unit tests for _next_pow2 helper
# ---------------------------------------------------------------------------


def test_next_pow2_small():
    assert _next_pow2(1) == 1
    assert _next_pow2(2) == 2
    assert _next_pow2(3) == 4
    assert _next_pow2(4) == 4
    assert _next_pow2(5) == 8
    assert _next_pow2(7) == 8
    assert _next_pow2(8) == 8
    assert _next_pow2(9) == 16


def test_next_pow2_large():
    assert _next_pow2(1 << 20) == 1 << 20
    assert _next_pow2((1 << 20) + 1) == 1 << 21


def test_next_pow2_one():
    assert _next_pow2(0) == 1


# ---------------------------------------------------------------------------
# Allocator creation
# ---------------------------------------------------------------------------


def test_vmem_pow2_allocator_creation():
    """VMemPow2Allocator can be created via the iris context."""
    ctx = iris.iris(4 << 20, allocator_type="vmem_pow2")

    assert ctx.cur_rank >= 0
    assert ctx.num_ranks >= 1
    assert ctx.heap_size == 4 << 20

    from iris.allocators.vmem_pow2_allocator import VMemPow2Allocator

    assert isinstance(ctx.heap.allocator, VMemPow2Allocator)
    print(f"Rank {ctx.cur_rank}: VMemPow2Allocator created successfully.")


# ---------------------------------------------------------------------------
# Basic allocation
# ---------------------------------------------------------------------------


def test_vmem_pow2_basic_allocation():
    """Basic tensor allocation and write."""
    ctx = iris.iris(8 << 20, allocator_type="vmem_pow2")

    tensor = ctx.zeros(1024, dtype=torch.float32)

    assert tensor.shape == (1024,)
    assert tensor.device.type == "cuda"
    assert torch.all(tensor == 0)

    tensor.fill_(42.0)
    assert torch.all(tensor == 42.0)

    print(f"Rank {ctx.cur_rank}: basic allocation test passed.")


def test_vmem_pow2_multiple_allocations():
    """Multiple allocations from the same heap."""
    ctx = iris.iris(16 << 20, allocator_type="vmem_pow2")

    tensors = []
    for i in range(8):
        t = ctx.zeros(256, dtype=torch.float32)
        t.fill_(float(i))
        tensors.append(t)

    for i, t in enumerate(tensors):
        assert torch.all(t == float(i)), f"Tensor {i} has wrong value."

    print(f"Rank {ctx.cur_rank}: multiple allocations test passed.")


def test_vmem_pow2_different_dtypes():
    """Allocations with different dtypes."""
    ctx = iris.iris(16 << 20, allocator_type="vmem_pow2")

    t_f32 = ctx.zeros(128, dtype=torch.float32)
    t_f16 = ctx.zeros(128, dtype=torch.float16)
    t_i32 = ctx.zeros(128, dtype=torch.int32)

    t_f32.fill_(1.0)
    t_f16.fill_(2.0)
    t_i32.fill_(3)

    assert torch.all(t_f32 == 1.0)
    assert torch.all(t_f16 == 2.0)
    assert torch.all(t_i32 == 3)

    print(f"Rank {ctx.cur_rank}: different dtypes test passed.")


# ---------------------------------------------------------------------------
# owns_tensor
# ---------------------------------------------------------------------------


def test_vmem_pow2_owns_tensor():
    """owns_tensor correctly identifies heap vs. non-heap tensors."""
    ctx = iris.iris(8 << 20, allocator_type="vmem_pow2")

    heap_tensor = ctx.zeros(100, dtype=torch.float32)
    assert ctx.heap.allocator.owns_tensor(heap_tensor), "Heap tensor should be owned."

    external = torch.zeros(100, dtype=torch.float32, device=ctx.device)
    assert not ctx.heap.allocator.owns_tensor(external), "External tensor should not be owned."

    del heap_tensor, external
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print(f"Rank {ctx.cur_rank}: owns_tensor test passed.")


# ---------------------------------------------------------------------------
# Free-list reuse
# ---------------------------------------------------------------------------


def test_vmem_pow2_free_reuse():
    """After free(), the same physical block is returned on the next allocate()."""
    ctx = iris.iris(16 << 20, allocator_type="vmem_pow2")
    allocator = ctx.heap.allocator

    # Allocate a tensor and record its pointer.
    t1 = ctx.zeros(512, dtype=torch.float32)
    ptr1 = t1.data_ptr()

    # Free it.
    allocator.free(t1)

    # The next allocation of the same size class must reuse the same VA.
    t2 = ctx.zeros(512, dtype=torch.float32)
    ptr2 = t2.data_ptr()

    assert ptr2 == ptr1, f"Expected reuse of VA 0x{ptr1:x}, got 0x{ptr2:x}."
    print(f"Rank {ctx.cur_rank}: free-list reuse test passed (VA 0x{ptr1:x}).")


def test_vmem_pow2_free_reuse_multiple():
    """Multiple free + realloc cycles for different size classes."""
    ctx = iris.iris(64 << 20, allocator_type="vmem_pow2")
    allocator = ctx.heap.allocator

    for num_elems in [64, 256, 1024]:
        t = ctx.zeros(num_elems, dtype=torch.float32)
        ptr = t.data_ptr()
        allocator.free(t)

        t2 = ctx.zeros(num_elems, dtype=torch.float32)
        assert t2.data_ptr() == ptr, f"Expected VA reuse for {num_elems} elements."

    print(f"Rank {ctx.cur_rank}: multi-size free-list reuse test passed.")


def test_vmem_pow2_free_wrong_tensor_raises():
    """Freeing a non-heap tensor raises ValueError."""
    ctx = iris.iris(8 << 20, allocator_type="vmem_pow2")
    allocator = ctx.heap.allocator

    external = torch.zeros(64, dtype=torch.float32, device=ctx.device)
    with pytest.raises(ValueError):
        allocator.free(external)

    print(f"Rank {ctx.cur_rank}: free() error-check test passed.")


# ---------------------------------------------------------------------------
# Heap bases
# ---------------------------------------------------------------------------


def test_vmem_pow2_heap_bases():
    """Heap bases are properly initialised."""
    ctx = iris.iris(4 << 20, allocator_type="vmem_pow2")

    assert ctx.heap_bases.shape == (ctx.num_ranks,)
    assert int(ctx.heap_bases[ctx.cur_rank].item()) > 0

    if ctx.num_ranks > 1:
        for peer in range(ctx.num_ranks):
            if peer != ctx.cur_rank:
                assert int(ctx.heap_bases[peer].item()) > 0
                assert int(ctx.heap_bases[peer].item()) != int(ctx.heap_bases[ctx.cur_rank].item())

    print(f"Rank {ctx.cur_rank}: heap bases test passed.")


# ---------------------------------------------------------------------------
# Granularity alignment
# ---------------------------------------------------------------------------


def test_vmem_pow2_granularity_alignment():
    """The aligned heap size must be a multiple of the HIP granularity."""
    from iris.hip import get_allocation_granularity

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    ctx = iris.iris(4 << 20, allocator_type="vmem_pow2")
    granularity = get_allocation_granularity(ctx.gpu_id)

    assert ctx.heap.allocator.aligned_heap_size % granularity == 0
    print(f"Rank {ctx.cur_rank}: granularity alignment test passed (granularity={granularity}).")


# ---------------------------------------------------------------------------
# Size-class rounding
# ---------------------------------------------------------------------------


def test_vmem_pow2_size_class_rounding():
    """
    Each allocation is rounded up to the nearest power-of-two >= granularity.
    Verify by checking that two allocations of slightly different sizes that
    round to the same size class produce VA blocks of the same physical size
    and can be interchangeably reused.
    """
    ctx = iris.iris(64 << 20, allocator_type="vmem_pow2")
    allocator = ctx.heap.allocator
    granularity = allocator.granularity

    # Two sizes that both round up to 2*granularity.
    size_a = granularity + 1  # bytes
    size_b = granularity + granularity // 2  # bytes

    elem_size = torch.tensor([], dtype=torch.int8).element_size()  # 1
    elems_a = size_a // elem_size
    elems_b = size_b // elem_size

    # Allocate with size_a, free, then allocate with size_b – should reuse.
    t_a = allocator.allocate(elems_a, torch.int8)
    ptr_a = t_a.data_ptr()
    allocator.free(t_a)

    t_b = allocator.allocate(elems_b, torch.int8)
    ptr_b = t_b.data_ptr()

    assert ptr_b == ptr_a, (
        f"Expected VA reuse: both sizes should share size class. ptr_a=0x{ptr_a:x}, ptr_b=0x{ptr_b:x}"
    )
    print(f"Rank {ctx.cur_rank}: size-class rounding test passed.")


# ---------------------------------------------------------------------------
# stats()
# ---------------------------------------------------------------------------


def test_vmem_pow2_stats():
    """stats() returns sensible values."""
    ctx = iris.iris(8 << 20, allocator_type="vmem_pow2")
    allocator = ctx.heap.allocator

    s0 = allocator.stats()
    assert s0["heap_size"] == 8 << 20
    assert s0["granularity"] > 0
    assert s0["num_live_allocations"] == 0

    t = ctx.zeros(512, dtype=torch.float32)
    s1 = allocator.stats()
    assert s1["num_live_allocations"] == 1

    allocator.free(t)
    s2 = allocator.stats()
    assert s2["num_live_allocations"] == 0

    print(f"Rank {ctx.cur_rank}: stats() test passed.")


# ---------------------------------------------------------------------------
# get_allocation_segments()
# ---------------------------------------------------------------------------


def test_vmem_pow2_allocation_segments_grow():
    """
    get_allocation_segments() grows when new physical segments are mapped
    but does NOT grow when free-listed blocks are reused.
    """
    ctx = iris.iris(64 << 20, allocator_type="vmem_pow2")
    allocator = ctx.heap.allocator

    # Segments after init (bootstrap only).
    seg_count_0 = len(allocator.get_allocation_segments())

    # First allocation -> maps a new segment.
    t1 = ctx.zeros(512, dtype=torch.float32)
    seg_count_1 = len(allocator.get_allocation_segments())
    assert seg_count_1 == seg_count_0 + 1

    # Free and reallocate same size -> reuses free-list, no new segment.
    allocator.free(t1)
    t2 = ctx.zeros(512, dtype=torch.float32)
    seg_count_2 = len(allocator.get_allocation_segments())
    assert seg_count_2 == seg_count_1, "Free-list reuse must not create a new segment."

    print(f"Rank {ctx.cur_rank}: allocation segments grow test passed.")


# ---------------------------------------------------------------------------
# as_symmetric() (import_external_tensor)
# ---------------------------------------------------------------------------


def test_vmem_pow2_import_external_tensor():
    """
    Importing an external tensor gives a symmetric-heap view that shares
    physical memory; the original tensor remains valid after ctx is destroyed.
    """
    ctx = iris.iris(8 << 20, allocator_type="vmem_pow2")

    original = torch.randn(64, dtype=torch.float32, device=ctx.device)
    original_data = original.clone()

    imported = ctx.as_symmetric(original)
    assert torch.allclose(imported, original_data), "Imported data should match original."

    # Mutation via imported is visible in original.
    imported.fill_(7.0)
    assert torch.all(original == 7.0), "Original should see changes through shared memory."

    # Mutation via original is visible in imported.
    original.fill_(13.0)
    assert torch.all(imported == 13.0), "Imported should see changes through shared memory."

    # Destroy ctx – original must survive.
    del ctx, imported
    gc.collect()
    torch.cuda.synchronize()

    assert torch.all(original == 13.0), "Original tensor should survive ctx destruction."
    original.fill_(99.0)
    assert torch.all(original == 99.0), "Original tensor should still be writable."

    print("import_external_tensor test passed.")


# ---------------------------------------------------------------------------
# Multi-rank tests
# ---------------------------------------------------------------------------


def test_vmem_pow2_multirank_heap_bases():
    """Multi-rank: each rank sees all peers' heap bases."""
    ctx = iris.iris(4 << 20, allocator_type="vmem_pow2")

    tensor = ctx.zeros(1024, dtype=torch.float32)
    tensor.fill_(float(ctx.cur_rank * 100))

    assert ctx.heap_bases.shape == (ctx.num_ranks,)
    assert int(ctx.heap_bases[ctx.cur_rank].item()) > 0

    if ctx.num_ranks > 1:
        for peer in range(ctx.num_ranks):
            if peer != ctx.cur_rank:
                assert int(ctx.heap_bases[peer].item()) > 0
                assert int(ctx.heap_bases[peer].item()) != int(ctx.heap_bases[ctx.cur_rank].item())

    ctx.barrier()
    tensor.fill_(float(ctx.cur_rank * 100))
    ctx.barrier()
    assert torch.all(tensor == float(ctx.cur_rank * 100))

    print(f"Rank {ctx.cur_rank}: multi-rank heap-bases test passed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_next_pow2_small()
    test_next_pow2_large()
    test_next_pow2_one()
    test_vmem_pow2_allocator_creation()
    test_vmem_pow2_basic_allocation()
    test_vmem_pow2_multiple_allocations()
    test_vmem_pow2_different_dtypes()
    test_vmem_pow2_owns_tensor()
    test_vmem_pow2_free_reuse()
    test_vmem_pow2_free_reuse_multiple()
    test_vmem_pow2_heap_bases()
    test_vmem_pow2_granularity_alignment()
    test_vmem_pow2_stats()
    test_vmem_pow2_allocation_segments_grow()
    test_vmem_pow2_import_external_tensor()
    print("All VMemPow2Allocator tests passed.")
