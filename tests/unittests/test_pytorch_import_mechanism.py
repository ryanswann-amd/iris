# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test PyTorch tensor import mechanism and as_symmetric() functionality.

This file tests the complete pipeline for importing PyTorch tensors into
VMem allocator's VA space:

Section 1: Low-level mechanism tests (test_dmabuf_*)
  - DMA-BUF export/import with offset preservation
  - The foundation that as_symmetric() is built on
  - Tests the exact behavior we depend on

Section 2: High-level as_symmetric() integration tests (test_as_symmetric_*)
  - Native VMem allocations + imported external tensors
  - Both living in the same VA space
  - End-to-end functionality
"""

import torch
import pytest
import os
import iris


# =============================================================================
# Section 1: Low-Level Mechanism Tests
# =============================================================================


def test_dmabuf_import_with_offset():
    """
    Core mechanism test: Validate PyTorch tensor import with offset preservation.
    
    This test validates the EXACT mechanism that as_symmetric() relies on:
    1. PyTorch caching allocator creates suballocations with offsets
    2. hipMemExportToShareableHandle returns BASE allocation (not suballocated ptr)
    3. hipMemGetAddressRange tells us the offset
    4. import_dmabuf_handle() imports base and adds offset back to access correct data
    
    This is the CRITICAL PATH that MUST work for as_symmetric() to function.
    """
    from iris.hip import (
        export_dmabuf_handle,
        get_address_range,
        import_dmabuf_handle,
        destroy_external_memory,
    )
    
    # STEP 1: Create a PyTorch tensor with GUARANTEED offset (via slicing)
    large_tensor = torch.randn(10000, dtype=torch.float32, device='cuda')
    
    # Slice to create offset (skip 1000 floats = 4000 bytes)
    sliced_tensor = large_tensor[1000:2000]
    sliced_tensor.fill_(42.0)  # Fill with known value
    
    original_ptr = sliced_tensor.data_ptr()
    original_size = sliced_tensor.element_size() * sliced_tensor.numel()
    
    # STEP 2: Query the base allocation and offset
    alloc_base, alloc_size = get_address_range(original_ptr)
    offset_in_alloc = original_ptr - alloc_base
    
    # CRITICAL ASSERTION 1: We must have an offset (slice guarantees this)
    assert offset_in_alloc > 0, f"Expected non-zero offset, got {offset_in_alloc}"
    expected_min_offset = 1000 * 4  # 1000 floats * 4 bytes
    assert offset_in_alloc >= expected_min_offset, (
        f"Expected offset >= {expected_min_offset}, got {offset_in_alloc}"
    )
    
    # STEP 3: Export DMA-BUF (should return BASE, not our pointer)
    dmabuf_fd, export_base, export_size = export_dmabuf_handle(original_ptr, original_size)
    
    # CRITICAL ASSERTION 2: Export returns base allocation, not our offset pointer
    assert export_base == alloc_base, (
        f"Export should return base {hex(alloc_base)}, got {hex(export_base)}"
    )
    assert export_size == alloc_size, (
        f"Export should return full size {alloc_size}, got {export_size}"
    )
    
    try:
        # STEP 4: Import the DMA-BUF with automatic offset correction
        remapped_ptr, ext_mem_handle = import_dmabuf_handle(dmabuf_fd, export_size, original_ptr, export_base)
        
        # STEP 5: Create a tensor from the remapped pointer
        class CUDAArrayInterface:
            def __init__(self, ptr, size):
                self.ptr = ptr
                self.size = size
            
            @property
            def __cuda_array_interface__(self):
                return {
                    "shape": (self.size // 4,),  # float32 = 4 bytes
                    "typestr": "<f4",
                    "data": (self.ptr, False),
                    "version": 3,
                }
        
        cuda_array = CUDAArrayInterface(remapped_ptr, original_size)
        remapped_tensor = torch.as_tensor(cuda_array, device='cuda')
        
        # STEP 6: CRITICAL TEST - Verify data is correct!
        torch.cuda.synchronize()
        
        # CRITICAL ASSERTION 3: Remapped tensor should have the same data
        assert remapped_tensor.shape == sliced_tensor.shape, (
            f"Shape mismatch: remapped={remapped_tensor.shape}, "
            f"original={sliced_tensor.shape}"
        )
        
        assert torch.allclose(remapped_tensor, sliced_tensor), (
            "Remapped tensor should have same data as original!\n"
            f"Original: {sliced_tensor[:10].tolist()}\n"
            f"Remapped: {remapped_tensor[:10].tolist()}"
        )
        
        # Verify with known fill value
        assert torch.all(remapped_tensor == 42.0), (
            "Remapped tensor should have value 42.0 (our fill value)"
        )
        
        # STEP 7: Test write access
        remapped_tensor.fill_(99.0)
        torch.cuda.synchronize()
        
        # The original tensor should now also be 99.0 (same physical memory!)
        assert torch.all(sliced_tensor == 99.0), (
            "Original tensor should reflect write to remapped tensor "
            "(same physical memory)"
        )
        assert torch.all(remapped_tensor == 99.0), (
            "Remapped tensor should have new value"
        )
        
    finally:
        # Cleanup: destroy external memory handle
        destroy_external_memory(ext_mem_handle)


def test_dmabuf_import_no_offset():
    """
    Mechanism test with NO offset (tensor at start of allocation).
    
    Validates the mechanism still works when offset = 0.
    """
    from iris.hip import (
        export_dmabuf_handle,
        get_address_range,
        import_dmabuf_handle,
        destroy_external_memory,
    )
    
    # Create a fresh allocation (likely offset = 0)
    torch.cuda.empty_cache()
    tensor = torch.zeros(1000, dtype=torch.float32, device='cuda')
    tensor.fill_(123.0)
    
    original_ptr = tensor.data_ptr()
    original_size = tensor.element_size() * tensor.numel()
    
    # Query base
    alloc_base, alloc_size = get_address_range(original_ptr)
    offset_in_alloc = original_ptr - alloc_base
    
    # Export DMA-BUF
    dmabuf_fd, export_base, export_size = export_dmabuf_handle(original_ptr, original_size)
    
    assert export_base == alloc_base
    
    try:
        # Import with offset correction (offset might be 0)
        remapped_ptr, ext_mem_handle = import_dmabuf_handle(dmabuf_fd, export_size, original_ptr, export_base)
        
        # Create tensor
        class CUDAArrayInterface:
            def __init__(self, ptr, size):
                self.ptr = ptr
                self.size = size
            
            @property
            def __cuda_array_interface__(self):
                return {
                    "shape": (self.size // 4,),
                    "typestr": "<f4",
                    "data": (self.ptr, False),
                    "version": 3,
                }
        
        cuda_array = CUDAArrayInterface(remapped_ptr, original_size)
        remapped_tensor = torch.as_tensor(cuda_array, device='cuda')
        
        # Verify data
        torch.cuda.synchronize()
        assert torch.all(remapped_tensor == 123.0), (
            "Should work correctly even with offset = 0"
        )
        
    finally:
        # Cleanup: destroy external memory handle
        destroy_external_memory(ext_mem_handle)


# =============================================================================
# Section 2: High-Level as_symmetric() Integration Tests
# =============================================================================


def test_as_symmetric_basic():
    """
    Test basic as_symmetric() functionality.
    
    Creates a native VMem tensor and an imported external tensor,
    verifying both work correctly in the same VA space.
    """
    ctx = iris.iris(1 << 20, allocator_type="vmem")  # 1 MB heap
    
    # 1. Allocate a native tensor from our VMem heap
    native_tensor = ctx.zeros(1000, dtype=torch.float32)
    native_tensor.fill_(42.0)
    native_ptr = native_tensor.data_ptr()
    
    # Verify native tensor is in our VA space
    heap_base = ctx.symmetric_heap.allocator.get_base_address()
    assert native_ptr >= heap_base, "Native tensor should be in our VA space"
    
    # 2. Create an external PyTorch tensor (caching allocator)
    external_tensor = torch.randn(500, dtype=torch.float32, device='cuda')
    external_tensor.fill_(99.0)
    external_ptr = external_tensor.data_ptr()
    
    # 3. Import external tensor via as_symmetric()
    imported_tensor = ctx.as_symmetric(external_tensor)
    imported_ptr = imported_tensor.data_ptr()
    
    # ASSERTION 1: Imported tensor should have different pointer (remapped into our VA)
    assert imported_ptr != external_ptr, (
        f"Imported tensor should be remapped, "
        f"external={hex(external_ptr)}, imported={hex(imported_ptr)}"
    )
    
    # ASSERTION 2: Imported tensor should be in our VA space
    assert imported_ptr >= heap_base, "Imported tensor should be in our VA space"
    
    # ASSERTION 3: Both tensors should be accessible
    torch.cuda.synchronize()
    assert torch.all(native_tensor == 42.0), "Native tensor should be accessible"
    assert torch.all(imported_tensor == 99.0), "Imported tensor should be accessible"
    
    # ASSERTION 4: Modifications to imported tensor should work
    imported_tensor.fill_(123.0)
    torch.cuda.synchronize()
    assert torch.all(imported_tensor == 123.0), "Imported tensor should be writable"
    
    # ASSERTION 5: Original external tensor should still be accessible
    # (it's backed by different physical memory)
    assert torch.all(external_tensor == 99.0), "External tensor should be unchanged"


def test_as_symmetric_with_offset():
    """
    Test as_symmetric() with a sliced tensor (guaranteed offset).
    
    This tests the critical PyTorch caching allocator offset handling.
    """
    ctx = iris.iris(1 << 20, allocator_type="vmem")  # 1 MB heap
    
    # Create a large external tensor
    large_external = torch.randn(10000, dtype=torch.float32, device='cuda')
    
    # Create a slice with guaranteed offset (skip 1000 elements = 4KB)
    sliced_external = large_external[1000:2000]
    sliced_external.fill_(55.0)
    
    # Import the sliced tensor
    imported_slice = ctx.as_symmetric(sliced_external)
    
    # ASSERTION 1: Should have different pointer
    assert imported_slice.data_ptr() != sliced_external.data_ptr()
    
    # ASSERTION 2: Should preserve the data correctly despite offset
    torch.cuda.synchronize()
    assert torch.all(imported_slice == 55.0), (
        "Imported sliced tensor should preserve data despite offset"
    )
    
    # ASSERTION 3: Modifications should work
    imported_slice.fill_(77.0)
    torch.cuda.synchronize()
    assert torch.all(imported_slice == 77.0)


def test_as_symmetric_multiple_imports():
    """
    Test importing multiple external tensors.
    
    Verifies that multiple imports work correctly and don't interfere
    with each other or native allocations.
    """
    ctx = iris.iris(2 << 20, allocator_type="vmem")  # 2 MB heap
    
    # Native allocations
    native1 = ctx.zeros(500, dtype=torch.float32)
    native1.fill_(1.0)
    
    # Import multiple external tensors
    external_tensors = []
    imported_tensors = []
    
    for i in range(5):
        ext = torch.zeros(200, dtype=torch.float32, device='cuda')
        ext.fill_(float(10 + i))
        external_tensors.append(ext)
        
        imp = ctx.as_symmetric(ext)
        imported_tensors.append(imp)
    
    # Another native allocation after imports
    native2 = ctx.zeros(500, dtype=torch.float32)
    native2.fill_(2.0)
    
    # Verify all tensors are accessible and have correct values
    torch.cuda.synchronize()
    
    assert torch.all(native1 == 1.0), "Native1 should be accessible"
    assert torch.all(native2 == 2.0), "Native2 should be accessible"
    
    for i, imp in enumerate(imported_tensors):
        expected_val = float(10 + i)
        assert torch.all(imp == expected_val), (
            f"Imported tensor {i} should have value {expected_val}"
        )


def test_as_symmetric_with_different_dtypes():
    """
    Test as_symmetric() with various data types.
    """
    ctx = iris.iris(1 << 20, allocator_type="vmem")  # 1 MB heap
    
    test_cases = [
        (torch.float32, 42.0),
        (torch.float16, 16.0),
        (torch.int32, 123),
        (torch.int64, 456),
    ]
    
    for dtype, fill_value in test_cases:
        # Create external tensor
        if dtype in [torch.float32, torch.float16]:
            ext = torch.randn(100, dtype=dtype, device='cuda')
        else:
            ext = torch.randint(0, 1000, (100,), dtype=dtype, device='cuda')
        
        ext.fill_(fill_value)
        
        # Import it
        imp = ctx.as_symmetric(ext)
        
        # Verify
        torch.cuda.synchronize()
        assert torch.all(imp == fill_value), f"Failed for dtype {dtype}"


def test_as_symmetric_preserves_shape():
    """
    Test that as_symmetric() preserves tensor shape and strides.
    """
    ctx = iris.iris(1 << 20, allocator_type="vmem")  # 1 MB heap
    
    # Test different shapes
    shapes = [
        (100,),
        (10, 20),
        (5, 10, 4),
    ]
    
    for shape in shapes:
        ext = torch.randn(*shape, dtype=torch.float32, device='cuda')
        imp = ctx.as_symmetric(ext)
        
        # ASSERTION 1: Shape should be preserved
        assert imp.shape == ext.shape, f"Shape mismatch for {shape}"
        
        # ASSERTION 2: Should be contiguous (our implementation always creates contiguous)
        assert imp.is_contiguous(), f"Imported tensor should be contiguous for {shape}"


def test_as_symmetric_error_non_cuda():
    """
    Test that as_symmetric() raises error for non-CUDA tensors.
    """
    ctx = iris.iris(1 << 20, allocator_type="vmem")  # 1 MB heap
    
    # CPU tensor
    cpu_tensor = torch.randn(100, dtype=torch.float32)
    
    with pytest.raises(RuntimeError, match="must be on CUDA"):
        ctx.as_symmetric(cpu_tensor)


def test_as_symmetric_error_torch_allocator():
    """
    Test that as_symmetric() raises error when used with TorchAllocator.
    
    as_symmetric() is a VMem-specific feature.
    """
    # Create context with default (Torch) allocator
    ctx = iris.iris(1 << 20)  # Default allocator
    
    ext = torch.randn(100, dtype=torch.float32, device='cuda')
    
    # Should raise error - TorchAllocator doesn't support import_external_tensor
    with pytest.raises(RuntimeError, match="does not support"):
        ctx.as_symmetric(ext)


def test_as_symmetric_integration():
    """
    Full integration test: native allocations + imports + computations.
    
    Tests the real use case: mixing native and imported tensors in computations.
    """
    ctx = iris.iris(2 << 20, allocator_type="vmem")  # 2 MB heap
    
    # Step 1: Create some native tensors
    a = ctx.zeros(1000, dtype=torch.float32)
    a.fill_(1.0)
    
    b = ctx.zeros(1000, dtype=torch.float32)
    b.fill_(2.0)
    
    # Step 2: Import an external tensor
    external = torch.ones(1000, dtype=torch.float32, device='cuda') * 3.0
    c = ctx.as_symmetric(external)
    
    # Step 3: Do computations mixing native and imported
    result = a + b + c  # 1 + 2 + 3 = 6
    
    torch.cuda.synchronize()
    assert torch.all(result == 6.0), "Mixed computation should work correctly"
    
    # Step 4: Verify original tensors are still correct
    assert torch.all(a == 1.0), "Native tensor a should be unchanged"
    assert torch.all(b == 2.0), "Native tensor b should be unchanged"
    assert torch.all(c == 3.0), "Imported tensor c should be unchanged"
    assert torch.all(external == 3.0), "External tensor should be unchanged"
