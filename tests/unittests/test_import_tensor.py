# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
import iris


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.uint8,
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (100,),
        (10, 10),
        (5, 5, 5),
        (2, 3, 4, 5),
    ],
)
def test_import_tensor_basic(dtype, shape):
    """Test basic import_tensor with various shapes and dtypes"""
    shmem = iris.iris(1 << 28)  # 256MB

    # Create external tensor
    external = torch.ones(shape, device="cuda", dtype=dtype)

    # Import it
    imported = shmem.import_tensor(external)

    # Verify shape and dtype match
    assert imported.shape == external.shape
    assert imported.dtype == external.dtype

    # Verify it's on symmetric heap
    assert shmem._Iris__on_symmetric_heap(imported)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
    ],
)
def test_import_tensor_shared_memory(dtype):
    """Test that imported tensor shares physical memory with external tensor"""
    shmem = iris.iris(1 << 28)

    # Create external tensor
    external = torch.ones(100, 100, device="cuda", dtype=dtype)

    # Import it
    imported = shmem.import_tensor(external)

    # Modify external tensor
    if dtype in [torch.float32, torch.float64]:
        external[0, 0] = 42.0
        assert imported[0, 0].item() == 42.0
    else:
        external[0, 0] = 42
        assert imported[0, 0].item() == 42

    # Modify imported tensor
    if dtype in [torch.float32, torch.float64]:
        imported[1, 1] = 99.0
        assert external[1, 1].item() == 99.0
    else:
        imported[1, 1] = 99
        assert external[1, 1].item() == 99


def test_import_only():
    """Test importing without any allocations"""
    shmem = iris.iris(1 << 28)

    # Just import a tensor
    external = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    imported = shmem.import_tensor(external)

    # Verify it works
    assert imported.shape == (256, 256)
    assert shmem._Iris__on_symmetric_heap(imported)

    # Verify shared memory (use approximate comparison for float32)
    external[0, 0] = 123.456
    assert torch.isclose(imported[0, 0], torch.tensor(123.456, device="cuda"), rtol=1e-5)


def test_allocate_then_import():
    """Test allocating first, then importing"""
    shmem = iris.iris(1 << 28)

    # Allocate some tensors first
    alloc1 = shmem.zeros(100, 100, dtype=torch.float32)
    alloc2 = shmem.ones(50, 50, dtype=torch.float32)

    # Now import
    external = torch.randn(200, 200, device="cuda", dtype=torch.float32)
    imported = shmem.import_tensor(external)

    # Verify all are on symmetric heap
    assert shmem._Iris__on_symmetric_heap(alloc1)
    assert shmem._Iris__on_symmetric_heap(alloc2)
    assert shmem._Iris__on_symmetric_heap(imported)

    # Verify shared memory for imported
    external[0, 0] = 999.0
    assert imported[0, 0].item() == 999.0


def test_import_then_allocate():
    """Test importing first, then allocating"""
    shmem = iris.iris(1 << 28)

    # Import first
    external = torch.randn(100, 100, device="cuda", dtype=torch.float32)
    imported = shmem.import_tensor(external)

    # Now allocate
    alloc1 = shmem.zeros(50, 50, dtype=torch.float32)
    alloc2 = shmem.ones(200, 200, dtype=torch.float32)

    # Verify all are on symmetric heap
    assert shmem._Iris__on_symmetric_heap(imported)
    assert shmem._Iris__on_symmetric_heap(alloc1)
    assert shmem._Iris__on_symmetric_heap(alloc2)

    # Verify shared memory for imported
    external[10, 10] = 777.0
    assert imported[10, 10].item() == 777.0


@pytest.mark.parametrize(
    "size",
    [
        1,
        100,
        1000,
        10000,
        100000,
    ],
)
def test_import_different_sizes(size):
    """Test importing tensors of different sizes"""
    shmem = iris.iris(1 << 28)

    external = torch.randn(size, device="cuda", dtype=torch.float32)
    imported = shmem.import_tensor(external)

    assert imported.shape == (size,)
    assert shmem._Iris__on_symmetric_heap(imported)

    # Verify shared memory
    external[0] = 555.0
    assert imported[0].item() == 555.0


def test_multiple_imports():
    """Test importing multiple tensors"""
    shmem = iris.iris(1 << 28)

    # Import multiple tensors
    externals = []
    importeds = []

    for i in range(5):
        ext = torch.full((100, 100), i, device="cuda", dtype=torch.float32)
        imp = shmem.import_tensor(ext)
        externals.append(ext)
        importeds.append(imp)

    # Verify all are on symmetric heap
    for imp in importeds:
        assert shmem._Iris__on_symmetric_heap(imp)

    # Verify shared memory for each
    for i, (ext, imp) in enumerate(zip(externals, importeds)):
        ext[0, 0] = float(i * 100)
        assert imp[0, 0].item() == float(i * 100)


def test_multiple_allocates():
    """Test multiple allocations"""
    shmem = iris.iris(1 << 28)

    # Allocate multiple tensors
    tensors = []
    for i in range(5):
        t = shmem.full((100, 100), i, dtype=torch.float32)
        tensors.append(t)

    # Verify all are on symmetric heap
    for t in tensors:
        assert shmem._Iris__on_symmetric_heap(t)


def test_mixed_allocate_import():
    """Test mixed allocation and import operations"""
    shmem = iris.iris(1 << 28)

    # Allocate
    alloc1 = shmem.zeros(50, 50, dtype=torch.float32)

    # Import
    ext1 = torch.ones(100, 100, device="cuda", dtype=torch.float32)
    imp1 = shmem.import_tensor(ext1)

    # Allocate
    alloc2 = shmem.full((75, 75), 2.0, dtype=torch.float32)

    # Import
    ext2 = torch.randn(200, 200, device="cuda", dtype=torch.float32)
    imp2 = shmem.import_tensor(ext2)

    # Allocate
    alloc3 = shmem.ones(25, 25, dtype=torch.float32)

    # Verify all are on symmetric heap
    assert shmem._Iris__on_symmetric_heap(alloc1)
    assert shmem._Iris__on_symmetric_heap(imp1)
    assert shmem._Iris__on_symmetric_heap(alloc2)
    assert shmem._Iris__on_symmetric_heap(imp2)
    assert shmem._Iris__on_symmetric_heap(alloc3)

    # Verify shared memory for imports
    ext1[0, 0] = 111.0
    assert imp1[0, 0].item() == 111.0

    ext2[0, 0] = 222.0
    assert imp2[0, 0].item() == 222.0


@pytest.mark.parametrize(
    "num_ops",
    [5, 10, 20],
)
def test_many_mixed_operations(num_ops):
    """Test many mixed allocate/import operations"""
    shmem = iris.iris(1 << 28)

    tensors = []
    externals = []

    for i in range(num_ops):
        if i % 2 == 0:
            # Allocate
            t = shmem.zeros(50, 50, dtype=torch.float32)
            tensors.append(t)
        else:
            # Import
            ext = torch.ones(50, 50, device="cuda", dtype=torch.float32) * i
            imp = shmem.import_tensor(ext)
            tensors.append(imp)
            externals.append(ext)

    # Verify all are on symmetric heap
    for t in tensors:
        assert shmem._Iris__on_symmetric_heap(t)

    # Verify shared memory for imports
    for i, ext in enumerate(externals):
        ext[0, 0] = float(i * 1000)


def test_import_tensor_metadata():
    """Test that imported tensor has correct metadata"""
    shmem = iris.iris(1 << 28)

    external = torch.randn(100, 100, device="cuda", dtype=torch.float32)
    imported = shmem.import_tensor(external)

    # Check metadata attributes
    assert hasattr(imported, "_iris_vmem_ptr")
    assert hasattr(imported, "_iris_vmem_size")
    assert hasattr(imported, "_iris_imported")
    assert hasattr(imported, "_iris_external_ptr")

    # Verify metadata values
    assert imported._iris_imported is True
    assert imported._iris_vmem_ptr > 0
    assert imported._iris_vmem_size > 0
    assert imported._iris_external_ptr == external.data_ptr()


def test_import_allocator_stats():
    """Test that allocator stats are updated for imports"""
    shmem = iris.iris(1 << 28)

    initial_allocs = shmem.vmem_allocator.active_allocations()

    # Import a tensor
    external = torch.randn(1000, 1000, device="cuda", dtype=torch.float32)
    imported = shmem.import_tensor(external)

    final_allocs = shmem.vmem_allocator.active_allocations()

    # Should have increased by 1
    assert final_allocs == initial_allocs + 1


@pytest.mark.parametrize(
    "shape1,shape2,shape3",
    [
        ((100,), (200,), (300,)),
        ((10, 10), (20, 20), (30, 30)),
        ((5, 5, 5), (10, 10, 10), (15, 15, 15)),
    ],
)
def test_import_different_shapes_sequence(shape1, shape2, shape3):
    """Test importing tensors with different shapes in sequence"""
    shmem = iris.iris(1 << 28)

    # Import three tensors with different shapes
    ext1 = torch.ones(shape1, device="cuda", dtype=torch.float32)
    imp1 = shmem.import_tensor(ext1)

    ext2 = torch.ones(shape2, device="cuda", dtype=torch.float32)
    imp2 = shmem.import_tensor(ext2)

    ext3 = torch.ones(shape3, device="cuda", dtype=torch.float32)
    imp3 = shmem.import_tensor(ext3)

    # Verify shapes
    assert imp1.shape == shape1
    assert imp2.shape == shape2
    assert imp3.shape == shape3

    # Verify all on symmetric heap
    assert shmem._Iris__on_symmetric_heap(imp1)
    assert shmem._Iris__on_symmetric_heap(imp2)
    assert shmem._Iris__on_symmetric_heap(imp3)


def test_import_edge_cases():
    """Test import edge cases"""
    shmem = iris.iris(1 << 28)

    # Small tensor (single element)
    ext_small = torch.tensor([42.0], device="cuda", dtype=torch.float32)
    imp_small = shmem.import_tensor(ext_small)
    assert imp_small.shape == (1,)
    assert shmem._Iris__on_symmetric_heap(imp_small)
    ext_small[0] = 99.0
    assert imp_small[0].item() == 99.0

    # Large tensor
    ext_large = torch.randn(1000, 1000, device="cuda", dtype=torch.float32)
    imp_large = shmem.import_tensor(ext_large)
    assert imp_large.shape == (1000, 1000)
    assert shmem._Iris__on_symmetric_heap(imp_large)
    ext_large[0, 0] = 123.0
    assert imp_large[0, 0].item() == 123.0


def test_import_with_pytorch_operations():
    """Test that imported tensors work with PyTorch operations"""
    shmem = iris.iris(1 << 28)

    # Create and import tensors
    ext1 = torch.ones(100, 100, device="cuda", dtype=torch.float32)
    imp1 = shmem.import_tensor(ext1)

    ext2 = torch.ones(100, 100, device="cuda", dtype=torch.float32) * 2
    imp2 = shmem.import_tensor(ext2)

    # Perform operations
    result = imp1 + imp2
    assert torch.allclose(result, torch.full((100, 100), 3.0, device="cuda"))

    result = imp1 * imp2
    assert torch.allclose(result, torch.full((100, 100), 2.0, device="cuda"))

    result = torch.matmul(imp1, imp2)
    assert result.shape == (100, 100)


def test_import_consecutive_layout():
    """Test that imports and allocations are laid out consecutively in VA space"""
    shmem = iris.iris(1 << 28)

    # Do a series of operations
    alloc1 = shmem.zeros(100, dtype=torch.float32)
    alloc1_ptr = alloc1.data_ptr()

    ext1 = torch.ones(100, device="cuda", dtype=torch.float32)
    imp1 = shmem.import_tensor(ext1)
    imp1_ptr = imp1.data_ptr()

    alloc2 = shmem.zeros(100, dtype=torch.float32)
    alloc2_ptr = alloc2.data_ptr()

    # All pointers should be in ascending order (consecutive allocations)
    assert alloc1_ptr < imp1_ptr < alloc2_ptr

    # Verify all on symmetric heap
    assert shmem._Iris__on_symmetric_heap(alloc1)
    assert shmem._Iris__on_symmetric_heap(imp1)
    assert shmem._Iris__on_symmetric_heap(alloc2)
