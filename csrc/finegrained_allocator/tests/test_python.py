#!/usr/bin/env python3

import torch
import numpy as np
import time
import sys
import os

# Add the parent directory to the path to import our module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch_allocator
except ImportError:
    print("torch_allocator module not found. Please build it first.")
    sys.exit(1)

def test_basic_functionality():
    """Test basic allocator functionality"""
    print("Testing basic functionality...")
    
    # Create allocator
    allocator = torch_allocator.create_allocator(heap_size=1024*1024, device_id=0)
    
    # Test allocation
    tensor = allocator.allocate_tensor([100, 100], torch.float32)
    assert tensor.defined(), "Tensor should be defined"
    assert tensor.shape == (100, 100), f"Expected shape (100, 100), got {tensor.shape}"
    assert tensor.dtype == torch.float32, f"Expected dtype float32, got {tensor.dtype}"
    assert tensor.device.type == 'cuda', f"Expected CUDA device, got {tensor.device}"
    
    # Test deallocation
    allocator.deallocate_tensor(tensor)
    print("Basic functionality test passed!")

def test_multiple_allocations():
    """Test multiple allocations and deallocations"""
    print("Testing multiple allocations...")
    
    allocator = torch_allocator.create_allocator(heap_size=10*1024*1024, device_id=0)
    
    tensors = []
    
    # Allocate multiple tensors
    for i in range(10):
        tensor = allocator.allocate_tensor([1000, 1000], torch.float32)
        tensors.append(tensor)
    
    assert allocator.get_num_allocations() == 10, f"Expected 10 allocations, got {allocator.get_num_allocations()}"
    
    # Deallocate half
    for i in range(5):
        allocator.deallocate_tensor(tensors[i])
    
    assert allocator.get_num_allocations() == 5, f"Expected 5 allocations, got {allocator.get_num_allocations()}"
    
    # Deallocate rest
    for i in range(5, 10):
        allocator.deallocate_tensor(tensors[i])
    
    assert allocator.get_num_allocations() == 0, f"Expected 0 allocations, got {allocator.get_num_allocations()}"
    
    print("Multiple allocations test passed!")

def test_different_dtypes():
    """Test different data types"""
    print("Testing different dtypes...")
    
    allocator = torch_allocator.create_allocator(heap_size=1024*1024, device_id=0)
    
    # Test different dtypes
    tensor_f32 = allocator.allocate_tensor([100, 100], torch.float32)
    tensor_f64 = allocator.allocate_tensor([100, 100], torch.float64)
    tensor_i32 = allocator.allocate_tensor([100, 100], torch.int32)
    tensor_i64 = allocator.allocate_tensor([100, 100], torch.int64)
    
    assert tensor_f32.dtype == torch.float32
    assert tensor_f64.dtype == torch.float64
    assert tensor_i32.dtype == torch.int32
    assert tensor_i64.dtype == torch.int64
    
    # Cleanup
    allocator.deallocate_tensor(tensor_f32)
    allocator.deallocate_tensor(tensor_f64)
    allocator.deallocate_tensor(tensor_i32)
    allocator.deallocate_tensor(tensor_i64)
    
    print("Different dtypes test passed!")

def test_convenience_functions():
    """Test convenience functions"""
    print("Testing convenience functions...")
    
    allocator = torch_allocator.create_allocator(heap_size=1024*1024, device_id=0)
    
    # Test allocate_zeros
    zeros_tensor = torch_allocator.allocate_zeros(allocator, [50, 50], torch.float32)
    assert torch.allclose(zeros_tensor, torch.zeros(50, 50, device='cuda:0')), "Zeros tensor not properly initialized"
    allocator.deallocate_tensor(zeros_tensor)
    
    # Test allocate_ones
    ones_tensor = torch_allocator.allocate_ones(allocator, [50, 50], torch.float32)
    assert torch.allclose(ones_tensor, torch.ones(50, 50, device='cuda:0')), "Ones tensor not properly initialized"
    allocator.deallocate_tensor(ones_tensor)
    
    # Test allocate_empty
    empty_tensor = torch_allocator.allocate_empty(allocator, [50, 50], torch.float32)
    assert empty_tensor.shape == (50, 50), "Empty tensor has wrong shape"
    allocator.deallocate_tensor(empty_tensor)
    
    print("Convenience functions test passed!")

def test_memory_reuse():
    """Test memory reuse and fragmentation"""
    print("Testing memory reuse...")
    
    allocator = torch_allocator.create_allocator(heap_size=1024*1024, device_id=0)
    
    # Allocate and deallocate repeatedly
    for i in range(100):
        tensor = allocator.allocate_tensor([100, 100], torch.float32)
        allocator.deallocate_tensor(tensor)
    
    assert allocator.get_num_allocations() == 0, f"Expected 0 allocations after reuse test, got {allocator.get_num_allocations()}"
    
    print("Memory reuse test passed!")

def test_performance():
    """Test performance"""
    print("Testing performance...")
    
    allocator = torch_allocator.create_allocator(heap_size=100*1024*1024, device_id=0)
    
    num_iterations = 1000
    tensor_size = 1000
    
    start_time = time.time()
    
    for i in range(num_iterations):
        tensor = allocator.allocate_tensor([tensor_size, tensor_size], torch.float32)
        allocator.deallocate_tensor(tensor)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Performance test: {num_iterations} allocations/deallocations in {duration:.3f} seconds")
    print(f"Average time per allocation: {duration/num_iterations*1000:.3f} ms")
    
    allocator.print_stats()

def test_statistics():
    """Test allocator statistics"""
    print("Testing statistics...")
    
    allocator = torch_allocator.create_allocator(heap_size=1024*1024, device_id=0)
    
    # Initial stats
    assert allocator.get_num_allocations() == 0
    assert allocator.get_total_allocated() == 0
    
    # Allocate some tensors
    tensors = []
    for i in range(5):
        tensor = allocator.allocate_tensor([100, 100], torch.float32)
        tensors.append(tensor)
    
    assert allocator.get_num_allocations() == 5
    assert allocator.get_total_allocated() > 0
    
    # Deallocate some
    for i in range(3):
        allocator.deallocate_tensor(tensors[i])
    
    assert allocator.get_num_allocations() == 2
    
    # Cleanup
    for i in range(3, 5):
        allocator.deallocate_tensor(tensors[i])
    
    print("Statistics test passed!")

def test_edge_cases():
    """Test edge cases"""
    print("Testing edge cases...")
    
    allocator = torch_allocator.create_allocator(heap_size=1024*1024, device_id=0)
    
    # Test empty tensor
    empty_tensor = allocator.allocate_tensor([], torch.float32)
    assert empty_tensor.numel() == 1, "Empty tensor should have 1 element"
    allocator.deallocate_tensor(empty_tensor)
    
    # Test scalar tensor
    scalar_tensor = allocator.allocate_tensor([1], torch.float32)
    assert scalar_tensor.numel() == 1, "Scalar tensor should have 1 element"
    allocator.deallocate_tensor(scalar_tensor)
    
    # Test deallocating undefined tensor
    undefined_tensor = torch.tensor([])
    try:
        allocator.deallocate_tensor(undefined_tensor)
        print("Deallocating undefined tensor handled gracefully")
    except Exception as e:
        print(f"Deallocating undefined tensor failed: {e}")
    
    print("Edge cases test passed!")

def main():
    """Run all tests"""
    print("Starting PyTorch Allocator Tests")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_multiple_allocations()
        test_different_dtypes()
        test_convenience_functions()
        test_memory_reuse()
        test_performance()
        test_statistics()
        test_edge_cases()
        
        print("\n" + "=" * 50)
        print("All tests passed! 🎉")
        return 0
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
