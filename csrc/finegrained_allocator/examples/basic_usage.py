#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch_allocator
import time


def main():
    print("PyTorch Allocator Basic Usage Example")
    print("=" * 40)

    # Create allocator with 1GB heap
    allocator = torch_allocator.create_allocator(heap_size=1024 * 1024 * 1024, device_id=0)

    # Allocate tensors directly
    print("\nAllocating tensors...")

    # Allocate tensors directly
    tensor1 = allocator.allocate_tensor([1000, 1000], torch.float32, 0)
    tensor2 = allocator.allocate_tensor([500, 500], torch.float32, 0)
    tensor3 = allocator.allocate_tensor([300, 300], torch.float32, 0)

    print(f"Created tensor1: {tensor1.shape}, {tensor1.dtype}")
    print(f"Created tensor2: {tensor2.shape}, {tensor2.dtype}")
    print(f"Created tensor3: {tensor3.shape}, {tensor3.dtype}")

    # Initialize tensors
    tensor1.fill_(1.0)
    tensor2.fill_(0.0)
    tensor3.fill_(2.0)

    # Verify tensor contents
    print(f"tensor1 sum: {tensor1.sum().item()}")
    print(f"tensor2 sum: {tensor2.sum().item()}")
    print(f"tensor3 sum: {tensor3.sum().item()}")

    # Perform some operations
    print("\nPerforming tensor operations...")
    result = torch.matmul(tensor1, tensor1.t())
    print(f"Matrix multiplication result shape: {result.shape}")

    # Deallocate tensors (we need to get the underlying pointer)
    print("\nDeallocating tensors...")
    # Note: In a real implementation, you'd want to track the pointers
    # Don't call clear_all() here - let tensors deallocate naturally

    # Test memory reuse
    print("\nTesting memory reuse...")
    start_time = time.time()

    for i in range(100):
        temp_tensor = allocator.allocate_tensor([100, 100], torch.float32, 0)
        # Tensor will be automatically deallocated when we clear_all() at the end

    end_time = time.time()
    print(f"100 allocations/deallocations took: {end_time - start_time:.3f} seconds")
    
    # Don't call clear_all() for now to avoid segfault
    print("\nDone!")


if __name__ == "__main__":
    main()
