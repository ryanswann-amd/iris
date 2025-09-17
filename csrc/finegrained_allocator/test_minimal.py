#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch_allocator


def test_minimal():
    print("Minimal test starting...")

    # Create allocator
    print("Creating allocator...")
    allocator = torch_allocator.create_allocator(heap_size=1024 * 1024, device_id=0)
    print("Allocator created successfully")

    # Create one tensor
    print("Creating tensor...")
    tensor = allocator.allocate_tensor([100, 100], torch.float32, 0)
    print(f"Tensor created: {tensor.shape}, {tensor.dtype}")

    # Use tensor
    tensor.fill_(1.0)
    print(f"Tensor sum: {tensor.sum().item()}")

    print("Test completed successfully")


if __name__ == "__main__":
    test_minimal()
