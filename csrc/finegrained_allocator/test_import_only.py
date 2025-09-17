#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch_allocator

def test_import_only():
    print("Import test starting...")
    print("PyTorch imported successfully")
    print("torch_allocator imported successfully")
    print("Test completed successfully")

if __name__ == "__main__":
    test_import_only()
