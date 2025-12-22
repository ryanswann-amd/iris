#!/usr/bin/env python3
"""
Quick environment check for Triton Reference MoE
"""

import sys
import os

print("=" * 80)
print("Environment Check for Triton Reference MoE")
print("=" * 80)

# Check Python version
print(f"\n✓ Python version: {sys.version}")

# Check PyTorch
try:
    import torch

    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError as e:
    print(f"✗ PyTorch not found: {e}")
    sys.exit(1)

# Check Triton
try:
    import triton

    print(f"✓ Triton version: {triton.__version__}")
except ImportError as e:
    print(f"✗ Triton not found: {e}")
    sys.exit(1)

# Check triton_kernels
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from triton_kernels import distributed, topk, matmul, reduce, tensor

    print("✓ triton_kernels imported successfully")
    print("  - distributed")
    print("  - topk")
    print("  - matmul")
    print("  - reduce")
    print("  - tensor")
except ImportError as e:
    print(f"✗ triton_kernels import failed: {e}")
    sys.exit(1)

# Check NCCL
try:
    import torch.distributed as dist

    print("✓ torch.distributed available")
    print(f"  NCCL available: {dist.is_nccl_available()}")
except Exception as e:
    print(f"✗ torch.distributed check failed: {e}")

print("\n" + "=" * 80)
print("Environment check complete!")
print("=" * 80)
print("\nYou can now run:")
print("  python test_triton_reference.py")
print("=" * 80)
