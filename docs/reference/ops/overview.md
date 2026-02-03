# iris.ops - Fused GEMM+CCL Operations

The `iris.ops` module provides high-level APIs for fused matrix multiplication and collective communication operations. These operations automatically infer dimensions, strides, and hardware parameters from input tensors.

## Overview

Fused operations combine GEMM (General Matrix Multiply) with collective communication, enabling better performance through computation-communication overlap.

```python
import iris
import torch

# Initialize Iris context
shmem = iris.iris(heap_size=2**30)  # 1GB heap

# Create tensors
M, K, N = 1024, 2048, 512
A = shmem.randn((M, K), dtype=torch.float16)
B = shmem.randn((K, N), dtype=torch.float16)
output = shmem.zeros((M, N), dtype=torch.float16)

# Fused GEMM + All-Reduce
shmem.ops.matmul_all_reduce(output, A, B)
```

## Available Operations

- **matmul_all_reduce**: Compute `output = all_reduce(A @ B + bias)`
- **all_gather_matmul**: Compute `output = all_gather(A_sharded) @ B + bias`
- **matmul_all_gather**: Compute `output = all_gather(A @ B + bias)` along M dimension
- **matmul_reduce_scatter**: Compute `output = reduce_scatter(A @ B + bias)` along N dimension

## Usage Patterns

### Via shmem.ops namespace (recommended)

```python
shmem = iris.iris(heap_size)
A = shmem.randn((M, K), dtype=torch.float16)
B = shmem.randn((K, N), dtype=torch.float16)
output = shmem.zeros((M, N), dtype=torch.float16)

# Call through shmem.ops
shmem.ops.matmul_all_reduce(output, A, B)
```

### Standalone usage

```python
import iris.ops as ops

shmem = iris.iris(heap_size)
A = shmem.randn((M, K), dtype=torch.float16)
B = shmem.randn((K, N), dtype=torch.float16)
output = shmem.zeros((M, N), dtype=torch.float16)

# Pass shmem as first parameter
ops.matmul_all_reduce(shmem, output, A, B)
```

## Configuration

Operations can be tuned using the `FusedConfig` class:

```python
from iris.ops import FusedConfig

config = FusedConfig(
    block_size_m=128,
    block_size_n=128,
    block_size_k=32
)

shmem.ops.matmul_all_reduce(output, A, B, config=config)
```

## Workspace Management

For repeated operations, you can pre-allocate and reuse workspace:

```python
from iris.ops import FusedWorkspace

# Pre-allocate workspace
workspace = FusedWorkspace()
workspace = shmem.ops.matmul_all_reduce(output, A, B, workspace=workspace)

# Reuse workspace in subsequent calls
workspace = shmem.ops.matmul_all_reduce(output, A, B, workspace=workspace)
```

## Asynchronous Operations

All operations support asynchronous execution:

```python
# Launch operation asynchronously
shmem.ops.matmul_all_reduce(output, A, B, async_op=True)

# Do other work here...

# Synchronize
shmem.barrier()
```

## API Reference

Explore the API by section:

- [Operations](operations.md) - Fused GEMM+CCL operations
- [Configuration](config.md) - Configuration options and workspace management
