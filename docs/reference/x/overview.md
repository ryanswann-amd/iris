# iris.x - Device-Side Tile-Level Primitives

The `iris.x` module provides composable tile-level functions for fine-grained compute and collective operations. Unlike `iris.ccl` which handles full tensors, `iris.x` operates on individual tiles, allowing users to manage tile iteration themselves within custom Triton kernels.

## Overview

`iris.x` enables users to build custom kernels with precise control over tile-level operations:

```python
import iris
import iris.x
import triton
import triton.language as tl

@triton.jit
def my_kernel(input_ptr, output_ptr, context_tensor: tl.tensor,
              M: tl.constexpr, N: tl.constexpr,
              stride_m: tl.constexpr, stride_n: tl.constexpr,
              rank: tl.constexpr, world_size: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Get tile coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create views
    tile = iris.x.TileView(pid_m, pid_n, BLOCK_M, BLOCK_N)
    src_view = iris.x.make_tensor_view(input_ptr, M, N, stride_m, stride_n)
    dst_view = iris.x.make_tensor_view(output_ptr, M, N, stride_m, stride_n)
    ctx = iris.DeviceContext.initialize(context_tensor, rank, world_size)
    
    # Perform tile-level collective operation
    iris.x.all_reduce_atomic(tile, dst_view, ctx)
```

## Core Abstractions

- **TileView**: Represents a tile's position and size in a 2D grid
- **TensorView**: Represents a tensor's memory layout (pointer, shape, strides)
- **make_tensor_view**: Factory function to create TensorView in JIT context
- **AllReduceConfig**: Configuration for selecting all-reduce algorithms

**Note:** `iris.DeviceContext` (from the main iris module) is used for device-side context, not `iris.x.DeviceContext`.

## Usage Patterns

### Standalone Functions (Recommended)

The recommended approach is to call collective operations directly as standalone functions:

```python
@triton.jit
def kernel(input_ptr, output_ptr, context_tensor: tl.tensor,
           M: tl.constexpr, N: tl.constexpr,
           stride_m: tl.constexpr, stride_n: tl.constexpr,
           rank: tl.constexpr, world_size: tl.constexpr,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    tile = iris.x.TileView(pid_m, pid_n, BLOCK_M, BLOCK_N)
    dst_view = iris.x.make_tensor_view(output_ptr, M, N, stride_m, stride_n)
    ctx = iris.DeviceContext.initialize(context_tensor, rank, world_size)
    
    # Call collectives directly
    iris.x.all_reduce_atomic(tile, dst_view, ctx)
    iris.x.all_gather(tile, dst_view, dst_view, dim=0, ctx)
    iris.x.all_to_all(tile, dst_view, dst_view, N_per_rank, ctx)
```

### Algorithm Selection

Use `AllReduceConfig` to select specific all-reduce algorithms. The config takes an integer variant code (0-4) and a locks pointer:

```python
@triton.jit
def kernel(input_ptr, output_ptr, locks_ptr, context_tensor: tl.tensor,
           M: tl.constexpr, N: tl.constexpr,
           stride_m: tl.constexpr, stride_n: tl.constexpr,
           rank: tl.constexpr, world_size: tl.constexpr,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    tile = iris.x.TileView(pid_m, pid_n, BLOCK_M, BLOCK_N)
    dst_view = iris.x.make_tensor_view(output_ptr, M, N, stride_m, stride_n)
    ctx = iris.DeviceContext.initialize(context_tensor, rank, world_size)
    
    # Use ring algorithm (variant_code = 1)
    # For variants that don't need locks, pass a dummy tensor
    dummy_locks = tl.zeros((1,), dtype=tl.int32)
    config = iris.x.AllReduceConfig(1, dummy_locks)
    iris.x.all_reduce_ring(tile, dst_view, ctx)
    
    # Use spinlock algorithm with locks (variant_code = 4)
    config = iris.x.AllReduceConfig(4, locks_ptr)
    tile_id = pid_m * num_tiles_n + pid_n
    iris.x.all_reduce_spinlock(tile, dst_view, locks_ptr, ctx)
```

**Variant codes:**
- 0 = atomic
- 1 = ring
- 2 = one_shot
- 3 = two_shot
- 4 = spinlock

## Available Operations

### All-Reduce Variants

- **all_reduce_atomic**: Atomic-based all-reduce (default)
- **all_reduce_ring**: Ring algorithm
- **all_reduce_two_shot**: Two-shot algorithm
- **all_reduce_one_shot**: One-shot algorithm
- **all_reduce_spinlock**: Spinlock-based algorithm

### Other Collectives

- **all_gather**: Gather data from all ranks
- **all_to_all**: Scatter-gather across all ranks
- **reduce_scatter**: Reduce and scatter across ranks
- **gather**: Point-to-point gather operation

## Helper Functions

- **tile_layout**: Compute memory layout for a tile
- **tile_ptr**: Compute pointer to tile data
- **offset_ptr**: Offset a pointer by tile coordinates
- **make_tensor_view**: Factory function to create TensorView in JIT context

## API Reference

Explore the API by section:

- [Core Abstractions](core.md) - TileView, TensorView, DeviceContext, and helper functions
- [Operations](operations.md) - Device-side collective operations
