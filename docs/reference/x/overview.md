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
def my_kernel(input_ptr, output_ptr, M, N, 
              rank, world_size, heap_bases,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Get tile coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create views
    tile = iris.x.TileView(pid_m, pid_n, BLOCK_M, BLOCK_N)
    src_view = iris.x.TensorView(input_ptr, M, N, stride_m=N, stride_n=1)
    dst_view = iris.x.TensorView(output_ptr, M, N, stride_m=N, stride_n=1)
    ctx = iris.x.DeviceContext(rank, world_size, heap_bases)
    
    # Perform tile-level collective operation
    ctx.all_reduce(tile, src_view, dst_view)
```

## Core Abstractions

- **TileView**: Represents a tile's position and size in a 2D grid
- **TensorView**: Represents a tensor's memory layout (pointer, shape, strides)
- **DeviceContext**: Holds rank, world size, and heap bases for communication
- **AllReduceConfig**: Configuration for selecting all-reduce algorithms

## Usage Patterns

### Using DeviceContext (Recommended)

The `DeviceContext` provides a clean API for calling collectives:

```python
@triton.jit
def kernel(input_ptr, output_ptr, ...):
    tile = iris.x.TileView(pid_m, pid_n, BLOCK_M, BLOCK_N)
    src_view = iris.x.TensorView(input_ptr, M, N, stride_m, stride_n)
    dst_view = iris.x.TensorView(output_ptr, M, N, stride_m, stride_n)
    ctx = iris.x.DeviceContext(rank, world_size, heap_bases)
    
    # Call collectives with default algorithms
    ctx.all_reduce(tile, src_view, dst_view)
    ctx.all_gather(tile, src_view, dst_view, dim=0)
    ctx.all_to_all(tile, src_view, dst_view, N_per_rank)
    ctx.reduce_scatter(tile, src_view, dst_view)
```

### Algorithm Selection

Use `AllReduceConfig` to select specific all-reduce algorithms:

```python
@triton.jit
def kernel(input_ptr, output_ptr, locks_ptr, ...):
    ctx = iris.x.DeviceContext(rank, world_size, heap_bases)
    
    # Use ring algorithm
    config = iris.x.AllReduceConfig("ring")
    ctx.all_reduce(tile, src_view, dst_view, config=config)
    
    # Use spinlock algorithm with locks
    config = iris.x.AllReduceConfig("spinlock", locks_ptr)
    tile_id = pid_m * num_tiles_n + pid_n
    ctx.all_reduce(tile, src_view, dst_view, config=config, tile_id=tile_id)
```

### Standalone Functions

You can also call operations directly without `DeviceContext`:

```python
@triton.jit
def kernel(input_ptr, output_ptr, ...):
    ctx = iris.x.DeviceContext(rank, world_size, heap_bases)
    
    # Call operations directly
    iris.x.all_reduce_atomic(tile, src_view, dst_view, ctx)
    iris.x.all_reduce_ring(tile, src_view, dst_view, ctx)
    iris.x.all_gather(tile, src_view, dst_view, dim, ctx)
```

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

## API Reference

Explore the API by section:

- [Core Abstractions](core.md) - TileView, TensorView, DeviceContext, and helper functions
- [Operations](operations.md) - Device-side collective operations
