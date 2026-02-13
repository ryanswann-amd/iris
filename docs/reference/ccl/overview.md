# iris.ccl - Collective Communication Library

The `iris.ccl` module provides high-level collective communication primitives for multi-GPU programming. These operations handle full tensors with automatic tiling and memory management.

## Overview

Collective operations are accessed through the `ccl` attribute of an Iris instance:

```python
import iris
import torch

# Initialize Iris context
ctx = iris.iris(heap_size=2**30)  # 1GB heap

# Create tensors
input_tensor = ctx.randn((1024, 2048), dtype=torch.float16)
output_tensor = ctx.zeros((1024, 2048), dtype=torch.float16)

# Perform collective operations
ctx.ccl.all_reduce(output_tensor, input_tensor)
ctx.ccl.all_gather(output_tensor, input_tensor)
ctx.ccl.all_to_all(output_tensor, input_tensor)
ctx.ccl.reduce_scatter(output_tensor, input_tensor)
```

## Available Operations

- **all_reduce**: Reduce values across all ranks and distribute the result
- **all_gather**: Gather data from all ranks and distribute to all ranks
- **all_to_all**: Scatter data from all ranks to all ranks
- **reduce_scatter**: Reduce values across all ranks and scatter the result

## Configuration

Operations can be tuned using the `Config` class:

```python
from iris.ccl import Config

config = Config(
    block_size_m=128,
    block_size_n=128,
    all_reduce_variant="ring",
    use_gluon=False
)

ctx.ccl.all_reduce(output_tensor, input_tensor, config=config)
```

## Asynchronous Operations

All collective operations support asynchronous execution:

```python
# Launch operation asynchronously
ctx.ccl.all_reduce(output_tensor, input_tensor, async_op=True)

# Do other work here...

# Synchronize
ctx.barrier()
```

## API Reference

Explore the API by section:

- [Operations](operations.md) - Collective communication primitives
- [Configuration](config.md) - Configuration options and tuning parameters
