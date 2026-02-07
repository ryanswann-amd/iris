# Benchmarking Harness (iris.bench)

The `iris.bench` module provides a unified infrastructure for benchmarking Iris operations. It standardizes warmup and iteration handling, timing and synchronization, statistics computation, parameter sweeps, and structured result output.

## Overview

The benchmarking harness reduces code duplication across `examples/` and `benchmark/` directories by providing reusable components for:

- **Warmup and iteration handling**: Automatic warmup runs before timing measurements
- **Timing and synchronization**: Built-in barrier support for multi-GPU synchronization
- **Statistics**: Automatic computation of mean, median, p50, p99, min, and max times
- **Parameter sweeps**: Easy iteration over different configurations
- **Structured output**: JSON export and human-readable summaries

## Quick Start

### Using the @benchmark Decorator

The simplest way to benchmark a function:

```python
from iris.bench import benchmark

@benchmark(name="my_kernel", warmup=5, iters=50)
def run_kernel(size):
    # Your benchmark code here
    kernel[grid](buffer, size)

# Run and get results
result = run_kernel(1024)
result.print_summary()
```

### Using BenchmarkRunner

For more control and parameter sweeps:

```python
from iris.bench import BenchmarkRunner

runner = BenchmarkRunner(name="gemm_sweep", barrier_fn=shmem.barrier)

for size in [1024, 2048, 4096]:
    def operation():
        # Your benchmark code
        kernel[grid](buffer, size)
    
    runner.run(fn=operation, warmup=5, iters=50, params={"size": size})

# Get all results
results = runner.get_results()
runner.print_summary()
runner.save_json("results.json")
```

## API Reference

### BenchmarkResult

Dataclass storing benchmark results.

**Attributes:**
- `name: str` - Benchmark name
- `mean_ms: float` - Mean time in milliseconds
- `median_ms: float` - Median time in milliseconds
- `p50_ms: float` - 50th percentile (same as median)
- `p99_ms: float` - 99th percentile
- `min_ms: float` - Minimum time
- `max_ms: float` - Maximum time
- `n_warmup: int` - Number of warmup iterations
- `n_repeat: int` - Number of timing iterations
- `params: Dict[str, Any]` - Additional parameters
- `metadata: Dict[str, Any]` - Additional metadata
- `raw_times: List[float]` - Raw timing measurements

**Methods:**
- `to_dict(include_raw_times=False)` - Convert to dictionary
- `to_json(include_raw_times=False, indent=2)` - Convert to JSON string
- `print_summary()` - Print human-readable summary

### BenchmarkRunner

Context manager and runner for benchmarks with parameter sweeps.

**Constructor:**
```python
BenchmarkRunner(name: str, barrier_fn: Optional[Callable] = None)
```

**Parameters:**
- `name` - Name of the benchmark suite
- `barrier_fn` - Optional barrier function for multi-GPU synchronization (e.g., `shmem.barrier`)

**Methods:**
- `run(fn, warmup=25, iters=100, params=None)` - Run a single benchmark
  - `fn` - Function to benchmark
  - `warmup` - Number of warmup iterations
  - `iters` - Number of timing iterations
  - `params` - Additional parameters to store with result
  - Returns: `BenchmarkResult`

- `get_results()` - Get all benchmark results
- `print_summary()` - Print summary of all results
- `save_json(filepath, include_raw_times=False)` - Save results to JSON file

### @benchmark Decorator

Decorator for benchmarking functions.

**Parameters:**
- `name: str` - Benchmark name
- `warmup: int = 25` - Number of warmup iterations
- `iters: int = 100` - Number of timing iterations
- `barrier_fn: Optional[Callable] = None` - Barrier function for synchronization
- `auto_print: bool = False` - Whether to automatically print results
- `params: Optional[Dict] = None` - Additional parameters

**Returns:** Function that returns `BenchmarkResult`

### Utility Functions

#### torch_dtype_from_str

Convert string datatype to `torch.dtype`.

```python
dtype = torch_dtype_from_str("fp16")  # torch.float16
```

Supported types: `"int8"`, `"fp16"`, `"bf16"`, `"fp32"`

#### compute_bandwidth_gbps

Compute bandwidth in GiB/s.

```python
bandwidth = compute_bandwidth_gbps(total_bytes, time_ms)
```

**Parameters:**
- `total_bytes: int` - Total bytes transferred
- `time_ms: float` - Time in milliseconds

**Returns:** Bandwidth in GiB/s

## Examples

### Example 1: Simple Benchmark

```python
import torch
from iris.bench import benchmark

@benchmark(name="vector_add", warmup=5, iters=50)
def bench_vector_add(size=1024):
    a = torch.randn(size, device="cuda")
    b = torch.randn(size, device="cuda")
    c = a + b
    return c

result = bench_vector_add()
result.print_summary()
```

### Example 2: Multi-GPU Benchmark with Barrier

```python
import iris
from iris.bench import BenchmarkRunner

# Initialize Iris
shmem = iris.iris(heap_size=1 << 33)

runner = BenchmarkRunner(
    name="multi_gpu_bench",
    barrier_fn=shmem.barrier  # Synchronize across GPUs
)

def operation():
    # Your multi-GPU operation
    tensor = shmem.zeros(1024, 1024)
    # ... operations ...

result = runner.run(fn=operation, warmup=5, iters=50)
result.print_summary()
```

### Example 3: Parameter Sweep

```python
from iris.bench import BenchmarkRunner, torch_dtype_from_str

runner = BenchmarkRunner(name="dtype_sweep")

for dtype_str in ["fp16", "fp32"]:
    for size in [1024, 2048, 4096]:
        dtype = torch_dtype_from_str(dtype_str)
        
        def operation():
            tensor = torch.zeros(size, size, dtype=dtype, device="cuda")
            result = tensor @ tensor
            return result
        
        runner.run(
            fn=operation,
            warmup=5,
            iters=20,
            params={"size": size, "dtype": dtype_str}
        )

runner.print_summary()
runner.save_json("sweep_results.json")
```

### Example 4: Bandwidth Benchmark

```python
from iris.bench import BenchmarkRunner, compute_bandwidth_gbps
import torch

size = 1024 * 1024 * 100  # 100M elements
dtype = torch.float16
element_size = torch.tensor([], dtype=dtype).element_size()

def copy_operation():
    src = torch.randn(size, dtype=dtype, device="cuda")
    dst = src.clone()
    return dst

runner = BenchmarkRunner(name="bandwidth_test")
result = runner.run(fn=copy_operation, warmup=5, iters=50)

total_bytes = size * element_size
bandwidth = compute_bandwidth_gbps(total_bytes, result.mean_ms)

print(f"Bandwidth: {bandwidth:.2f} GiB/s")
```

## Migration Guide

### Before (Old Pattern)

```python
import argparse
import iris

# Duplicate argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-w", "--num_warmup", type=int, default=1)
parser.add_argument("-n", "--num_experiments", type=int, default=10)
args = vars(parser.parse_args())

# Manual warmup and timing
def run_experiment():
    kernel[grid](...)

# Warmup
run_experiment()
shmem.barrier()

# Benchmark
triton_ms = iris.do_bench(
    run_experiment,
    shmem.barrier,
    n_repeat=args["num_experiments"],
    n_warmup=args["num_warmup"]
)

# Manual statistics and printing
print(f"Time: {triton_ms:.4f} ms")
```

### After (New Pattern)

```python
import iris
from iris.bench import BenchmarkRunner

# Initialize
shmem = iris.iris(heap_size=1 << 33)
runner = BenchmarkRunner(name="my_bench", barrier_fn=shmem.barrier)

# Benchmark with automatic warmup, timing, and statistics
def operation():
    kernel[grid](...)

result = runner.run(fn=operation, warmup=5, iters=50)
result.print_summary()  # Automatic formatting with mean/p50/p99
```

## Integration with Existing Code

The benchmark harness is designed to work alongside existing `iris.do_bench` usage. You can gradually migrate benchmarks to use the new infrastructure while maintaining backward compatibility.

### Compatibility

- `BenchmarkRunner` internally uses `iris.do_bench` for timing
- All existing barrier functions work with `barrier_fn` parameter
- Results can be exported to JSON for integration with CI/CD pipelines
- The module is available as `iris.bench` after importing `iris`

## See Also

- `iris.do_bench()` - Lower-level timing function used internally
- `examples/benchmark/bench_harness_example.py` - Complete working examples
