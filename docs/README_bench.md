# iris.bench - Unified Benchmarking Harness

A standardized benchmarking infrastructure for Iris that reduces code duplication and provides consistent performance measurement across examples and benchmarks.

## Quick Start

```python
import iris
from iris.bench import benchmark

# Simple decorator-based benchmarking
@benchmark(name="my_kernel", warmup=5, iters=50)
def run_kernel():
    kernel[grid](buffer, size)

result = run_kernel()
result.print_summary()
```

## Features

- ✅ **Automatic warmup and timing** - No more manual warmup loops
- ✅ **Rich statistics** - mean, median, p50, p99, min, max
- ✅ **Parameter sweeps** - Easy iteration over configurations
- ✅ **Multi-GPU support** - Built-in barrier synchronization
- ✅ **JSON export** - Structured results for CI/CD integration
- ✅ **Utility functions** - `torch_dtype_from_str`, `compute_bandwidth_gbps`

## What Problem Does This Solve?

Before `iris.bench`, every benchmark had ~100 lines of duplicated code for:
- Argument parsing (datatype, warmup, iterations)
- Dtype string-to-torch conversion
- Manual warmup loops
- Timing and synchronization
- Result formatting and printing

This led to:
- 🔴 Copy-pasted code across 20+ benchmark files
- 🔴 Inconsistent measurement patterns
- 🔴 No standardized statistics (p50, p99)
- 🔴 Hard to maintain and extend

With `iris.bench`:
- ✅ ~50% less code per benchmark
- ✅ Standardized API across all benchmarks
- ✅ Easy to add new benchmarks
- ✅ CI-ready JSON export

## Examples

### Example 1: Simple Benchmark
```python
from iris.bench import BenchmarkRunner

runner = BenchmarkRunner(name="test", barrier_fn=shmem.barrier)

def operation():
    kernel[grid](buffer)

result = runner.run(fn=operation, warmup=5, iters=50)
result.print_summary()
```

### Example 2: Parameter Sweep
```python
from iris.bench import BenchmarkRunner, torch_dtype_from_str

runner = BenchmarkRunner(name="dtype_sweep")

for dtype_str in ["fp16", "fp32"]:
    for size in [1024, 2048]:
        dtype = torch_dtype_from_str(dtype_str)
        
        def op():
            tensor = torch.zeros(size, size, dtype=dtype, device="cuda")
            result = tensor @ tensor
        
        runner.run(fn=op, warmup=5, iters=20, 
                  params={"size": size, "dtype": dtype_str})

runner.save_json("results.json")
```

## Documentation

- 📖 [Full API Documentation](bench_harness.md)
- 📖 [Migration Guide](bench_migration_example.md)
- 💻 [Complete Examples](../examples/benchmark/bench_harness_example.py)

## Testing

```bash
# Run basic tests (no GPU required)
python3 tests/unittests/test_bench_basic.py

# Run full test suite (requires GPU)
pytest tests/unittests/test_bench.py
```

## API Overview

### BenchmarkResult
Stores benchmark results with automatic statistics computation.

### BenchmarkRunner
Main class for running benchmarks with parameter sweeps.

### @benchmark
Decorator for simple function benchmarking.

### Utilities
- `torch_dtype_from_str(dtype_str)` - Convert string to torch.dtype
- `compute_bandwidth_gbps(bytes, time_ms)` - Calculate bandwidth

## Integration

The harness is designed to work alongside existing `iris.do_bench` usage:
- `BenchmarkRunner` internally uses `iris.do_bench`
- All existing barrier functions work with `barrier_fn` parameter
- Gradual migration path - old benchmarks continue to work

## Contributing

When adding new benchmarks:
1. Use `iris.bench` for all new code
2. Consider migrating nearby old benchmarks
3. Export results to JSON for CI integration
4. Follow examples in `examples/benchmark/`

## License

MIT License - Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
