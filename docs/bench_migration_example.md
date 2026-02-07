# Benchmark Harness Migration Example

This document shows a concrete example of how to migrate an existing Iris benchmark to use the new `iris.bench` module.

## Before: Original Pattern (Duplicated Code)

The original benchmarks had duplicated code across multiple files for:
- Argument parsing
- Dtype conversion
- Warmup and timing loops
- Statistics computation
- Result printing

Here's a typical example from `examples/00_load/load_bench.py`:

```python
import argparse
import iris
import torch

def torch_dtype_from_str(datatype: str) -> torch.dtype:
    """Duplicated in many files"""
    dtype_map = {
        "int8": torch.int8,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    try:
        return dtype_map[datatype]
    except KeyError:
        print(f"Unknown datatype: {datatype}")
        exit(1)

def parse_args():
    """Duplicated argument parsing logic"""
    parser = argparse.ArgumentParser(
        description="Parse Message Passing configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-t", "--datatype", type=str, default="fp16", 
                       choices=["int8", "fp16", "bf16", "fp32"])
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-d", "--validate", action="store_true")
    parser.add_argument("-n", "--num_experiments", type=int, default=10)
    parser.add_argument("-w", "--num_warmup", type=int, default=1)
    # ... more arguments
    return vars(parser.parse_args())

def bench_load(shmem, source_rank, dest_rank, source_buffer, result_buffer,
               BLOCK_SIZE, dtype, verbose=False, validate=False,
               num_experiments=1, num_warmup=0):
    """Manual warmup and timing"""
    cur_rank = shmem.get_rank()
    n_elements = source_buffer.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    def run_store():
        if cur_rank == source_rank:
            store_kernel[grid](result_buffer, n_elements, BLOCK_SIZE)
    
    def run_load():
        if cur_rank == source_rank:
            load_kernel[grid](source_buffer, result_buffer, n_elements,
                            source_rank, dest_rank, BLOCK_SIZE,
                            shmem.get_heap_bases())
    
    # Manual warmup and timing
    store_ms = iris.do_bench(run_store, shmem.barrier, 
                            n_repeat=num_experiments, 
                            n_warmup=num_warmup)
    get_ms = iris.do_bench(run_load, shmem.barrier, 
                          n_repeat=num_experiments, 
                          n_warmup=num_warmup)
    
    # Manual statistics computation
    triton_ms = get_ms - store_ms
    
    # Manual bandwidth computation
    bandwidth_gbps = 0
    if cur_rank == source_rank:
        triton_sec = triton_ms * 1e-3
        element_size_bytes = torch.tensor([], dtype=dtype).element_size()
        total_bytes = n_elements * element_size_bytes
        bandwidth_gbps = total_bytes / triton_sec / 2**30
        
        # Manual verbose printing
        if verbose:
            shmem.info(f"Copied {total_bytes / 2**30:.2f} GiB in {triton_sec:.4f} seconds")
            shmem.info(f"Bandwidth is {bandwidth_gbps:.4f} GiB/s")
    
    # Manual synchronization
    shmem.barrier()
    bandwidth_gbps = shmem.broadcast(bandwidth_gbps, source_rank)
    
    # Manual validation (another ~50 lines)
    # ...
    
    return bandwidth_gbps
```

**Issues with this approach:**
- ~100 lines of boilerplate per benchmark
- `torch_dtype_from_str()` duplicated in 10+ files
- Argument parsing logic duplicated in 20+ files
- No standardized statistics (p50, p99)
- No easy JSON export for CI integration
- Manual bandwidth calculation repeated everywhere

## After: Using iris.bench

The new approach eliminates duplication and provides a clean, reusable interface:

```python
import iris
from iris.bench import BenchmarkRunner, torch_dtype_from_str, compute_bandwidth_gbps

def bench_load_refactored(shmem, source_rank, dest_rank, source_buffer, 
                         result_buffer, BLOCK_SIZE, dtype, 
                         warmup=5, iters=50):
    """Clean benchmark using iris.bench"""
    cur_rank = shmem.get_rank()
    n_elements = source_buffer.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    # Define operations
    def run_store():
        if cur_rank == source_rank:
            store_kernel[grid](result_buffer, n_elements, BLOCK_SIZE)
    
    def run_load():
        if cur_rank == source_rank:
            load_kernel[grid](source_buffer, result_buffer, n_elements,
                            source_rank, dest_rank, BLOCK_SIZE,
                            shmem.get_heap_bases())
    
    # Benchmark with automatic warmup, timing, and statistics
    runner = BenchmarkRunner(name="load_operation", barrier_fn=shmem.barrier)
    
    store_result = runner.run(fn=run_store, warmup=warmup, iters=iters,
                             params={"operation": "store"})
    load_result = runner.run(fn=run_load, warmup=warmup, iters=iters,
                            params={"operation": "load"})
    
    # Compute net time (automatic statistics available)
    net_ms = load_result.mean_ms - store_result.mean_ms
    
    # Compute bandwidth using helper function
    bandwidth_gbps = 0
    if cur_rank == source_rank:
        element_size_bytes = torch.tensor([], dtype=dtype).element_size()
        total_bytes = n_elements * element_size_bytes
        bandwidth_gbps = compute_bandwidth_gbps(total_bytes, net_ms)
        
        # Print structured results
        load_result.print_summary()
        print(f"Bandwidth: {bandwidth_gbps:.4f} GiB/s")
    
    shmem.barrier()
    bandwidth_gbps = shmem.broadcast(bandwidth_gbps, source_rank)
    
    return bandwidth_gbps, runner.get_results()
```

**Benefits:**
- ~50% less code (~50 lines vs ~100 lines)
- No duplicated utility functions (use `iris.bench.torch_dtype_from_str`)
- Automatic statistics: mean, median, p50, p99, min, max
- Structured results with `BenchmarkResult` objects
- Easy JSON export: `runner.save_json("results.json")`
- Consistent API across all benchmarks
- Built-in parameter tracking

## Complete Example: Parameter Sweep

Here's how to do a complete parameter sweep with the new harness:

```python
import iris
from iris.bench import BenchmarkRunner, torch_dtype_from_str

def benchmark_all_configs(shmem, source_buffer, result_buffer):
    """Benchmark across multiple configurations"""
    runner = BenchmarkRunner(name="load_sweep", barrier_fn=shmem.barrier)
    
    # Parameter sweep
    dtypes = ["fp16", "fp32"]
    block_sizes = [256, 512, 1024]
    
    for dtype_str in dtypes:
        dtype = torch_dtype_from_str(dtype_str)
        
        for block_size in block_sizes:
            def operation():
                # Your kernel launch
                load_kernel[grid](source_buffer, result_buffer, 
                                n_elements, source_rank, dest_rank,
                                block_size, shmem.get_heap_bases())
            
            runner.run(
                fn=operation,
                warmup=5,
                iters=50,
                params={
                    "dtype": dtype_str,
                    "block_size": block_size,
                }
            )
    
    # Print summary and export
    runner.print_summary()
    runner.save_json("sweep_results.json")
    
    return runner.get_results()
```

## Code Size Comparison

| File | Before (lines) | After (lines) | Reduction |
|------|----------------|---------------|-----------|
| Argument parsing | 25-40 | 0 (use standard args) | 100% |
| Dtype conversion | 15 | 1 (import) | 93% |
| Warmup/timing | 10-15 | 3 | 70-80% |
| Statistics | 5-10 (mean only) | 0 (automatic) | 100% |
| Bandwidth calc | 5 | 1 (helper fn) | 80% |
| Result printing | 20-50 | 1 (print_summary) | 95-98% |
| **Total** | **~100-150** | **~50-70** | **~50-60%** |

## Migration Strategy

1. **Start with new benchmarks**: Use `iris.bench` for all new benchmarks
2. **Gradual migration**: Refactor existing benchmarks incrementally
3. **Backward compatibility**: Old benchmarks continue to work
4. **CI integration**: Use JSON export for automated performance tracking

## Next Steps

- See `examples/benchmark/bench_harness_example.py` for complete working examples
- See `docs/bench_harness.md` for full API documentation
- Run tests: `pytest tests/unittests/test_bench.py`
