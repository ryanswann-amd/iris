#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Example demonstrating the unified benchmarking harness (iris.bench).

This example shows different ways to use the benchmarking infrastructure:
1. Using the @benchmark decorator
2. Using BenchmarkRunner directly
3. Using BenchmarkRunner for parameter sweeps
4. Saving results to JSON
"""

import torch
import iris
from iris.bench import benchmark, BenchmarkRunner, torch_dtype_from_str, compute_bandwidth_gbps


# Example 1: Using the @benchmark decorator
@benchmark(name="simple_operation", warmup=2, iters=5, auto_print=True)
def benchmark_simple_operation():
    """Simple benchmark using decorator."""
    tensor = torch.zeros(1024, 1024, dtype=torch.float32, device="cuda")
    result = tensor + 1.0
    return result


# Example 2: Using BenchmarkRunner directly
def benchmark_with_runner():
    """Benchmark using BenchmarkRunner."""

    def operation():
        tensor = torch.zeros(2048, 2048, dtype=torch.float16, device="cuda")
        result = tensor * 2.0
        return result

    runner = BenchmarkRunner(name="direct_runner_example")
    result = runner.run(fn=operation, warmup=2, iters=5)
    result.print_summary()


# Example 3: Parameter sweep
def benchmark_parameter_sweep():
    """Benchmark with parameter sweep."""
    runner = BenchmarkRunner(name="parameter_sweep")

    sizes = [512, 1024, 2048]
    dtypes = ["fp16", "fp32"]

    for size in sizes:
        for dtype_str in dtypes:
            dtype = torch_dtype_from_str(dtype_str)

            def operation(s=size, d=dtype):
                tensor = torch.zeros(s, s, dtype=d, device="cuda")
                result = tensor + 1.0
                return result

            runner.run(
                fn=operation,
                warmup=2,
                iters=5,
                params={"size": size, "dtype": dtype_str},
            )

    # Print summary and save to JSON
    runner.print_summary()
    runner.save_json("benchmark_results.json", include_raw_times=False)
    print(f"\nResults saved to benchmark_results.json")


# Example 4: Bandwidth calculation
def benchmark_with_bandwidth():
    """Benchmark with bandwidth calculation."""
    size = 1024 * 1024 * 256  # 256M elements
    dtype = torch.float16
    element_size = torch.tensor([], dtype=dtype).element_size()

    def operation():
        tensor = torch.zeros(size, dtype=dtype, device="cuda")
        result = tensor + 1.0
        return result

    runner = BenchmarkRunner(name="bandwidth_example")
    result = runner.run(fn=operation, warmup=2, iters=5)

    # Compute bandwidth
    total_bytes = size * element_size
    bandwidth = compute_bandwidth_gbps(total_bytes, result.mean_ms)

    print(f"\nBandwidth Calculation:")
    print(f"Size: {size} elements ({total_bytes / 2**30:.2f} GiB)")
    print(f"Mean time: {result.mean_ms:.4f} ms")
    print(f"Bandwidth: {bandwidth:.2f} GiB/s")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a CUDA-enabled GPU.")
        exit(1)

    print("=" * 70)
    print("Iris Benchmarking Harness Examples")
    print("=" * 70)

    print("\n### Example 1: Using @benchmark decorator ###")
    result1 = benchmark_simple_operation()

    print("\n### Example 2: Using BenchmarkRunner directly ###")
    benchmark_with_runner()

    print("\n### Example 3: Parameter sweep ###")
    benchmark_parameter_sweep()

    print("\n### Example 4: Bandwidth calculation ###")
    benchmark_with_bandwidth()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
