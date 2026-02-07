# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Unified benchmarking harness for Iris.

This module provides a standardized infrastructure for benchmarking operations:
- Warmup and iteration handling
- Timing and synchronization
- Statistics computation (mean, p50, p99)
- Parameter sweeps
- Structured result output (JSON or dict)

Example usage:

    from iris.bench import benchmark

    @benchmark(name="my_kernel", warmup=5, iters=50)
    def run(size, dtype):
        # setup tensors
        # launch kernel
        kernel(...)

    # Or use BenchmarkRunner for parameter sweeps:
    runner = BenchmarkRunner(name="gemm_sweep")
    for size in [1024, 2048, 4096]:
        with runner.run(warmup=5, iters=50, params={"size": size}):
            kernel(...)
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
import functools
import torch

if TYPE_CHECKING:
    from .util import do_bench


def _compute_percentile(values: List[float], percentile: float) -> float:
    """Compute percentile from a list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (percentile / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_values) else f
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


@dataclass
class BenchmarkResult:
    """
    Stores results from a benchmark run.

    Attributes:
        name: Name of the benchmark
        mean_ms: Mean time in milliseconds
        median_ms: Median time in milliseconds
        p50_ms: 50th percentile time in milliseconds
        p99_ms: 99th percentile time in milliseconds
        min_ms: Minimum time in milliseconds
        max_ms: Maximum time in milliseconds
        n_warmup: Number of warmup iterations
        n_repeat: Number of timing iterations
        params: Additional parameters passed to the benchmark
        metadata: Additional metadata
        raw_times: Raw timing measurements in milliseconds
    """

    name: str
    mean_ms: float
    median_ms: float
    p50_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    n_warmup: int
    n_repeat: int
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_times: List[float] = field(default_factory=list)

    def to_dict(self, include_raw_times: bool = False) -> Dict[str, Any]:
        """
        Convert result to dictionary.

        Args:
            include_raw_times: Whether to include raw timing measurements

        Returns:
            Dictionary representation of the result
        """
        result = asdict(self)
        if not include_raw_times:
            result.pop("raw_times", None)
        return result

    def to_json(self, include_raw_times: bool = False, indent: int = 2) -> str:
        """
        Convert result to JSON string.

        Args:
            include_raw_times: Whether to include raw timing measurements
            indent: JSON indentation level

        Returns:
            JSON string representation of the result
        """
        return json.dumps(self.to_dict(include_raw_times=include_raw_times), indent=indent)

    def print_summary(self):
        """Print a human-readable summary of the benchmark result."""
        print(f"\n{'=' * 60}")
        print(f"Benchmark: {self.name}")
        if self.params:
            print(f"Parameters: {self.params}")
        print(f"{'-' * 60}")
        print(f"Mean:   {self.mean_ms:10.4f} ms")
        print(f"Median: {self.median_ms:10.4f} ms")
        print(f"P50:    {self.p50_ms:10.4f} ms")
        print(f"P99:    {self.p99_ms:10.4f} ms")
        print(f"Min:    {self.min_ms:10.4f} ms")
        print(f"Max:    {self.max_ms:10.4f} ms")
        print(f"{'-' * 60}")
        print(f"Warmup iterations: {self.n_warmup}")
        print(f"Timing iterations: {self.n_repeat}")
        if self.metadata:
            print(f"Metadata: {self.metadata}")
        print(f"{'=' * 60}\n")


class BenchmarkRunner:
    """
    Context manager and runner for benchmarks with parameter sweeps.

    Example:
        runner = BenchmarkRunner(name="my_benchmark")
        for size in [1024, 2048]:
            with runner.run(warmup=5, iters=50, params={"size": size}):
                kernel(...)

        # Get all results
        results = runner.get_results()
        runner.print_summary()
        runner.save_json("results.json")
    """

    def __init__(self, name: str, barrier_fn: Optional[Callable] = None):
        """
        Initialize benchmark runner.

        Args:
            name: Name of the benchmark suite
            barrier_fn: Optional barrier function for multi-GPU synchronization
        """
        self.name = name
        self.barrier_fn = barrier_fn if barrier_fn is not None else lambda: None
        self.results: List[BenchmarkResult] = []
        self._current_fn: Optional[Callable] = None
        self._current_params: Dict[str, Any] = {}
        self._current_warmup: int = 25
        self._current_iters: int = 100

    class _RunContext:
        """Context manager for a single benchmark run."""

        def __init__(
            self,
            runner: "BenchmarkRunner",
            fn: Optional[Callable],
            warmup: int,
            iters: int,
            params: Dict[str, Any],
        ):
            self.runner = runner
            self.fn = fn
            self.warmup = warmup
            self.iters = iters
            self.params = params
            self._start_time = None

        def __enter__(self):
            self._start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                # Exception occurred, don't run benchmark
                return False

            if self.fn is not None:
                # Function was provided, benchmark it
                result = self.runner._run_benchmark(
                    self.fn,
                    warmup=self.warmup,
                    iters=self.iters,
                    params=self.params,
                )
                self.runner.results.append(result)

    def run(
        self,
        fn: Optional[Callable] = None,
        warmup: int = 25,
        iters: int = 100,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Run a benchmark (can be used as context manager or direct call).

        Args:
            fn: Function to benchmark (optional if using as context manager)
            warmup: Number of warmup iterations
            iters: Number of timing iterations
            params: Additional parameters to store with the result

        Returns:
            Context manager or BenchmarkResult
        """
        params = params or {}

        if fn is None:
            # Used as context manager
            return self._RunContext(self, None, warmup, iters, params)
        else:
            # Direct function call
            result = self._run_benchmark(fn, warmup=warmup, iters=iters, params=params)
            self.results.append(result)
            return result

    def _run_benchmark(
        self,
        fn: Callable,
        warmup: int,
        iters: int,
        params: Dict[str, Any],
    ) -> BenchmarkResult:
        """Internal method to run a benchmark and compute statistics."""
        # Import do_bench at runtime to avoid circular dependencies
        from .util import do_bench

        # Use iris.do_bench to get all timing measurements
        raw_times = do_bench(
            fn,
            barrier_fn=self.barrier_fn,
            n_warmup=warmup,
            n_repeat=iters,
            return_mode="all",
        )

        # Compute statistics
        mean_ms = sum(raw_times) / len(raw_times) if raw_times else 0.0
        median_ms = _compute_percentile(raw_times, 50)
        p50_ms = median_ms  # P50 is the same as median
        p99_ms = _compute_percentile(raw_times, 99)
        min_ms = min(raw_times) if raw_times else 0.0
        max_ms = max(raw_times) if raw_times else 0.0

        return BenchmarkResult(
            name=self.name,
            mean_ms=mean_ms,
            median_ms=median_ms,
            p50_ms=p50_ms,
            p99_ms=p99_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            n_warmup=warmup,
            n_repeat=iters,
            params=params,
            raw_times=raw_times,
        )

    def get_results(self) -> List[BenchmarkResult]:
        """Get all benchmark results."""
        return self.results

    def print_summary(self):
        """Print summary of all benchmark results."""
        print(f"\n{'=' * 70}")
        print(f"Benchmark Suite: {self.name}")
        print(f"Total Runs: {len(self.results)}")
        print(f"{'=' * 70}\n")

        for i, result in enumerate(self.results, 1):
            print(f"Run #{i}:")
            result.print_summary()

    def save_json(self, filepath: str, include_raw_times: bool = False):
        """
        Save all results to JSON file.

        Args:
            filepath: Path to output file
            include_raw_times: Whether to include raw timing measurements
        """
        output = {
            "benchmark_suite": self.name,
            "total_runs": len(self.results),
            "results": [r.to_dict(include_raw_times=include_raw_times) for r in self.results],
        }
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)


def benchmark(
    name: str,
    warmup: int = 25,
    iters: int = 100,
    barrier_fn: Optional[Callable] = None,
    auto_print: bool = False,
    params: Optional[Dict[str, Any]] = None,
):
    """
    Decorator for benchmarking functions.

    Args:
        name: Name of the benchmark
        warmup: Number of warmup iterations
        iters: Number of timing iterations
        barrier_fn: Optional barrier function for multi-GPU synchronization
        auto_print: Whether to automatically print results
        params: Additional parameters to store with the result

    Returns:
        Decorated function that returns BenchmarkResult

    Example:
        @benchmark(name="my_kernel", warmup=5, iters=50)
        def run_kernel(size):
            kernel[grid](buffer, size)

        result = run_kernel(1024)
        result.print_summary()
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract function parameters for metadata
            func_params = params.copy() if params else {}

            # Create runner
            runner = BenchmarkRunner(name=name, barrier_fn=barrier_fn)

            # Run benchmark
            result = runner.run(
                fn=lambda: func(*args, **kwargs),
                warmup=warmup,
                iters=iters,
                params=func_params,
            )

            if auto_print:
                result.print_summary()

            return result

        return wrapper

    return decorator


# Utility functions for common patterns


def torch_dtype_from_str(datatype: str) -> torch.dtype:
    """
    Convert string datatype to torch.dtype.

    Args:
        datatype: String representation of datatype

    Returns:
        torch.dtype object

    Raises:
        ValueError: If datatype is not recognized
    """
    dtype_map = {
        "int8": torch.int8,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if datatype not in dtype_map:
        raise ValueError(f"Unknown datatype: {datatype}. Expected one of {list(dtype_map.keys())}")
    return dtype_map[datatype]


def compute_bandwidth_gbps(
    total_bytes: int,
    time_ms: float,
) -> float:
    """
    Compute bandwidth in GiB/s.

    Args:
        total_bytes: Total number of bytes transferred
        time_ms: Time in milliseconds

    Returns:
        Bandwidth in GiB/s
    """
    time_sec = time_ms * 1e-3
    return total_bytes / time_sec / (2**30)


__all__ = [
    "BenchmarkResult",
    "BenchmarkRunner",
    "benchmark",
    "torch_dtype_from_str",
    "compute_bandwidth_gbps",
]
