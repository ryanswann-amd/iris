#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import json
import tempfile
import os

import iris.bench as bench


def test_benchmark_result_creation():
    """Test creating a BenchmarkResult object."""
    result = bench.BenchmarkResult(
        name="test_benchmark",
        mean_ms=10.5,
        median_ms=10.2,
        p50_ms=10.2,
        p99_ms=15.3,
        min_ms=8.1,
        max_ms=16.2,
        n_warmup=5,
        n_repeat=50,
        params={"size": 1024},
        metadata={"gpu": "MI300X"},
        raw_times=[10.1, 10.2, 10.3],
    )

    assert result.name == "test_benchmark"
    assert result.mean_ms == 10.5
    assert result.median_ms == 10.2
    assert result.p50_ms == 10.2
    assert result.p99_ms == 15.3
    assert result.min_ms == 8.1
    assert result.max_ms == 16.2
    assert result.n_warmup == 5
    assert result.n_repeat == 50
    assert result.params == {"size": 1024}
    assert result.metadata == {"gpu": "MI300X"}
    assert result.raw_times == [10.1, 10.2, 10.3]


def test_benchmark_result_to_dict():
    """Test converting BenchmarkResult to dictionary."""
    result = bench.BenchmarkResult(
        name="test",
        mean_ms=10.0,
        median_ms=10.0,
        p50_ms=10.0,
        p99_ms=12.0,
        min_ms=9.0,
        max_ms=13.0,
        n_warmup=5,
        n_repeat=10,
        raw_times=[9.0, 10.0, 11.0, 12.0, 13.0],
    )

    # Without raw times
    d = result.to_dict(include_raw_times=False)
    assert "raw_times" not in d
    assert d["name"] == "test"
    assert d["mean_ms"] == 10.0

    # With raw times
    d = result.to_dict(include_raw_times=True)
    assert "raw_times" in d
    assert d["raw_times"] == [9.0, 10.0, 11.0, 12.0, 13.0]


def test_benchmark_result_to_json():
    """Test converting BenchmarkResult to JSON."""
    result = bench.BenchmarkResult(
        name="test",
        mean_ms=10.0,
        median_ms=10.0,
        p50_ms=10.0,
        p99_ms=12.0,
        min_ms=9.0,
        max_ms=13.0,
        n_warmup=5,
        n_repeat=10,
    )

    json_str = result.to_json()
    parsed = json.loads(json_str)
    assert parsed["name"] == "test"
    assert parsed["mean_ms"] == 10.0


def test_benchmark_result_print_summary(capsys):
    """Test printing BenchmarkResult summary."""
    result = bench.BenchmarkResult(
        name="test",
        mean_ms=10.0,
        median_ms=10.0,
        p50_ms=10.0,
        p99_ms=12.0,
        min_ms=9.0,
        max_ms=13.0,
        n_warmup=5,
        n_repeat=10,
        params={"size": 1024},
    )

    result.print_summary()
    captured = capsys.readouterr()
    assert "Benchmark: test" in captured.out
    assert "Mean:" in captured.out
    assert "10.0000 ms" in captured.out
    assert "Parameters: {'size': 1024}" in captured.out


def test_compute_percentile():
    """Test percentile computation."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    p50 = bench._compute_percentile(values, 50)
    assert 5.0 <= p50 <= 6.0

    p99 = bench._compute_percentile(values, 99)
    assert p99 > 9.0

    # Edge cases
    assert bench._compute_percentile([], 50) == 0.0
    assert bench._compute_percentile([5.0], 50) == 5.0


def test_benchmark_runner_basic():
    """Test basic BenchmarkRunner usage."""
    counter = {"count": 0}

    def test_fn():
        counter["count"] += 1
        # Simulate some work
        torch.zeros(100, 100, device="cuda")

    runner = bench.BenchmarkRunner(name="test_runner")

    # Run benchmark
    result = runner.run(fn=test_fn, warmup=2, iters=5)

    assert result.name == "test_runner"
    assert result.n_warmup == 2
    assert result.n_repeat == 5
    assert len(result.raw_times) == 5
    # Check that function was called (warmup + iters times)
    assert counter["count"] >= 5


def test_benchmark_runner_context_manager():
    """Test BenchmarkRunner as context manager."""
    runner = bench.BenchmarkRunner(name="context_test")

    # Use as context manager - we can't easily benchmark inside the context
    # so we'll just test that it doesn't crash
    with runner.run(warmup=1, iters=2, params={"size": 1024}):
        pass  # In real usage, code would be here

    # No results should be added when no function is provided
    assert len(runner.get_results()) == 0


def test_benchmark_runner_multiple_runs():
    """Test running multiple benchmarks."""

    def test_fn(size):
        torch.zeros(size, size, device="cuda")

    runner = bench.BenchmarkRunner(name="multi_test")

    # Run multiple benchmarks
    for size in [100, 200]:
        runner.run(fn=lambda s=size: test_fn(s), warmup=1, iters=2, params={"size": size})

    results = runner.get_results()
    assert len(results) == 2
    assert results[0].params["size"] == 100
    assert results[1].params["size"] == 200


def test_benchmark_runner_save_json():
    """Test saving results to JSON."""

    def test_fn():
        torch.zeros(10, 10, device="cuda")

    runner = bench.BenchmarkRunner(name="json_test")
    runner.run(fn=test_fn, warmup=1, iters=2, params={"size": 10})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        filepath = f.name

    try:
        runner.save_json(filepath, include_raw_times=True)

        # Load and verify
        with open(filepath, "r") as f:
            data = json.load(f)

        assert data["benchmark_suite"] == "json_test"
        assert data["total_runs"] == 1
        assert len(data["results"]) == 1
        assert "raw_times" in data["results"][0]
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_benchmark_runner_print_summary(capsys):
    """Test printing benchmark summary."""

    def test_fn():
        torch.zeros(10, 10, device="cuda")

    runner = bench.BenchmarkRunner(name="summary_test")
    runner.run(fn=test_fn, warmup=1, iters=2)

    runner.print_summary()
    captured = capsys.readouterr()
    assert "Benchmark Suite: summary_test" in captured.out
    assert "Total Runs: 1" in captured.out


def test_benchmark_decorator():
    """Test benchmark decorator."""

    @bench.benchmark(name="decorator_test", warmup=1, iters=2, auto_print=False)
    def test_fn(size):
        return torch.zeros(size, size, device="cuda")

    result = test_fn(10)

    assert isinstance(result, bench.BenchmarkResult)
    assert result.name == "decorator_test"
    assert result.n_warmup == 1
    assert result.n_repeat == 2


def test_benchmark_decorator_with_barrier():
    """Test benchmark decorator with barrier function."""
    barrier_called = {"count": 0}

    def barrier_fn():
        barrier_called["count"] += 1

    @bench.benchmark(name="barrier_test", warmup=1, iters=2, barrier_fn=barrier_fn)
    def test_fn():
        torch.zeros(10, 10, device="cuda")

    result = test_fn()

    assert isinstance(result, bench.BenchmarkResult)
    # Barrier should be called multiple times during benchmarking
    assert barrier_called["count"] > 0


def test_torch_dtype_from_str():
    """Test torch_dtype_from_str utility."""
    assert bench.torch_dtype_from_str("int8") == torch.int8
    assert bench.torch_dtype_from_str("fp16") == torch.float16
    assert bench.torch_dtype_from_str("bf16") == torch.bfloat16
    assert bench.torch_dtype_from_str("fp32") == torch.float32

    with pytest.raises(ValueError, match="Unknown datatype"):
        bench.torch_dtype_from_str("invalid")


def test_compute_bandwidth_gbps():
    """Test bandwidth computation."""
    # 1 GiB in 1 second = 1 GiB/s
    bandwidth = bench.compute_bandwidth_gbps(2**30, 1000)
    assert abs(bandwidth - 1.0) < 0.001

    # 2 GiB in 0.5 seconds = 4 GiB/s
    bandwidth = bench.compute_bandwidth_gbps(2 * 2**30, 500)
    assert abs(bandwidth - 4.0) < 0.001

    # 512 MiB in 100ms = 5 GiB/s
    bandwidth = bench.compute_bandwidth_gbps(512 * 2**20, 100)
    assert abs(bandwidth - 5.0) < 0.01


def test_benchmark_runner_with_barrier():
    """Test BenchmarkRunner with barrier function."""
    barrier_called = {"count": 0}

    def barrier_fn():
        barrier_called["count"] += 1

    def test_fn():
        torch.zeros(10, 10, device="cuda")

    runner = bench.BenchmarkRunner(name="barrier_runner", barrier_fn=barrier_fn)
    runner.run(fn=test_fn, warmup=1, iters=2)

    # Barrier should be called during benchmarking
    assert barrier_called["count"] > 0


def test_empty_benchmark():
    """Test benchmarking an empty function."""

    def empty_fn():
        pass

    runner = bench.BenchmarkRunner(name="empty_test")
    result = runner.run(fn=empty_fn, warmup=1, iters=5)

    assert result is not None
    assert len(result.raw_times) == 5
    # All times should be very small (likely close to 0)
    assert all(t >= 0 for t in result.raw_times)
