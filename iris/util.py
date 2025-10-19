# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import statistics
import math
import triton
import triton.language as tl
import torch
from contextlib import nullcontext


def get_empty_cache_for_benchmark():
    cache_size = 256 * 1024 * 1024
    return torch.empty(int(cache_size // 4), dtype=torch.int, device="cuda")


def clear_cache(cache):
    cache.zero_()


def create_timing_event():
    return torch.cuda.Event(enable_timing=True)


def _quantile(a, q):
    n = len(a)
    a = sorted(a)

    def get_quantile(q):
        if not (0 <= q <= 1):
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = q * (n - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        t = point - lower
        return (1 - t) * a[lower] + t * a[upper]

    return [get_quantile(q) for q in q]


def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = _quantile(times, quantiles)
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times
    elif return_mode == "min":
        return min(times)
    elif return_mode == "max":
        return max(times)
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)


def do_bench(
    fn,
    barrier_fn=lambda: None,
    preamble_fn=lambda: None,
    n_warmup=25,
    n_repeat=100,
    quantiles=None,
    return_mode="mean",
):
    """
    Benchmark a function by timing its execution.

    Args:
        fn (callable): Function to benchmark.
        barrier_fn (callable, optional): Function to call for synchronization. Default: no-op.
        preamble_fn (callable, optional): Function to call before each execution. Default: no-op.
        n_warmup (int, optional): Number of warmup iterations. Default: 25.
        n_repeat (int, optional): Number of timing iterations. Default: 100.
        quantiles (list, optional): Quantiles to return instead of summary statistic. Default: None.
        return_mode (str, optional): Summary statistic to return ("mean", "min", "max", "median", "all"). Default: "mean".

    Returns:
        float or list: Timing result(s) in milliseconds.

    Example:
        >>> import iris
        >>> iris_ctx = iris.iris(1 << 20)
        >>> def test_fn():
        >>>     tensor = iris_ctx.zeros(1000, 1000)
        >>> time_ms = iris.do_bench(test_fn, barrier_fn=iris_ctx.barrier)
    """
    # Wait for anything that happened before
    barrier_fn()
    preamble_fn()
    fn()
    barrier_fn()
    # Wait for all GPUs to finish their work

    cache = get_empty_cache_for_benchmark()

    start_event = [create_timing_event() for i in range(n_repeat)]
    end_event = [create_timing_event() for i in range(n_repeat)]

    # Warm-up
    for _ in range(n_warmup):
        barrier_fn()  # Wait for all GPUs before we clear the cache
        preamble_fn()
        clear_cache(cache)
        barrier_fn()  # Wait for clearing the cache before launching any kernels
        fn()

    # Benchmark
    for i in range(n_repeat):
        barrier_fn()  # Wait for all GPUs before we clear the cache
        preamble_fn()
        clear_cache(cache)
        barrier_fn()  # Wait for clearing the cache before launching any kernels
        start_event[i].record()
        fn()
        end_event[i].record()

    barrier_fn()  # Record clocks barrier

    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return _summarize_statistics(times, quantiles, return_mode)


class profile:
    """
    Context manager for PyTorch profiling with automatic trace file generation.
    
    This is a convenient wrapper around torch.profiler.profile that simplifies
    profiling in distributed/multi-rank scenarios.
    
    Args:
        enabled (bool, optional): Whether to enable profiling. Default: False.
        rank (int, optional): Current rank for trace file naming. Default: None.
        trace_file (str, optional): Custom trace file name. If not provided, 
            generates a name based on rank. Default: None.
        activities (list, optional): List of profiler activities. Default: [CUDA, CPU].
        record_shapes (bool, optional): Whether to record tensor shapes. Default: True.
        with_stack (bool, optional): Whether to record stack traces. Default: True.
    
    Returns:
        Context manager that yields the profiler object (or None if disabled).
    
    Example:
        >>> import iris
        >>> shmem = iris.iris(1 << 30)
        >>> rank = shmem.get_rank()
        >>>
        >>> with iris.profile(enabled=True, rank=rank) as prof:
        >>>     # Your code to profile
        >>>     result = my_function()
        >>> # Trace file is automatically saved as iris_trace_rank{rank}.json.gz
        
        >>> # Or with custom trace file name
        >>> with iris.profile(enabled=True, trace_file="my_trace.json.gz"):
        >>>     result = my_function()
    """
    
    def __init__(
        self,
        enabled=False,
        rank=None,
        trace_file=None,
        activities=None,
        record_shapes=True,
        with_stack=True,
    ):
        self.enabled = enabled
        self.rank = rank
        self.trace_file = trace_file
        self.activities = activities
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self._profiler = None
        self._context = None
        
    def __enter__(self):
        if not self.enabled:
            self._context = nullcontext()
            return self._context.__enter__()
        
        # Set default activities if not provided
        if self.activities is None:
            self.activities = [
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.CPU,
            ]
        
        # Create profiler
        self._profiler = torch.profiler.profile(
            activities=self.activities,
            record_shapes=self.record_shapes,
            with_stack=self.with_stack,
        )
        
        return self._profiler.__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return self._context.__exit__(exc_type, exc_val, exc_tb)
        
        # Exit profiler context
        result = self._profiler.__exit__(exc_type, exc_val, exc_tb)
        
        # Generate trace file name if not provided
        if self.trace_file is None:
            if self.rank is not None:
                self.trace_file = f"iris_trace_rank{self.rank}.json"
            else:
                self.trace_file = "iris_trace.json"
        
        # Export trace
        self._profiler.export_chrome_trace(self.trace_file)
        
        return result
