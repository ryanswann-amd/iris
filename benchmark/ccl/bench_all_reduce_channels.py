#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for iris-ccl two-shot all-reduce across the multi-channel knob.

Sweeps ``Config.all_reduce_num_channels`` (NCH ∈ {1, 2, 4, 8}) at the K-228
message-size buckets (16 / 64 / 256 / 1024 MB per rank, fp16) on 8 ranks.
Per-link GB/s = ``size_bytes / (W * t_median)`` matches the K-228 methodology
so results land directly on the K-228 efficiency table.

Empirical pilot (K-377, 8× MI300X, c42, 2026-05-11): the predicted
+1.5–2× speedup at ≥64 MB does not materialize. Best Δ vs NCH=1 is +5 %;
some sizes regress. See ``output/results_summary.md`` for analysis. The knob
is retained as a research tunable; the default (NCH=1) is unchanged.

Allocator hygiene
-----------------
The ``iris.bench`` framework calls our setup body once per ``(size_mb,
num_channels)`` combination and then hands the registered ``state.exec``
callable to ``iris.do_bench`` for warmup + timed iteration. Two pieces of
per-combination work used to leak into measurement:

1. A fresh ``ctx.zeros((M, N))`` pair was allocated for every cell, costing
   an allocator round-trip and a symmetric-heap barrier on each iteration of
   the outer ``num_channels`` sweep.
2. ``ctx.ccl.all_reduce_preamble(...)`` was invoked per cell as well, even
   though for ``two_shot`` the workspace state only depends on
   ``(shape, dtype, num_channels)``.

We now allocate the **largest** ``(M, N)`` buffers exactly once per
``(rank, dtype)`` (cached at module scope) and slice views per bucket; the
workspace is cached per ``(size_mb, dtype, num_channels)``. The framework's
``preamble_fn`` (``out.zero_()``) is the only per-iteration work and runs
outside the timed CUDA-event window.
"""

import torch

import iris.bench as bench
from iris.ccl import Config


# Equivalent of K-228's row sizes: (M, N) where M*N*2 bytes ~= size
# We fix N=8192 so M scales linearly with size_mb.
_K228_SIZES_MB = [16, 64, 256, 1024]
_NUM_CHANNELS = [1, 2, 4, 8]
_FIXED_N = 8192


# Module-level caches survive across (size_mb, num_channels) combinations
# inside a single rank-worker process — see _run_benchmarks_worker which
# iterates the Cartesian product in a single for-loop.
#
# _buffer_cache keys: (dtype). Holds the largest (M_max, N) input/output pair.
# _workspace_cache keys: (size_mb, dtype, num_channels). One workspace per
# distinct kernel configuration.
_buffer_cache: dict = {}
_workspace_cache: dict = {}


def _get_buffers(ctx, dtype):
    """Return (inp_full, out_full) at the maximum size, allocated once."""
    key = dtype
    if key not in _buffer_cache:
        elem = torch.tensor([], dtype=dtype).element_size()
        max_bytes = max(_K228_SIZES_MB) * (1 << 20)
        max_numel = max_bytes // elem
        max_M = max_numel // _FIXED_N
        inp_full = ctx.zeros((max_M, _FIXED_N), dtype=dtype)
        out_full = ctx.zeros((max_M, _FIXED_N), dtype=dtype)
        _buffer_cache[key] = (inp_full, out_full)
    return _buffer_cache[key]


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("size_mb", _K228_SIZES_MB)
@bench.axis("num_channels", _NUM_CHANNELS)
@bench.axis("dtype", [torch.float16])
def all_reduce_channels(state, ctx):
    size_mb = state["size_mb"]
    nch = state["num_channels"]
    dtype = state["dtype"]
    world_size = ctx.get_num_ranks()

    elem = torch.tensor([], dtype=dtype).element_size()
    bytes_per_rank = size_mb * (1 << 20)
    numel = bytes_per_rank // elem
    M = numel // _FIXED_N
    if M * _FIXED_N != numel:
        raise ValueError(f"size {size_mb} MiB does not fit M*{_FIXED_N} elements")

    # Reuse the largest (M_max, N) pair allocated on the symmetric heap and
    # take contiguous prefix views. Each rank's heap layout is identical
    # (allocations are collective), so the slice base address lines up across
    # ranks — the iris kernel's per-rank pointer arithmetic remains valid.
    inp_full, out_full = _get_buffers(ctx, dtype)
    inp = inp_full[:M]
    out = out_full[:M]

    # Re-stamp inp with the rank value (cheap; not part of the timed region).
    inp.fill_(float(ctx.get_rank() + 1))

    # Bus bandwidth bytes (NCCL convention): 2 * (W-1)/W * data_size
    state.set_bytes(int(M * _FIXED_N * inp.element_size() * 2 * (world_size - 1) / world_size))

    config = Config(all_reduce_variant="two_shot", all_reduce_num_channels=nch)

    # Workspace is hoisted out of the per-iter loop and cached per
    # (size_mb, dtype, num_channels). For VARIANT_TWO_SHOT this is metadata
    # only (no allocations), but caching makes the no-allocation contract
    # explicit and avoids re-binding to the (inp, out) views every cell.
    ws_key = (size_mb, dtype, nch)
    workspace = _workspace_cache.get(ws_key)
    if workspace is None:
        workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)
        _workspace_cache[ws_key] = workspace

    # Only the kernel launch is inside the timed CUDA-event window. The
    # preamble (out.zero_()) is run by the framework BEFORE every iteration
    # but OUTSIDE the timed region — see iris.do_bench / iris.bench docs.
    state.exec(
        lambda: ctx.ccl.all_reduce(out, inp, config=config, workspace=workspace),
        preamble_fn=lambda: out.zero_(),
    )


if __name__ == "__main__":
    bench.main()
