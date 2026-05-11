#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for iris-ccl two-shot all-reduce across the multi-channel knob.

Sweeps ``Config.all_reduce_num_channels`` (NCH ∈ {1, 2, 4, 8, 16}) at the
K-228 message-size buckets (16 / 64 / 256 / 1024 MB per rank, fp16) on
8 ranks. Per-link GB/s = ``size_bytes / (W * t_median)`` matches the K-228
methodology so results land directly on the K-228 efficiency table.

Empirical pilot (K-377, 8× MI300X, c42, 2026-05-11): the predicted
+1.5–2× speedup at ≥64 MB does not materialize. Best Δ vs NCH=1 is +5 %;
some sizes regress. See `output/results_summary.md` for analysis. The knob
is retained as a research tunable; the default (NCH=1) is unchanged.
"""

import torch

import iris.bench as bench
from iris.ccl import Config


# Equivalent of K-228's row sizes: (M, N) where M*N*2 bytes ~= size
# We fix N=8192 so M scales linearly with size_mb.
_K228_SIZES_MB = [16, 64, 256, 1024]
_NUM_CHANNELS = [1, 2, 4, 8]


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
    N = 8192
    M = numel // N
    if M * N != numel:
        raise ValueError(f"size {size_mb} MiB does not fit M*{N} elements")

    inp = ctx.zeros((M, N), dtype=dtype)
    out = ctx.zeros((M, N), dtype=dtype)
    inp.fill_(float(ctx.get_rank() + 1))

    # Bus bandwidth bytes (NCCL convention): 2 * (W-1)/W * data_size
    state.set_bytes(int(M * N * inp.element_size() * 2 * (world_size - 1) / world_size))

    config = Config(all_reduce_variant="two_shot", all_reduce_num_channels=nch)
    workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)

    state.exec(
        lambda: ctx.ccl.all_reduce(out, inp, config=config, workspace=workspace),
        preamble_fn=lambda: out.zero_(),
    )


if __name__ == "__main__":
    bench.main()
