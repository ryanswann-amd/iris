#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for HBM-buffered all-gather + GEMM (iris.ops)."""

import torch
import iris.bench as bench
from iris.ops import FusedConfig
from iris.ops.all_gather_matmul_hbm_buffer import (
    all_gather_matmul_hbm_buffer as _all_gather_matmul_hbm_buffer,
    all_gather_matmul_hbm_buffer_preamble,
)


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def bench_all_gather_matmul_hbm_buffer(state, ctx):
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    world_size = ctx.get_num_ranks()
    K_local = K // world_size

    A_sharded = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded.fill_(1.0)
    B = torch.randn((K, N), device="cuda", dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)

    config = FusedConfig()
    workspace = all_gather_matmul_hbm_buffer_preamble(ctx, A_sharded, B, config)

    state.set_flops(2 * M * N * K)
    state.set_bytes((world_size - 1) * M * K_local * A_sharded.element_size())

    state.exec(
        lambda: _all_gather_matmul_hbm_buffer(ctx, C, A_sharded, B, config=config, workspace=workspace),
        preamble_fn=lambda: C.zero_(),
    )


if __name__ == "__main__":
    bench.main()
