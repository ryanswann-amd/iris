#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for all-gather + GEMM: RCCL baseline vs iris one_shot vs iris prefetch."""

import torch
import torch.distributed as dist
import iris.bench as bench
from iris.ops import FusedConfig, all_gather_matmul_preamble
from iris.ops.all_gather_matmul_hbm_buffer import (
    all_gather_matmul_hbm_buffer as _hbm_buffer,
    all_gather_matmul_hbm_buffer_preamble,
)


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def rccl_all_gather_matmul(state, ctx):
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    world_size = dist.get_world_size()
    K_local = K // world_size

    A_sharded = torch.ones((M, K_local), device="cuda", dtype=dtype)
    B = torch.randn((K, N), device="cuda", dtype=dtype)
    A_gathered = torch.empty((M, K), device="cuda", dtype=dtype)
    C = torch.empty((M, N), device="cuda", dtype=dtype)

    state.set_flops(2 * M * N * K)
    state.set_bytes((world_size - 1) * M * K_local * A_sharded.element_size())

    state.exec(
        lambda: (
            dist.all_gather_into_tensor(A_gathered, A_sharded),
            torch.mm(A_gathered, B, out=C),
        ),
    )


@bench.register
@bench.axis("num_ranks", [2, 4, 8])
@bench.axis("algorithm", ["one_shot", "prefetch"])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def all_gather_matmul(state, ctx):
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    algorithm = state["algorithm"]
    world_size = ctx.get_num_ranks()
    K_local = K // world_size

    A_sharded = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded.fill_(1.0)
    B = torch.randn((K, N), device="cuda", dtype=dtype)

    config = FusedConfig()

    state.set_flops(2 * M * N * K)
    state.set_bytes((world_size - 1) * M * K_local * A_sharded.element_size())

    if algorithm == "one_shot":
        C = torch.zeros((M, N), device="cuda", dtype=dtype)
        workspace = all_gather_matmul_preamble(ctx, A_sharded, B, config)
        state.exec(
            lambda: ctx.ops.all_gather_matmul(C, A_sharded, B, config=config, workspace=workspace),
        )
    else:  # prefetch
        C = ctx.zeros((M, N), dtype=dtype)
        workspace = all_gather_matmul_hbm_buffer_preamble(ctx, A_sharded, B, config)
        state.exec(
            lambda: _hbm_buffer(ctx, C, A_sharded, B, config=config, workspace=workspace),
            preamble_fn=lambda: C.zero_(),
        )


if __name__ == "__main__":
    bench.main()
