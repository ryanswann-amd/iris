#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: iris.ccl.reduce_scatter

Each rank contributes an (M, N) tensor. The reduce-scatter collective reduces (sums)
the inputs from all ranks and partitions the result: each rank receives the reduced
values for its assigned tile partition only, with all other elements remaining zero.

Together, all ranks' outputs form a complete partition of the full reduced result.

Run with:
    torchrun --nproc_per_node=<num_gpus> --standalone example.py [--validate]
"""

import argparse
import os

import torch
import torch.distributed as dist

import iris


def parse_args():
    parser = argparse.ArgumentParser(
        description="CCL reduce-scatter example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=1024, help="Number of rows")
    parser.add_argument("-n", type=int, default=512, help="Number of columns")
    parser.add_argument("--heap_size", type=int, default=1 << 31, help="Iris heap size")
    parser.add_argument("--datatype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"], help="Data type")
    parser.add_argument("-v", "--validate", action="store_true", help="Validate output against reference")
    return vars(parser.parse_args())


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")

    ctx = iris.iris(heap_size=args["heap_size"])
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    dtype = dtype_map[args["datatype"]]
    M, N = args["m"], args["n"]

    # Each rank fills its input with (rank + 1)
    input_tensor = ctx.zeros((M, N), dtype=dtype)
    input_tensor.fill_(float(rank + 1))
    output_tensor = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()
    ctx.ccl.reduce_scatter(output_tensor, input_tensor)
    torch.cuda.synchronize()

    if rank == 0:
        ctx.info(f"reduce_scatter: world_size={world_size}, shape=({M},{N}), dtype={dtype}")

    if args["validate"]:
        # Each rank owns a partition of tiles. The value at each assigned tile is the
        # element-wise sum of all ranks' inputs: sum(r+1 for r in 0..world_size-1).
        # Tiles not assigned to this rank remain 0.
        # Summing the outputs across all ranks (all_reduce) fills every element with
        # the expected per-element sum, since the tile partition is complete.
        expected = float(world_size * (world_size + 1) // 2)

        aggregated = output_tensor.clone()
        dist.all_reduce(aggregated, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        assert torch.allclose(aggregated, torch.full_like(aggregated, expected), atol=0.5), (
            f"Rank {rank}: mismatch after aggregation. Got {aggregated[0, 0].item():.1f}, expected {expected:.1f}"
        )
        if rank == 0:
            ctx.info(f"Validation passed: aggregated[0,0] = {aggregated[0, 0].item():.1f}")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
