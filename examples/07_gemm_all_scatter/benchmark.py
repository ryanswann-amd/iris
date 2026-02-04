#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import hip
hip.hip.hipInit(0)

import argparse
import os
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
from matmul_wrapper import matmul

import iris
from examples.common.utils import JSONWriter, Timestamps, is_triton_interpret_set
from examples.common.validation import validate_gemm

torch.manual_seed(123)
random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse matrix dimensions and configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=8192, help="Number of rows in matrix A")
    parser.add_argument("-n", type=int, default=4608, help="Number of columns in matrix B")
    parser.add_argument("-k", type=int, default=36864, help="Common dimension between matrices A and B")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-v", "--validate", action="store_true", help="Enable validation mode")
    parser.add_argument("-t", "--trace_tiles", action="store_true", help="Enable tile-tracing mode")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Enable benchmarking mode")
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Datatype of computation",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="log.json",
        help="Output file",
    )
    parser.add_argument("--BLK_M", type=int, default=256, help="Block size M")
    parser.add_argument("--BLK_N", type=int, default=64, help="Block size N")
    parser.add_argument("--BLK_K", type=int, default=64, help="Block size K")
    parser.add_argument("--gsize_m", type=int, default=6, help="L2-cache locality swizzle parameter")
    parser.add_argument("--num_stages", type=int, default=2, help="Number of stages")
    parser.add_argument("--heap_size", type=int, default=1 << 33, help="Iris heap size")
    parser.add_argument(
        "--gemm_sms",
        type=int,
        default=256,
        help="Number of SMs for persistent GEMM algorithm (default: 256)",
    )
    parser.add_argument("--num_experiments", type=int, default=1, help="Number of experiment iterations")
    parser.add_argument("--num_warmup", type=int, default=0, help="Number of warmup iterations")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-r", "--num_ranks", type=int, default=2, help="Number of ranks/processes")

    return vars(parser.parse_args())


def _worker(local_rank: int = None, world_size: int = None, init_url: str = None, args: dict = None):
    """Worker function for PyTorch distributed execution."""
    # Support torchrun: read from environment variables if available
    if local_rank is None:
        local_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    if init_url is None:
        # torchrun sets MASTER_ADDR and MASTER_PORT
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29500")
        init_url = f"tcp://{master_addr}:{master_port}"
    
    # Use gloo backend for simulator compatibility (nccl requires GPU kernels)
    backend = "gloo"
    if args.get("verbose"):
        print(f"Using backend: {backend}")
    
    # Use environment-based initialization if torchrun is detected
    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        # For torchrun, init_process_group reads from environment
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        dist.init_process_group(
            backend=backend,
            init_method=init_url,
            world_size=world_size,
            rank=local_rank,
            device_id=torch.device(f"cuda:{local_rank}"),
        )

    # Main benchmark logic
    shmem = iris.iris(args["heap_size"])
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # gemm_sms is already set from command line args (default: 256)

    # GEMM
    datatype = torch.float32
    if args["datatype"] == "fp16":
        datatype = torch.float16
    elif args["datatype"] == "fp32":
        datatype = torch.float32
    elif args["datatype"] == "bf16":
        datatype = torch.bfloat16
    else:
        print("Unknown datatype.")
        exit(1)

    assert args["n"] % world_size == 0, f"N ({args['n']}) must be divisible by world size ({world_size})."
    assert args["k"] % world_size == 0, f"K ({args['k']}) must be divisible by world size ({world_size})."

    # Use ones() + zeros_like() pattern to work around simulator compatibility issues
    temp_A_list = shmem.ones(args["m"] * args["k"], device="cuda", dtype=datatype)
    temp_A = temp_A_list[0]
    A = shmem.zeros_like(temp_A).reshape(args["m"], args["k"])
    # Construct B already transposed (k x n) instead of transposing later
    temp_B_list = shmem.ones(args["k"] * args["n"], device="cuda", dtype=datatype)
    temp_B = temp_B_list[0]
    B = shmem.zeros_like(temp_B).reshape(args["k"], args["n"])

    args["M"] = args["m"]
    args["N"] = args["n"]
    args["K"] = args["k"]

    json_writer = JSONWriter(args["output_file"])
    json_writer.add_field("world_size", world_size)

    # Splitting
    args["n"] = args["n"] // world_size
    # Use slice directly without clone/contiguous to avoid GPU kernel calls
    local_B = B[:, rank * args["n"] : (rank + 1) * args["n"]]
    local_A = A

    for key, value in args.items():
        json_writer.add_field(key, value)

    # Use ones() + zeros_like() pattern for all allocations
    temp_global_C_list = shmem.ones(args["M"] * args["N"], device="cuda", dtype=A.dtype)
    temp_global_C = temp_global_C_list[0]
    global_C = shmem.zeros_like(temp_global_C).reshape(args["M"], args["N"])
    
    temp_local_C_list = shmem.ones(args["m"] * args["n"], device="cuda", dtype=A.dtype)
    temp_local_C = temp_local_C_list[0]
    local_C = shmem.zeros_like(temp_local_C).reshape(args["m"], args["n"])

    total_blocks_M = triton.cdiv(args["m"], args["BLK_M"])
    total_blocks_N = triton.cdiv(args["n"], args["BLK_N"])
    total_tiles = total_blocks_M * total_blocks_N

    bias = None

    gemm_stream = torch.cuda.Stream()

    json_writer.add_field("gemm_sms", args["gemm_sms"])

    kernel_timing = {
        "gemm": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
    }

    # Allocate Timestamps only if tracing is enabled
    timestamps = Timestamps(num_tiles=total_tiles) if args["trace_tiles"] else None

    def run_experiment():
        nonlocal local_C
        nonlocal global_C
        nonlocal kernel_timing

        shmem.barrier()

        if args["trace_tiles"] and timestamps is not None:
            timestamps.reset()
            shmem.barrier()

        torch.cuda.nvtx.range_push("GEMM + Communication")
        torch.cuda.nvtx.range_push("GEMM")
        with torch.cuda.stream(gemm_stream):
            kernel_timing["gemm"]["start_event"].record()
            local_C = matmul.apply(
                local_A,
                local_B,
                local_C,
                global_C,
                bias,
                rank,
                world_size,
                args["gemm_sms"],
                args["BLK_M"],
                args["BLK_N"],
                args["BLK_K"],
                args["gsize_m"],
                args["num_stages"],
                shmem.get_heap_bases(),
                "gfx942",
                args["trace_tiles"],
                timestamps.mm_begin_timestamp if timestamps else None,
                timestamps.mm_end_timestamp if timestamps else None,
            )
            kernel_timing["gemm"]["end_event"].record()
            kernel_timing["gemm"]["experiments"] += 1

        torch.cuda.nvtx.range_pop()
        shmem.barrier()

        for k in ["gemm"]:
            ms = kernel_timing[k]["start_event"].elapsed_time(kernel_timing[k]["end_event"])
            kernel_timing[k]["ms"] += ms

        torch.cuda.nvtx.range_pop()

    # Synchronize across all GPUs
    shmem.barrier()

    # Warmup
    if args["verbose"]:
        shmem.info(f"Running {args['num_warmup']} warmup iterations...")
    for _ in range(args["num_warmup"]):
        run_experiment()

    shmem.barrier()

    for k in ["gemm"]:
        kernel_timing[k]["ms"] = 0
        kernel_timing[k]["experiments"] = 0

    # Run experiments
    if args["verbose"]:
        shmem.info(f"Running {args['num_experiments']} experiments...")
    for _ in range(args["num_experiments"]):
        run_experiment()
        shmem.barrier()

    if args["validate"]:
        shmem.info("Validating...")
        matmul.set_debug(True)
        # Validate global result
        success = validate_gemm(A, B, global_C, shmem)
        passed_str = "passed" if success else "failed"
        shmem.info(f"Final C validation {passed_str}.")

        # Wait for all to finish validation
        shmem.barrier()
        shmem.info("Validating local C...")

        json_writer.add_field("success", success)

        if not is_triton_interpret_set():
            gemm_registers = matmul.get_matmul_registers()
            gemm_spills = matmul.get_matmul_spills()

            json_writer.add_field("gemm_registers", gemm_registers)
            json_writer.add_field("gemm_spills", gemm_spills)

        shmem.info("Validation completed")

    if args["benchmark"]:
        matmul.set_debug(False)
        shmem.info("Benchmarking...")
        perf = lambda ms: 2 * args["M"] * args["N"] * args["K"] * 1e-12 / (ms * 1e-3)
        triton_ms = iris.do_bench(run_experiment, shmem.barrier)
        triton_tflops = perf(triton_ms)
        algo_string = "all_scatter"
        shmem.info(
            f"tile matmul + {algo_string} (total_tiles={total_tiles}): {triton_ms:.3f} ms  {triton_tflops:.3f} tflops"
        )

        json_writer.add_field("tflops", triton_tflops)
        json_writer.add_field("total_ms", triton_ms)

        for k in ["gemm"]:
            json_writer.add_field(k + "_ms", kernel_timing[k]["ms"] / kernel_timing[k]["experiments"])
            json_writer.add_field(k + "_experiments", kernel_timing[k]["experiments"])

        # Wait for all to finish benchmarking
        shmem.barrier()

    if rank == 0:
        json_writer.flush()
        json_writer.display()

    if args["trace_tiles"] and rank == 0 and timestamps is not None:
        gpu_freq = iris.hip.get_wall_clock_rate(rank) * 1e-3
        algo_string = "all_scatter"
        filename = f"gemm_tiles_{algo_string}_trace_rank{rank}.json"
        timestamps.to_json(filename, gpu_freq)

    shmem.barrier()

    dist.barrier()
    dist.destroy_process_group()


def main():
    print("Starting GEMM all_scatter benchmark...")
    args = parse_args()

    # Check if running with torchrun (detected by environment variables)
    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        # torchrun handles process spawning, so call _worker directly
        print("Detected torchrun execution mode")
        _worker(args=args)
    else:
        # Use multiprocessing spawn for backward compatibility
        num_ranks = args["num_ranks"]
        init_url = "tcp://127.0.0.1:29500"
        mp.spawn(
            fn=_worker,
            args=(num_ranks, init_url, args),
            nprocs=num_ranks,
            join=True,
        )


if __name__ == "__main__":
    main()
