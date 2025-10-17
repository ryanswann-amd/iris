#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.profiler
import triton
import random
import sys
import os
import argparse
import json
import math
from contextlib import nullcontext

from examples.common.utils import (
    JSONWriter,
    Timestamps,
    is_triton_interpret_set,
)

import iris

from matmul_wrapper import matmul
from examples.common.validation import validate_gemm
from gemm_all_reduce_ring_based import persistent_all_reduce

torch.manual_seed(123)
random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse matrix dimensions and configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=8192, help="Number of rows in matrix A (GEMM)")
    parser.add_argument("-n", type=int, default=4608, help="Number of columns in matrix B (GEMM)")
    parser.add_argument("-k", type=int, default=36864, help="Common dimension between matrices A and B (GEMM)")
    parser.add_argument(
        "--m_comm", type=int, default=None, help="Number of rows for communication tensor (defaults to m)"
    )
    parser.add_argument(
        "--n_comm", type=int, default=None, help="Number of columns for communication tensor (defaults to n)"
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-v", "--validate", action="store_true", help="Enable validation mode")
    parser.add_argument("-t", "--trace_tiles", action="store_true", help="Enable tile-tracing mode")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Enable benchmarking mode")
    parser.add_argument("-p", "--profile", action="store_true", help="Enable PyTorch profiler to generate chrome trace")
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "int8", "bf16"],
        help="Datatype of computation",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="log.json",
        help="Output file",
    )
    # For All Scatter, use: 256x64x64
    # For One Shot, use: 256x256x64
    parser.add_argument("--BLK_M", type=int, default=128, help="Block size M")
    parser.add_argument("--BLK_N", type=int, default=128, help="Block size N")
    parser.add_argument("--BLK_K", type=int, default=64, help="Block size K")

    # Best to try 1, 6 or 8
    parser.add_argument("--gsize_m", type=int, default=6, help="Grid size M")
    parser.add_argument("--heap_size", type=int, default=1 << 33, help="Iris heap size")

    # For All Scatter, use: 288
    # For One Shot, use: 256
    parser.add_argument("--gemm_sms", type=int, default=256, help="Number of SMs for GEMM")
    parser.add_argument("--comm_sms", type=int, default=48, help="Number of SMs for All-Scatter kernel")
    parser.add_argument("-r", "--num_ranks", type=int, default=2, help="Number of ranks/processes")
    parser.add_argument(
        "--benchmark_mode",
        type=str,
        default="both",
        choices=["both", "gemm", "comm", "all"],
        help="Benchmark mode: 'both' for GEMM+Comm together, 'gemm' for GEMM only, 'comm' for communication only, 'all' for sequential GEMM->Comm->Overlap",
    )

    return vars(parser.parse_args())


def _worker(local_rank: int, world_size: int, init_url: str, args: dict):
    """Worker function for PyTorch distributed execution."""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
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
    cu_count = shmem.get_cu_count()
    num_xcds = iris.hip.get_num_xcc()

    # GEMM
    datatype = torch.float32
    if args["datatype"] == "fp16":
        datatype = torch.float16
    elif args["datatype"] == "fp32":
        datatype = torch.float32
    elif args["datatype"] == "int8":
        datatype = torch.int8
    elif args["datatype"] == "bf16":
        datatype = torch.bfloat16
    else:
        print("Unknown datatype.")
        exit(1)

    # Set default values for communication dimensions if not provided
    if args["m_comm"] is None:
        args["m_comm"] = args["m"]
    if args["n_comm"] is None:
        args["n_comm"] = args["n"]

    A = shmem.randn(args["m"], args["k"], device="cuda", dtype=datatype)
    B = shmem.randn(args["n"], args["k"], device="cuda", dtype=datatype).T

    json_writer = JSONWriter(args["output_file"])
    json_writer.add_field("world_size", world_size)

    local_B = B
    local_A = A

    for key, value in args.items():
        json_writer.add_field(key, value)

    # GEMM output tensors (using original m, n dimensions)
    C = shmem.zeros((args["m"], args["n"]), device="cuda", dtype=A.dtype)
    local_C = shmem.zeros((args["m"], args["n"]), device="cuda", dtype=torch.float32)

    # Communication tensors (using independent m_comm, n_comm dimensions)
    C_comm = shmem.zeros((args["m_comm"], args["n_comm"]), device="cuda", dtype=torch.float32)
    local_C_comm = shmem.zeros((args["m_comm"], args["n_comm"]), device="cuda", dtype=torch.float32)

    # Calculate tiles based on communication dimensions
    total_blocks_M = triton.cdiv(args["m_comm"], args["BLK_M"])
    total_blocks_N = triton.cdiv(args["n_comm"], args["BLK_N"])
    total_tiles = total_blocks_M * total_blocks_N

    flags = shmem.zeros((total_tiles,), device="cuda", dtype=torch.int32)
    ring_buffer = shmem.zeros((args["m_comm"], args["n_comm"]), device="cuda", dtype=torch.float32)

    bias = None
    ar = None  # Will hold the all_reduce kernel reference for resource inspection

    gemm_stream = torch.cuda.Stream()
    comm_stream = torch.cuda.Stream()

    json_writer.add_field("gemm_sms", args["gemm_sms"])
    json_writer.add_field("comm_sms", args["comm_sms"])

    kernel_timing = {
        "gemm": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
        "communication": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
    }

    # Timestamps
    timestamps = Timestamps(num_tiles=total_tiles)

    def preamble():
        shmem.barrier()
        flags.zero_()
        ring_buffer.zero_()
        local_C_comm.zero_()
        C_comm.zero_()
        shmem.barrier()

    def run_experiment():
        nonlocal local_C
        nonlocal C
        nonlocal C_comm
        nonlocal local_C_comm
        nonlocal kernel_timing
        nonlocal ring_buffer
        nonlocal ar

        shmem.barrier()

        if args["trace_tiles"]:
            timestamps.reset()
            shmem.barrier()

        torch.cuda.nvtx.range_push("GEMM + Communication")

        torch.cuda.nvtx.range_push("Communication")
        with torch.cuda.stream(comm_stream):
            kernel_timing["communication"]["start_event"].record(comm_stream)
            ar = persistent_all_reduce[(args["comm_sms"],)](
                C_comm,
                local_C_comm,
                ring_buffer,
                flags,
                args["m_comm"],
                args["n_comm"],
                C_comm.stride(0),
                C_comm.stride(1),
                args["BLK_M"],
                args["BLK_N"],
                args["gsize_m"],
                args["comm_sms"],
                num_xcds,
                shmem.get_heap_bases(),
                rank,
                world_size,
            )
            kernel_timing["communication"]["end_event"].record(comm_stream)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("GEMM")
        with torch.cuda.stream(gemm_stream):
            kernel_timing["gemm"]["start_event"].record(gemm_stream)
            local_C = matmul.apply(
                local_A,
                local_B,
                local_C,
                bias,
                rank,
                world_size,
                args["gemm_sms"],
                args["BLK_M"],
                args["BLK_N"],
                args["BLK_K"],
                args["gsize_m"],
                shmem.get_heap_bases(),
                "gfx942",
                args["trace_tiles"],
                timestamps.mm_begin_timestamp,
                timestamps.mm_end_timestamp,
            )
            kernel_timing["gemm"]["end_event"].record(gemm_stream)

        torch.cuda.nvtx.range_pop()

        # Synchronize events before calculating time
        torch.cuda.synchronize()
        for k in ["gemm", "communication"]:
            ms = kernel_timing[k]["start_event"].elapsed_time(kernel_timing[k]["end_event"])
            kernel_timing[k]["ms"] += ms
            kernel_timing[k]["experiments"] += 1

        shmem.barrier()

        torch.cuda.nvtx.range_pop()

    def run_gemm_only():
        nonlocal local_C
        nonlocal kernel_timing

        shmem.barrier()

        if args["trace_tiles"]:
            timestamps.reset()
            shmem.barrier()

        torch.cuda.nvtx.range_push("GEMM Only")
        kernel_timing["gemm"]["start_event"].record()
        local_C = matmul.apply(
            local_A,
            local_B,
            local_C,
            bias,
            rank,
            world_size,
            args["gemm_sms"],
            args["BLK_M"],
            args["BLK_N"],
            args["BLK_K"],
            args["gsize_m"],
            shmem.get_heap_bases(),
            "gfx942",
            args["trace_tiles"],
            timestamps.mm_begin_timestamp,
            timestamps.mm_end_timestamp,
        )
        kernel_timing["gemm"]["end_event"].record()
        torch.cuda.nvtx.range_pop()

        # Synchronize events before calculating time
        torch.cuda.synchronize()
        ms = kernel_timing["gemm"]["start_event"].elapsed_time(kernel_timing["gemm"]["end_event"])
        kernel_timing["gemm"]["ms"] += ms
        kernel_timing["gemm"]["experiments"] += 1

        shmem.barrier()

    def run_comm_only():
        nonlocal C_comm
        nonlocal local_C_comm
        nonlocal kernel_timing
        nonlocal ring_buffer
        nonlocal ar

        shmem.barrier()

        torch.cuda.nvtx.range_push("Communication Only")
        kernel_timing["communication"]["start_event"].record()
        ar = persistent_all_reduce[(args["comm_sms"],)](
            C_comm,
            local_C_comm,
            ring_buffer,
            flags,
            args["m_comm"],
            args["n_comm"],
            C_comm.stride(0),
            C_comm.stride(1),
            args["BLK_M"],
            args["BLK_N"],
            args["gsize_m"],
            args["comm_sms"],
            num_xcds,
            shmem.get_heap_bases(),
            rank,
            world_size,
        )
        kernel_timing["communication"]["end_event"].record()
        torch.cuda.nvtx.range_pop()

        # Synchronize events before calculating time
        torch.cuda.synchronize()
        ms = kernel_timing["communication"]["start_event"].elapsed_time(kernel_timing["communication"]["end_event"])
        kernel_timing["communication"]["ms"] += ms
        kernel_timing["communication"]["experiments"] += 1

        shmem.barrier()

    # Synchronize across all GPUs
    shmem.barrier()

    # Warmup
    run_experiment()

    shmem.barrier()
    preamble()
    shmem.barrier()

    for k in ["gemm", "communication"]:
        kernel_timing[k]["ms"] = 0
        kernel_timing[k]["experiments"] = 0

    # Get kernel resource usage after warmup (before set_debug(False) in benchmark mode)
    gemm_registers = None
    gemm_spills = None
    comm_registers = None
    comm_spills = None
    if not is_triton_interpret_set():
        gemm_registers = matmul.get_matmul_registers()
        gemm_spills = matmul.get_matmul_spills()

        # Get communication kernel resource usage
        comm_registers = ar.n_regs if ar is not None else None
        comm_spills = ar.n_spills if ar is not None else None

    # Start PyTorch profiler (if enabled)
    profiler_context = (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            with_stack=True,
        )
        if args["profile"]
        else nullcontext()
    )

    with profiler_context as prof:
        if args["validate"]:
            if rank == 0:
                shmem.info("Validating...")
            matmul.set_debug(True)
            # Validate global result
            success = validate_gemm(A, B, C, shmem, atol=2)
            passed_str = "passed" if success else "failed"
            if rank == 0:
                shmem.info(f"Final C validation {passed_str}.")

            # Wait for all to finish validation
            shmem.barrier()
            if rank == 0:
                shmem.info("Validation completed")

            json_writer.add_field("success", success)

        if args["benchmark"]:
            matmul.set_debug(False)
            if rank == 0:
                shmem.info("Benchmarking...")
            perf = lambda ms: 2 * args["m"] * args["n"] * args["k"] * 1e-12 / (ms * 1e-3)

            benchmark_mode = args["benchmark_mode"]

            if benchmark_mode == "all":
                # Run all three benchmarks sequentially: GEMM -> Comm -> Overlap
                if rank == 0:
                    shmem.info("=" * 60)
                    shmem.info("Running sequential benchmarks: GEMM -> Comm -> Overlap")
                    shmem.info("=" * 60)

                # 1. GEMM Only
                if rank == 0:
                    shmem.info("\n[1/3] Benchmarking GEMM only...")
                shmem.barrier()
                preamble()
                for k in ["gemm", "communication"]:
                    kernel_timing[k]["ms"] = 0
                    kernel_timing[k]["experiments"] = 0
                shmem.barrier()

                gemm_ms = iris.do_bench(run_gemm_only, shmem.barrier, preamble)
                gemm_tflops = perf(gemm_ms)
                if rank == 0:
                    shmem.info(f"  GEMM only: {gemm_ms:.3f} ms  {gemm_tflops:.3f} tflops")
                json_writer.add_field("gemm_only_ms", gemm_ms)
                json_writer.add_field("gemm_only_tflops", gemm_tflops)
                json_writer.add_field(
                    "gemm_only_measured_ms", kernel_timing["gemm"]["ms"] / kernel_timing["gemm"]["experiments"]
                )

                # 2. Communication Only
                if rank == 0:
                    shmem.info("\n[2/3] Benchmarking Communication only...")
                shmem.barrier()
                preamble()
                for k in ["gemm", "communication"]:
                    kernel_timing[k]["ms"] = 0
                    kernel_timing[k]["experiments"] = 0
                shmem.barrier()

                comm_ms = iris.do_bench(run_comm_only, shmem.barrier, preamble)
                algo_string = "all_reduce"
                if rank == 0:
                    shmem.info(f"  Communication only ({algo_string}): {comm_ms:.3f} ms")
                json_writer.add_field("comm_only_ms", comm_ms)
                json_writer.add_field(
                    "comm_only_measured_ms",
                    kernel_timing["communication"]["ms"] / kernel_timing["communication"]["experiments"],
                )

                # 3. Overlap (Both)
                if rank == 0:
                    shmem.info("\n[3/3] Benchmarking Overlap (GEMM + Comm)...")
                shmem.barrier()
                preamble()
                for k in ["gemm", "communication"]:
                    kernel_timing[k]["ms"] = 0
                    kernel_timing[k]["experiments"] = 0
                shmem.barrier()

                overlap_ms = iris.do_bench(run_experiment, shmem.barrier, preamble)
                overlap_tflops = perf(overlap_ms)
                if rank == 0:
                    shmem.info(f"  Overlap (GEMM + {algo_string}): {overlap_ms:.3f} ms  {overlap_tflops:.3f} tflops")
                json_writer.add_field("overlap_ms", overlap_ms)
                json_writer.add_field("overlap_tflops", overlap_tflops)
                json_writer.add_field(
                    "overlap_gemm_measured_ms", kernel_timing["gemm"]["ms"] / kernel_timing["gemm"]["experiments"]
                )
                json_writer.add_field(
                    "overlap_comm_measured_ms",
                    kernel_timing["communication"]["ms"] / kernel_timing["communication"]["experiments"],
                )

                # Summary
                if rank == 0:
                    shmem.info("\n" + "=" * 60)
                    shmem.info("Summary:")
                    shmem.info(f"  GEMM only:       {gemm_ms:.3f} ms ({gemm_tflops:.3f} tflops)")
                    shmem.info(f"  Comm only:       {comm_ms:.3f} ms")
                    shmem.info(f"  Overlap:         {overlap_ms:.3f} ms ({overlap_tflops:.3f} tflops)")
                    shmem.info(f"  GEMM + Comm sum: {gemm_ms + comm_ms:.3f} ms")
                    shmem.info(f"  Speedup:         {(gemm_ms + comm_ms) / overlap_ms:.2f}x")
                    shmem.info("=" * 60)

                json_writer.add_field("gemm_comm_sum_ms", gemm_ms + comm_ms)
                json_writer.add_field("speedup", (gemm_ms + comm_ms) / overlap_ms)

            elif benchmark_mode == "both":
                triton_ms = iris.do_bench(run_experiment, shmem.barrier, preamble)
                triton_tflops = perf(triton_ms)
                algo_string = "all_reduce"
                if rank == 0:
                    shmem.info(
                        f"tile matmul + {algo_string} (grid={total_tiles}): {triton_ms:.3f} ms  {triton_tflops:.3f} tflops"
                    )
                json_writer.add_field("tflops", triton_tflops)
                json_writer.add_field("total_ms", triton_ms)

                for k in ["gemm", "communication"]:
                    json_writer.add_field(k + "_ms", kernel_timing[k]["ms"] / kernel_timing[k]["experiments"])
                    json_writer.add_field(k + "_experiments", kernel_timing[k]["experiments"])

            elif benchmark_mode == "gemm":
                triton_ms = iris.do_bench(run_gemm_only, shmem.barrier, preamble)
                triton_tflops = perf(triton_ms)
                if rank == 0:
                    shmem.info(f"GEMM only (grid={total_tiles}): {triton_ms:.3f} ms  {triton_tflops:.3f} tflops")
                json_writer.add_field("tflops", triton_tflops)
                json_writer.add_field("total_ms", triton_ms)
                json_writer.add_field("gemm_ms", kernel_timing["gemm"]["ms"] / kernel_timing["gemm"]["experiments"])
                json_writer.add_field("gemm_experiments", kernel_timing["gemm"]["experiments"])

            elif benchmark_mode == "comm":
                triton_ms = iris.do_bench(run_comm_only, shmem.barrier, preamble)
                algo_string = "all_reduce"
                if rank == 0:
                    shmem.info(f"Communication only ({algo_string}, grid={total_tiles}): {triton_ms:.3f} ms")
                json_writer.add_field("total_ms", triton_ms)
                json_writer.add_field(
                    "communication_ms",
                    kernel_timing["communication"]["ms"] / kernel_timing["communication"]["experiments"],
                )
                json_writer.add_field("communication_experiments", kernel_timing["communication"]["experiments"])

            # Wait for all to finish benchmarking
            shmem.barrier()

    # Export profiler trace (if profiler was enabled)
    if args["profile"] and rank == 0:
        mode_suffix = args["benchmark_mode"]
        trace_file = f"iris_gemm_iris_allreduce_{mode_suffix}_trace_rank{rank}.json.gz"
        prof.export_chrome_trace(trace_file)
        shmem.info(f"Profiler trace saved to {trace_file}")

    # Record kernel resource usage (for both validation and benchmark modes)
    if not is_triton_interpret_set():
        json_writer.add_field("gemm_registers", gemm_registers)
        json_writer.add_field("gemm_spills", gemm_spills)
        json_writer.add_field("comm_registers", comm_registers)
        json_writer.add_field("comm_spills", comm_spills)

    if rank == 0:
        json_writer.flush()
        json_writer.display()

    if args["trace_tiles"] and rank == 0:
        gpu_freq = iris.hip.get_wall_clock_rate(rank) * 1e-3
        algo_string = "all_reduce"
        mode_suffix = args["benchmark_mode"]
        filename = f"gemm_tiles_{algo_string}_{mode_suffix}_trace_rank{rank}.json"
        timestamps.to_json(filename, gpu_freq)

    shmem.barrier()

    dist.barrier()
    dist.destroy_process_group()


def main():
    args = parse_args()

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
