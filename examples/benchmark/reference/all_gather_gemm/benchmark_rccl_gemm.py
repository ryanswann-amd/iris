#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import random
import sys
import os
import argparse
import json

from examples.common.utils import JSONWriter
import iris

torch.manual_seed(123)
random.seed(123)


@triton.jit
def local_gemm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm_load = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn_load = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)

        rm_load = tl.max_contiguous(tl.multiple_of(rm_load, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn_load = tl.max_contiguous(tl.multiple_of(rn_load, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm_load[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn_load[None, :] * stride_bn

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm_load[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn_load[None, :] * stride_bn
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)

        rm_store = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn_store = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        rm_store = tl.max_contiguous(tl.multiple_of(rm_store, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn_store = tl.max_contiguous(tl.multiple_of(rn_store, BLOCK_SIZE_N), BLOCK_SIZE_N)
        C_BASE = C + rm_store[:, None] * stride_cm + rn_store[None, :] * stride_cn

        mask = (rm_store[:, None] < M) & (rn_store[None, :] < N)
        tl.store(C_BASE, c, mask=mask)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a sweep of RCCL All-Gather + Triton GEMM benchmarks from a config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--validate", action="store_true", help="Enable validation mode.")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Enable benchmarking mode.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="dataset/ag_gemm.json",
        help="Path to the JSON file with benchmark configurations.",
    )
    parser.add_argument("--output_file", type=str, default="rccl_gemm_log.json", help="Base name for output files")

    parser.add_argument("-m", type=int, default=1024, help="Number of rows in matrix A (M)")
    parser.add_argument("-n", type=int, default=3584, help="Total number of columns in matrix B (N)")
    parser.add_argument("-k", type=int, default=8192, help="Common dimension between matrices A and B (K)")

    parser.add_argument(
        "--datatype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"], help="Datatype of computation"
    )

    parser.add_argument("--BLK_M", type=int, default=256, help="Block size M for the kernel")
    parser.add_argument("--BLK_N", type=int, default=64, help="Block size N for the kernel")
    parser.add_argument("--BLK_K", type=int, default=64, help="Block size K for the kernel")
    parser.add_argument("--gsize_m", type=int, default=6, help="Group size in M dimension")
    parser.add_argument("--num_sms", type=int, default=304, help="Number of SMs for the kernel")

    return parser.parse_args()


def main():
    default_args = parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    output_dir = "results/rccl_gemm"
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    dist.barrier()

    with open(default_args.config_file, "r") as f:
        configs_to_run = json.load(f)

    if rank == 0:
        print(f"Loaded {len(configs_to_run)} configurations from {default_args.config_file}")

    for config in configs_to_run:
        run_args = vars(default_args).copy()
        run_args.update(config)

        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        datatype = dtype_map.get(run_args["datatype"])

        M, N, K = run_args["m"], run_args["n"], run_args["k"]
        if rank == 0:
            print(f"\n--- Running Benchmark for M={M}, N={N}, K={K} ---")
            sys.stdout.flush()

        base_name, extension = os.path.splitext(default_args.output_file)
        unique_filename = f"{base_name}_m_{M}{extension}"
        full_output_path = os.path.join(output_dir, unique_filename)

        json_writer = JSONWriter(full_output_path)
        json_writer.add_field("world_size", world_size)
        for key, value in run_args.items():
            json_writer.add_field(key, value)

        K_local = K // world_size

        if rank == 0:
            A_global = torch.randn((M, K), dtype=datatype, device="cuda")
        else:
            A_global = torch.empty((M, K), dtype=datatype, device="cuda")

        dist.broadcast(A_global, src=0)
        dist.barrier()

        A_local = A_global[:, rank * K_local : (rank + 1) * K_local].contiguous()

        if rank == 0:
            B = torch.randn((K, N), device="cuda", dtype=datatype)
        else:
            B = torch.empty((K, N), device="cuda", dtype=datatype)

        dist.broadcast(B, src=0)
        dist.barrier()

        C = torch.empty((M, N), device="cuda", dtype=datatype)

        all_a_shards = [torch.empty_like(A_local) for _ in range(world_size)]

        num_sms = torch.cuda.get_device_properties(rank).multi_processor_count

        main_stream = torch.cuda.Stream()

        kernel_timing = {
            "rccl_all_gather": {
                "start_event": torch.cuda.Event(enable_timing=True),
                "end_event": torch.cuda.Event(enable_timing=True),
                "ms": 0,
                "experiments": 0,
            },
            "local_gemm": {
                "start_event": torch.cuda.Event(enable_timing=True),
                "end_event": torch.cuda.Event(enable_timing=True),
                "ms": 0,
                "experiments": 0,
            },
        }

        A_gathered = torch.empty((M, K), dtype=datatype, device="cuda")

        def run_experiment():
            nonlocal kernel_timing, A_gathered
            with torch.cuda.stream(main_stream):
                kernel_timing["rccl_all_gather"]["start_event"].record()
                dist.all_gather(all_a_shards, A_local)
                A_gathered = torch.cat(all_a_shards, dim=1)
                kernel_timing["rccl_all_gather"]["end_event"].record()

                kernel_timing["local_gemm"]["start_event"].record()
                local_gemm_kernel[(num_sms,)](
                    A_gathered,
                    B,
                    C,
                    M,
                    N,
                    K,
                    A_gathered.stride(0),
                    A_gathered.stride(1),
                    B.stride(0),
                    B.stride(1),
                    C.stride(0),
                    C.stride(1),
                    run_args["BLK_M"],
                    run_args["BLK_N"],
                    run_args["BLK_K"],
                    run_args["gsize_m"],
                    num_sms,
                    1,
                    (K % run_args["BLK_K"] == 0),
                )
                kernel_timing["local_gemm"]["end_event"].record()

            torch.cuda.synchronize()
            kernel_timing["rccl_all_gather"]["ms"] += kernel_timing["rccl_all_gather"]["start_event"].elapsed_time(
                kernel_timing["rccl_all_gather"]["end_event"]
            )
            kernel_timing["rccl_all_gather"]["experiments"] += 1
            kernel_timing["local_gemm"]["ms"] += kernel_timing["local_gemm"]["start_event"].elapsed_time(
                kernel_timing["local_gemm"]["end_event"]
            )
            kernel_timing["local_gemm"]["experiments"] += 1

        run_experiment()
        dist.barrier()

        for key in kernel_timing:
            kernel_timing[key]["ms"] = 0
            kernel_timing[key]["experiments"] = 0

        if default_args.benchmark:
            total_ms = iris.do_bench(run_experiment, barrier_fn=dist.barrier)
            tflops = 2 * M * N * K * 1e-12 / (total_ms * 1e-3)

            if rank == 0:
                print(f"Result (iris.do_bench): {total_ms:.3f} ms, {tflops:.3f} TFLOPS")
                json_writer.add_field("total_ms", total_ms)
                json_writer.add_field("tflops", tflops)

            for key in kernel_timing:
                if kernel_timing[key]["experiments"] > 0:
                    avg_kernel_ms = kernel_timing[key]["ms"] / kernel_timing[key]["experiments"]
                    json_writer.add_field(key + "_ms", avg_kernel_ms)
                    if rank == 0:
                        print(f"Result (CUDA Events) - {key}: {avg_kernel_ms:.3f} ms")

        if default_args.validate:
            if not default_args.benchmark:
                run_experiment()
                dist.barrier()

            if rank == 0:
                print("Validating...")

            C_ref = torch.matmul(A_global, B)
            success = torch.allclose(C, C_ref, atol=1.0, rtol=0.05)

            passed_str = "passed" if success else "failed"
            print(f"Final C validation for rank {rank} is {passed_str}.")
            json_writer.add_field("validation_passed", success)

        if rank == 0:
            json_writer.flush()
            print(f"Saved results to {full_output_path}")
            sys.stdout.flush()

    if rank == 0:
        print("\nBenchmark sweep complete.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
