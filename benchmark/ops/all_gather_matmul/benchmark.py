#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for iris.ops all_gather_matmul fused operation.

This benchmark showcases the fused All-Gather + GEMM operation where each rank
has a sharded A matrix that gets gathered, then multiplied with B.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import argparse

from examples.common.utils import JSONWriter

import iris
from iris.ops import FusedConfig

torch.manual_seed(123)
random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark all_gather_matmul fused operation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=16384, help="Number of rows in matrix A (M)")
    parser.add_argument("-n", type=int, default=2048, help="Number of columns in matrix B (N)")
    parser.add_argument("-k", type=int, default=131072, help="Common dimension total (K)")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-v", "--validate", action="store_true", help="Enable validation mode")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Enable benchmarking mode")
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Datatype of tensors",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="all_gather_matmul.json",
        help="Output file",
    )
    parser.add_argument("--heap_size", type=int, default=1 << 34, help="Iris heap size")
    parser.add_argument("--comm_sms", type=int, default=None, help="Number of SMs for operation (auto-detect if None)")
    parser.add_argument(
        "--benchmark_pytorch",
        action="store_true",
        help="Also benchmark PyTorch (all_gather_into_tensor + matmul) for comparison",
    )
    parser.add_argument("--block_size_m", type=int, default=256, help="Block size for M dimension")
    parser.add_argument("--block_size_n", type=int, default=64, help="Block size for N dimension")
    parser.add_argument("--block_size_k", type=int, default=64, help="Block size for K dimension")
    parser.add_argument("--group_size_m", type=int, default=1, help="Group size for M dimension tiling")
    parser.add_argument("--num_xcds", type=int, default=None, help="Number of XCDs (auto-detected if not set)")
    parser.add_argument("-r", "--num_ranks", type=int, default=8, help="Number of ranks/processes")
    parser.add_argument(
        "--init_url", type=str, default="tcp://127.0.0.1:29530", help="Initialization URL for distributed setup"
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

    shmem = iris.iris(args["heap_size"])
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Datatype mapping
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

    M = args["m"]
    N = args["n"]
    K = args["k"]
    K_local = K // world_size  # Sharded K dimension

    # Create config with parameters
    config_kwargs = {
        "block_size_m": args["block_size_m"],
        "block_size_n": args["block_size_n"],
        "block_size_k": args["block_size_k"],
        "group_size_m": args["group_size_m"],
    }
    if args["comm_sms"] is not None:
        config_kwargs["num_sms"] = args["comm_sms"]
    if args["num_xcds"] is not None:
        config_kwargs["num_xcds"] = args["num_xcds"]

    config = FusedConfig(**config_kwargs)

    json_writer = JSONWriter(args["output_file"])
    json_writer.add_field("world_size", world_size)
    json_writer.add_field("operation", "all_gather_matmul")
    json_writer.add_field("k_local", K_local)
    json_writer.add_field("k_total", K)

    for key, value in args.items():
        json_writer.add_field(key, value)

    # Export actual config values to JSON (including defaults)
    json_writer.add_field("block_size_m", config.block_size_m)
    json_writer.add_field("block_size_n", config.block_size_n)
    json_writer.add_field("block_size_k", config.block_size_k)
    json_writer.add_field("group_size_m", config.group_size_m)
    json_writer.add_field("num_sms", config.num_sms)
    json_writer.add_field("num_xcds", config.num_xcds)

    # Create input and output tensors
    # A_sharded is M x K_local, B is K x N, output is M x N
    A_sharded = shmem.zeros((M, K_local), dtype=datatype)
    B = shmem.zeros((K, N), dtype=datatype)
    C = shmem.zeros((M, N), dtype=datatype)
    expected_tensor = None

    # Fill inputs with deterministic values
    # Each rank has different A_sharded, same B
    torch.manual_seed(123 + rank)
    A_sharded_data = torch.randn((M, K_local), dtype=datatype, device=f"cuda:{rank}")
    A_sharded.copy_(A_sharded_data)

    torch.manual_seed(456)  # Same B for all ranks
    B_data = torch.randn((K, N), dtype=datatype, device=f"cuda:{rank}")
    B.copy_(B_data)

    # For validation: compute expected result
    if args["validate"]:
        # Gather all A_sharded matrices and compute expected result
        A_sharded_list = [torch.zeros((M, K_local), dtype=datatype, device=f"cuda:{rank}") for _ in range(world_size)]
        dist.all_gather(A_sharded_list, A_sharded_data)

        # Concatenate along K dimension: A_gathered = [A_0 | A_1 | ... | A_n]
        A_gathered = torch.cat(A_sharded_list, dim=1)  # (M, K)

        # Expected: A_gathered @ B
        expected_tensor = shmem.zeros((M, N), dtype=datatype)
        expected_result = torch.matmul(A_gathered, B_data)
        expected_tensor.copy_(expected_result)

    comm_stream = torch.cuda.Stream()

    kernel_timing = {
        "all_gather_matmul": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
    }

    workspace = None

    def run_experiment():
        nonlocal kernel_timing, workspace

        # Preamble if available
        if hasattr(shmem.ops, "all_gather_matmul_preamble"):
            workspace = shmem.ops.all_gather_matmul_preamble(
                C,
                A_sharded,
                B,
                config=config,
                workspace=workspace,
            )

        shmem.barrier()

        torch.cuda.nvtx.range_push("All-Gather-Matmul")
        with torch.cuda.stream(comm_stream):
            kernel_timing["all_gather_matmul"]["start_event"].record()
            shmem.ops.all_gather_matmul(
                C,
                A_sharded,
                B,
                config=config,
                async_op=False,
                workspace=workspace,
            )
            kernel_timing["all_gather_matmul"]["end_event"].record()
            kernel_timing["all_gather_matmul"]["experiments"] += 1
        torch.cuda.nvtx.range_pop()

        # Synchronize before querying event timing
        shmem.barrier()

        # Update timing
        ms = kernel_timing["all_gather_matmul"]["start_event"].elapsed_time(
            kernel_timing["all_gather_matmul"]["end_event"]
        )
        kernel_timing["all_gather_matmul"]["ms"] += ms

    # Synchronize across all GPUs
    shmem.barrier()

    if args["validate"]:
        shmem.info("Validating...")

        # Reset output before validation
        C.zero_()
        shmem.barrier()

        run_experiment()
        torch.cuda.synchronize()
        shmem.barrier()

        atol = 1e-1 if datatype == torch.float16 else 1e-3
        success = torch.allclose(C, expected_tensor, atol=atol)
        if not success:
            max_diff = torch.abs(C - expected_tensor).max().item()
            shmem.error(f"Rank {rank}: Validation failed, max diff: {max_diff}")

        if success:
            shmem.info("All-gather-matmul validation passed!")
        else:
            shmem.error("All-gather-matmul validation failed!")

        json_writer.add_field("success", success)

        # Wait for all to finish validation
        shmem.barrier()

    if args["benchmark"]:
        # Warmup for benchmarking
        for k in ["all_gather_matmul"]:
            kernel_timing[k]["ms"] = 0
            kernel_timing[k]["experiments"] = 0

        iris.do_bench(run_experiment, shmem.barrier, n_warmup=25, n_repeat=1)

        for k in ["all_gather_matmul"]:
            kernel_timing[k]["ms"] = 0
            kernel_timing[k]["experiments"] = 0

        # Reset output before benchmarking
        C.zero_()
        shmem.barrier()

        shmem.info("Benchmarking...")

        # Calculate TFLOPS: 2*M*N*K flops
        total_flops = 2 * M * N * K
        total_tflops_unit = total_flops * 1e-12

        triton_ms = iris.do_bench(run_experiment, shmem.barrier)
        tflops = total_tflops_unit / (
            (kernel_timing["all_gather_matmul"]["ms"] / kernel_timing["all_gather_matmul"]["experiments"]) * 1e-3
        )

        # Calculate bandwidth for all-gather part
        # All-gather moves (world_size - 1) * M * K_local * element_size bytes
        element_size = torch.tensor([], dtype=datatype).element_size()
        input_bytes = M * K_local * element_size
        total_bytes = input_bytes * (world_size - 1)
        total_bytes_gb = total_bytes / (1024**3)

        bandwidth_gbps = total_bytes_gb / (
            (kernel_timing["all_gather_matmul"]["ms"] / kernel_timing["all_gather_matmul"]["experiments"]) * 1e-3
        )

        shmem.info(
            f"All-gather-matmul (M={M}, K_local={K_local}, K_total={K}, N={N}, world_size={world_size}, dtype={args['datatype']}): "
            f"{triton_ms:.3f} ms, {tflops:.3f} TFLOPS, {bandwidth_gbps:.3f} GB/s"
        )

        json_writer.add_field("tflops", tflops)
        json_writer.add_field("bandwidth_gbps", bandwidth_gbps)
        json_writer.add_field("total_ms", triton_ms)
        json_writer.add_field("total_flops", total_flops)
        json_writer.add_field("total_bytes", total_bytes)
        json_writer.add_field("total_bytes_gb", total_bytes_gb)
        json_writer.add_field(
            "all_gather_matmul_ms",
            kernel_timing["all_gather_matmul"]["ms"] / kernel_timing["all_gather_matmul"]["experiments"],
        )
        json_writer.add_field("all_gather_matmul_experiments", kernel_timing["all_gather_matmul"]["experiments"])

        # Wait for all to finish benchmarking
        shmem.barrier()

    # Benchmark PyTorch (all_gather_into_tensor + matmul) for comparison
    if args["benchmark_pytorch"]:
        shmem.info("Benchmarking PyTorch (all_gather_into_tensor + matmul)...")

        # Create PyTorch tensors (not on Iris heap)
        pytorch_A_sharded = torch.randn(M, K_local, dtype=datatype, device=f"cuda:{rank}")
        pytorch_B = torch.randn(K, N, dtype=datatype, device=f"cuda:{rank}")
        pytorch_A_gathered = torch.zeros(M, K, dtype=datatype, device=f"cuda:{rank}")
        pytorch_C = torch.zeros(M, N, dtype=datatype, device=f"cuda:{rank}")

        # Warmup
        for _ in range(10):
            dist.all_gather_into_tensor(pytorch_A_gathered, pytorch_A_sharded)
            pytorch_C = torch.matmul(pytorch_A_gathered, pytorch_B)
        torch.cuda.synchronize()
        dist.barrier()

        # Benchmark
        dist.barrier()

        def run_pytorch_experiment():
            dist.all_gather_into_tensor(pytorch_A_gathered, pytorch_A_sharded)
            pytorch_C = torch.matmul(pytorch_A_gathered, pytorch_B)

        pytorch_ms = iris.do_bench(run_pytorch_experiment, dist.barrier)

        # Calculate TFLOPS and bandwidth
        pytorch_tflops = total_tflops_unit / (pytorch_ms * 1e-3)
        pytorch_bandwidth_gbps = total_bytes_gb / (pytorch_ms * 1e-3)

        shmem.info(
            f"PyTorch all_gather_into_tensor+matmul (M={M}, K_local={K_local}, K_total={K}, N={N}, world_size={world_size}, dtype={args['datatype']}): "
            f"{pytorch_ms:.3f} ms, {pytorch_tflops:.3f} TFLOPS, {pytorch_bandwidth_gbps:.3f} GB/s"
        )

        if args["benchmark"]:
            # Calculate performance ratio
            iris_tflops = tflops
            speedup = (iris_tflops / pytorch_tflops) if pytorch_tflops > 0 else 0
            shmem.info(f"Speedup (Iris/PyTorch): {speedup:.2f}x")

            json_writer.add_field("pytorch_tflops", pytorch_tflops)
            json_writer.add_field("pytorch_bandwidth_gbps", pytorch_bandwidth_gbps)
            json_writer.add_field("pytorch_ms", pytorch_ms)
            json_writer.add_field("iris_speedup", speedup)

        # Wait for all to finish PyTorch benchmarking
        shmem.barrier()

    if rank == 0:
        json_writer.flush()
        json_writer.display()

    shmem.barrier()
    dist.destroy_process_group()


def main():
    args = parse_args()
    num_ranks = args["num_ranks"]
    init_url = args["init_url"]

    mp.spawn(
        fn=_worker,
        args=(num_ranks, init_url, args),
        nprocs=num_ranks,
        join=True,
    )


if __name__ == "__main__":
    main()
