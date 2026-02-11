#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for the HBM-buffered all_gather_matmul variant.

This variant cooperatively gathers A into a local HBM buffer with per-tile
ready flags, then runs GEMM from local memory. No global barriers -- CUs
that finish gathering early start GEMM immediately, spinning on flags for
any tile not yet available.

Usage with torchrun:
    torchrun --nproc_per_node=8 benchmark/ops/all_gather_matmul/benchmark_hbm_buffer.py \\
        -m 2048 -n 16384 -k 131072 --benchmark

    torchrun --nproc_per_node=8 benchmark/ops/all_gather_matmul/benchmark_hbm_buffer.py \\
        -m 2048 -n 16384 -k 131072 --benchmark --benchmark_pytorch --b_col_major
"""

import os
import time
import torch
import torch.distributed as dist
import random
import argparse

import iris
from iris.ops.all_gather_matmul_hbm_buffer import (
    all_gather_matmul_hbm_buffer,
    all_gather_matmul_hbm_buffer_preamble,
)
from iris.ops import FusedConfig

torch.manual_seed(123)
random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark HBM-buffered all_gather_matmul (per-tile flags).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=2048, help="M dimension")
    parser.add_argument("-n", type=int, default=16384, help="N dimension")
    parser.add_argument("-k", type=int, default=131072, help="K dimension (total)")
    parser.add_argument("-v", "--validate", action="store_true", help="Validate correctness")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument(
        "--datatype", type=str, default="fp16",
        choices=["fp16", "fp32", "bf16"], help="Tensor datatype",
    )
    parser.add_argument("--heap_size", type=int, default=1 << 34, help="Iris heap size")
    parser.add_argument("--comm_sms", type=int, default=None, help="Number of SMs (auto if None)")
    parser.add_argument(
        "--benchmark_pytorch", action="store_true",
        help="Also benchmark PyTorch (all_gather_into_tensor + matmul)",
    )
    parser.add_argument("--block_size_m", type=int, default=256, help="Block size M")
    parser.add_argument("--block_size_n", type=int, default=64, help="Block size N")
    parser.add_argument("--block_size_k", type=int, default=64, help="Block size K")
    parser.add_argument("--group_size_m", type=int, default=1, help="Group size M")
    parser.add_argument("--num_xcds", type=int, default=None, help="Number of XCDs (auto if None)")
    parser.add_argument("--b_col_major", action="store_true", help="B col-major (K-contiguous)")
    parser.add_argument("--a_col_major", action="store_true", help="A col-major (M-contiguous)")
    parser.add_argument("--single-run", action="store_true", help="1 iteration (for profiling)")
    return vars(parser.parse_args())


def _worker(args):
    """Worker function for torchrun."""
    local_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world_size_env = int(os.environ.get("WORLD_SIZE", 1))

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        dist.init_process_group(
            backend=backend, init_method="env://",
            device_id=torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None,
        )
    else:
        dist.init_process_group(
            backend=backend, init_method="tcp://127.0.0.1:29530",
            world_size=world_size_env, rank=local_rank,
            device_id=torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None,
        )

    shmem = iris.iris(args["heap_size"])
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    datatype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    datatype = datatype_map.get(args["datatype"], torch.float16)

    M = args["m"]
    N = args["n"]
    K = args["k"]
    K_local = K // world_size

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

    buffer_mb = M * K * torch.tensor([], dtype=datatype).element_size() / (1024 ** 2)
    num_m_tiles = M // config.block_size_m
    num_k_blocks = K // config.block_size_k
    shmem.info(
        f"HBM-Buffer variant: M={M} N={N} K={K} K_local={K_local} "
        f"block=({config.block_size_m},{config.block_size_n},{config.block_size_k}) "
        f"buffer={buffer_mb:.0f}MB flags={num_m_tiles}x{num_k_blocks}"
    )

    # ── Allocate tensors ─────────────────────────────────────────────────
    C = shmem.zeros((M, N), dtype=datatype)

    if args["a_col_major"]:
        A_storage = shmem.zeros((K_local, M), dtype=datatype)
        A_sharded = A_storage.T
    else:
        A_sharded = shmem.zeros((M, K_local), dtype=datatype)

    if args["b_col_major"]:
        B_storage = shmem.zeros((N, K), dtype=datatype)
        B = B_storage.T
    else:
        B = shmem.zeros((K, N), dtype=datatype)

    shmem.info(f"A strides={A_sharded.stride()}, B strides={B.stride()}")

    # Fill
    torch.manual_seed(123 + rank)
    A_data = torch.randn((M, K_local), dtype=datatype, device=f"cuda:{rank}")
    A_sharded.copy_(A_data)

    torch.manual_seed(456)
    B_data = torch.randn((K, N), dtype=datatype, device=f"cuda:{rank}")
    B.copy_(B_data)

    # Expected
    expected_tensor = None
    if args["validate"]:
        A_list = [torch.zeros((M, K_local), dtype=datatype, device=f"cuda:{rank}") for _ in range(world_size)]
        dist.all_gather(A_list, A_data)
        A_gathered = torch.cat(A_list, dim=1)
        expected_tensor = shmem.zeros((M, N), dtype=datatype)
        expected_tensor.copy_(torch.matmul(A_gathered, B_data))

    # Pre-allocate workspace
    workspace = all_gather_matmul_hbm_buffer_preamble(shmem, A_sharded, B, config)

    # ── Timing ───────────────────────────────────────────────────────────
    comm_stream = torch.cuda.Stream()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    num_experiments = 0

    def run_experiment():
        nonlocal total_ms, num_experiments
        shmem.barrier()
        with torch.cuda.stream(comm_stream):
            start_ev.record()
            all_gather_matmul_hbm_buffer(
                shmem, C, A_sharded, B,
                config=config, async_op=False, workspace=workspace,
            )
            end_ev.record()
            num_experiments += 1
        shmem.barrier()
        total_ms += start_ev.elapsed_time(end_ev)

    shmem.barrier()

    # ── Validate ─────────────────────────────────────────────────────────
    if args["validate"]:
        shmem.info("Validating...")
        C.zero_()
        shmem.barrier()
        run_experiment()
        torch.cuda.synchronize()
        shmem.barrier()

        atol = 1e-1 if datatype == torch.float16 else 1e-3
        success = torch.allclose(C, expected_tensor, atol=atol)
        if not success:
            max_diff = torch.abs(C - expected_tensor).max().item()
            shmem.error(f"Rank {rank}: Validation FAILED, max diff: {max_diff}")
        else:
            shmem.info("Validation PASSED!")
        shmem.barrier()

    # ── Benchmark ────────────────────────────────────────────────────────
    if args["benchmark"]:
        if args.get("single_run"):
            n_warmup, n_repeat = 0, 1
        else:
            n_warmup, n_repeat = 25, 100

        # Warmup
        total_ms = 0.0
        num_experiments = 0
        if n_warmup > 0:
            iris.do_bench(run_experiment, shmem.barrier, n_warmup=n_warmup, n_repeat=1)

        total_ms = 0.0
        num_experiments = 0
        C.zero_()
        shmem.barrier()

        iris.do_bench(run_experiment, shmem.barrier, n_warmup=0, n_repeat=n_repeat)
        avg_ms = total_ms / num_experiments if num_experiments > 0 else 0

        total_flops = 2 * M * N * K
        tflops = (total_flops * 1e-12) / (avg_ms * 1e-3) if avg_ms > 0 else 0
        element_size = torch.tensor([], dtype=datatype).element_size()
        total_bytes = M * K_local * element_size * (world_size - 1)
        bw_gbps = (total_bytes / (1024 ** 3)) / (avg_ms * 1e-3) if avg_ms > 0 else 0

        shmem.info(
            f"HBM-Buffer (M={M}, K_local={K_local}, K={K}, N={N}, "
            f"ws={world_size}, dtype={args['datatype']}): "
            f"{avg_ms:.3f} ms, {tflops:.3f} TFLOPS, {bw_gbps:.3f} GB/s"
        )
        shmem.barrier()

        # ── Per-rank finish time measurement ─────────────────────────────
        # Run a single iteration and record wall-clock finish time per rank
        # to see if ranks complete at different times (load imbalance).
        shmem.barrier()
        torch.cuda.synchronize()
        dist.barrier()

        # Synchronized start
        dist.barrier()
        t_start = time.perf_counter()

        all_gather_matmul_hbm_buffer(
            shmem, C, A_sharded, B,
            config=config, async_op=False, workspace=workspace,
        )
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        finish_ms = (t_end - t_start) * 1000.0

        # Gather all finish times to rank 0 for display
        finish_tensor = torch.tensor([finish_ms], dtype=torch.float64, device=f"cuda:{rank}")
        all_finish = [torch.zeros(1, dtype=torch.float64, device=f"cuda:{rank}") for _ in range(world_size)]
        dist.all_gather(all_finish, finish_tensor)

        if rank == 0:
            times = [t.item() for t in all_finish]
            min_t = min(times)
            max_t = max(times)
            print(f"\n  Per-rank finish times (single run):")
            print(f"  {'Rank':>6}  {'Finish ms':>10}  {'Delta ms':>10}")
            print(f"  {'-' * 30}")
            for r, t in enumerate(times):
                delta = t - min_t
                print(f"  {r:>6}  {t:>10.3f}  {delta:>+10.3f}")
            print(f"  {'-' * 30}")
            print(f"  Spread (max - min): {max_t - min_t:.3f} ms")
            print()

        shmem.barrier()

    # ── PyTorch baseline ─────────────────────────────────────────────────
    if args["benchmark_pytorch"]:
        shmem.info("Benchmarking PyTorch (all_gather_into_tensor + matmul)...")

        pt_A = torch.randn(M, K_local, dtype=datatype, device=f"cuda:{rank}")
        pt_B = torch.randn(K, N, dtype=datatype, device=f"cuda:{rank}")
        pt_Ag = torch.zeros(M, K, dtype=datatype, device=f"cuda:{rank}")

        for _ in range(10):
            dist.all_gather_into_tensor(pt_Ag, pt_A)
            _ = torch.matmul(pt_Ag, pt_B)
        torch.cuda.synchronize()
        dist.barrier()

        def run_pt():
            dist.all_gather_into_tensor(pt_Ag, pt_A)
            _ = torch.matmul(pt_Ag, pt_B)

        total_flops = 2 * M * N * K
        element_size = torch.tensor([], dtype=datatype).element_size()
        total_bytes = M * K_local * element_size * (world_size - 1)

        pt_ms = iris.do_bench(run_pt, dist.barrier)
        pt_tflops = (total_flops * 1e-12) / (pt_ms * 1e-3) if pt_ms > 0 else 0
        pt_bw = (total_bytes / (1024 ** 3)) / (pt_ms * 1e-3) if pt_ms > 0 else 0

        shmem.info(
            f"PyTorch (M={M}, K_local={K_local}, K={K}, N={N}, ws={world_size}, "
            f"dtype={args['datatype']}): "
            f"{pt_ms:.3f} ms, {pt_tflops:.3f} TFLOPS, {pt_bw:.3f} GB/s"
        )

        if args["benchmark"]:
            avg_ms = total_ms / num_experiments if num_experiments > 0 else 0
            iris_tflops = (total_flops * 1e-12) / (avg_ms * 1e-3) if avg_ms > 0 else 0
            speedup = iris_tflops / pt_tflops if pt_tflops > 0 else 0
            shmem.info(f"Speedup (HBM-Buffer / PyTorch): {speedup:.2f}x")

        shmem.barrier()

    shmem.barrier()
    dist.destroy_process_group()


def main():
    print("Starting HBM-buffer all_gather_matmul benchmark...")
    args = parse_args()
    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        _worker(args)
    else:
        print(
            "Please run with torchrun:\n"
            "  torchrun --nproc_per_node=N "
            "benchmark/ops/all_gather_matmul/benchmark_hbm_buffer.py [OPTIONS]"
        )


if __name__ == "__main__":
    main()
