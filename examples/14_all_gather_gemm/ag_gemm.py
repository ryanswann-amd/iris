#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import iris
import os
import sys
import torch.distributed as dist
import argparse
import socket


def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP


@triton.jit
def persistent_ag_gemm(
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
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
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

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        K_local = K // world_size

        for source_rank_id in range(world_size):
            loop_k_local = tl.cdiv(K_local, BLOCK_SIZE_K)
            if not EVEN_K:
                loop_k_local -= 1

            for k_block_idx in range(0, loop_k_local):
                k_offset = k_block_idx * BLOCK_SIZE_K
                rk_local = k_offset + tl.arange(0, BLOCK_SIZE_K)
                A_ptr = A + rm[:, None] * stride_am + rk_local[None, :] * stride_ak
                a = iris.load(tl.multiple_of(A_ptr, (1, 16)), cur_rank, source_rank_id, heap_bases)

                rk_global = (source_rank_id * K_local) + rk_local
                B_ptr = B + rk_global[:, None] * stride_bk + rn[None, :] * stride_bn
                b = tl.load(tl.multiple_of(B_ptr, (16, 1)))

                acc += tl.dot(a, b)

            if not EVEN_K:
                k_offset = loop_k_local * BLOCK_SIZE_K
                rk_local = k_offset + tl.arange(0, BLOCK_SIZE_K)
                rk_local_mask = rk_local < K_local
                A_ptr = A + rm[:, None] * stride_am + rk_local[None, :] * stride_ak
                a = iris.load(tl.multiple_of(A_ptr, (1, 16)), cur_rank, source_rank_id, heap_bases, mask=rk_local_mask[None, :], other=0.0)

                rk_global = (source_rank_id * K_local) + rk_local
                rk_global_mask = rk_global < K
                B_ptr = B + rk_global[:, None] * stride_bk + rn[None, :] * stride_bn
                b = tl.load(tl.multiple_of(B_ptr, (16, 1)), mask=rk_global_mask[:, None], other=0.0)

                acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)
        C_BASE = C + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_cm + \
                 (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_cn
        mask = ((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] < M) & \
               ((pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] < N)
        tl.store(C_BASE, c, mask=mask)

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


def test_correctness():
    iris_instance = iris.iris()
    rank = iris_instance.get_rank()
    world_size = iris_instance.get_num_ranks()

    torch.cuda.set_device(rank)

    batch_sizes = [1, 2, 4, 8, 16, 1024, 5]
    D = 8192
    F = 28672
    dtype = torch.float16
    TP = world_size
    if F % TP != 0 or D % TP != 0:
        if rank == 0:
            print(f"Error: F ({F}) and D ({D}) must be divisible by TP ({TP})")
        return

    for b_size in batch_sizes:
        if rank == 0:
            print(f"\n{'=' * 25} Testing Batch Size: {b_size} {'=' * 25}", flush=True)

        M = b_size
        N = F // TP
        K = D
        K_local = K // TP

        A_global, B_sharded = None, None
        if rank == 0:
            A_global = torch.randn((M, K), dtype=dtype, device="cuda")
            B_sharded = torch.randn((K, N), dtype=dtype, device="cuda")
        else:
            A_global = torch.empty((M, K), dtype=dtype, device="cpu")
            B_sharded = torch.empty((K, N), dtype=dtype, device="cpu")

        B_replicated = (
            torch.from_numpy(iris_instance.broadcast_tensor(B_sharded.cpu().numpy(), source_rank=0))
            .to(dtype)
            .to("cuda")
        )
        A_global_broadcasted = (
            torch.from_numpy(iris_instance.broadcast_tensor(A_global.cpu().numpy(), source_rank=0)).to(dtype).to("cuda")
        )

        A_local_iris = iris_instance.empty((M, K_local), dtype=dtype)
        A_slice_from_global = A_global_broadcasted[:, rank * K_local : (rank + 1) * K_local].contiguous()
        A_local_iris.copy_(A_slice_from_global)

        iris_instance.barrier()

        C_local = torch.empty((M, N), dtype=dtype, device="cuda")

        num_sms = torch.cuda.get_device_properties(rank).multi_processor_count
        persistent_ag_gemm[(num_sms,)](
            A_local_iris,
            B_replicated,
            C_local,
            M,
            N,
            K,
            A_local_iris.stride(0),
            A_local_iris.stride(1),
            B_replicated.stride(0),
            B_replicated.stride(1),
            C_local.stride(0),
            C_local.stride(1),
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=8,
            NUM_SMS=num_sms,
            NUM_XCDS=8,
            EVEN_K=True,
            heap_bases=iris_instance.get_heap_bases(),
            cur_rank=rank,
            world_size=world_size,
        )

        iris_instance.barrier()

        if rank == 0:
            C_ref = torch.matmul(A_global, B_sharded)
            is_correct = torch.allclose(C_local, C_ref, atol=1.0, rtol=0.1)
            if is_correct:
                print("✅ Test PASSED!")
                max_diff = torch.max(torch.abs(C_local - C_ref))
                print(f"  Max absolute difference: {max_diff:.4f}")
            else:
                print("❌ Test FAILED!")
            print("\n--- Output Samples (Batch Size {}) ---".format(b_size))
            print("Kernel output C_local sample:\n", C_local[:2, :4])
            print("Reference C_ref sample:\n", C_ref[:2, :4])


def test_performance():
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
    
    iris_instance = iris.iris(heap_size=8 * 1024**3)
    rank = iris_instance.get_rank()
    world_size = iris_instance.get_num_ranks()
    torch.cuda.set_device(rank)

    torch.manual_seed(42)

    if not dist.is_initialized():
        rendezvous_file = "/tmp/torch_rendezvous"
        if rank == 0:
            master_addr, master_port = get_ip_address(), 12355
            with open(rendezvous_file, "w") as f:
                f.write(f"{master_addr}\n{master_port}\n")
        iris_instance.barrier()
        with open(rendezvous_file, "r") as f:
            master_addr, master_port = [s.strip() for s in f.readlines()]
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{master_addr}:{master_port}", rank=rank, world_size=world_size
        )
        if rank == 0:
            os.remove(rendezvous_file)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 256, 512, 1024]
    D, F, dtype = 8192, 28672, torch.float16
    TP = world_size

    if F % TP != 0 or D % TP != 0:
        if rank == 0:
            print(f"Error: F ({F}) and D ({D}) must be divisible by TP ({TP})")
        return

    results = []
    for b_size in batch_sizes:
        if rank == 0:
            print(f"\n{'=' * 25} Benchmarking Batch Size: {b_size} {'=' * 25}", flush=True)
        M, N, K, K_local = b_size, F // TP, D, D // TP

        A_local_src = torch.randn((M, K_local), dtype=dtype, device="cuda")
        B_sharded_src = (
            torch.randn((K, N), dtype=dtype, device="cuda")
            if rank == 0
            else torch.empty((K, N), dtype=dtype, device="cuda")
        )

        dist.broadcast(B_sharded_src, src=0)
        all_a_shards = [torch.empty_like(A_local_src) for _ in range(world_size)]
        dist.all_gather(all_a_shards, A_local_src)
        A_global_src = torch.cat(all_a_shards, dim=1)
        dist.barrier()

        # --- 1. Benchmark Iris Fused Kernel ---
        A_local_iris = iris_instance.empty((M, K_local), dtype=dtype)
        A_local_iris.copy_(A_local_src)
        C_local_iris = torch.empty((M, N), dtype=dtype, device="cuda")

        num_sms = torch.cuda.get_device_properties(rank).multi_processor_count
        fn_to_benchmark_iris = lambda: persistent_ag_gemm[(num_sms,)](
            A_local_iris,
            B_sharded_src,
            C_local_iris,
            M,
            N,
            K,
            A_local_iris.stride(0),
            A_local_iris.stride(1),
            B_sharded_src.stride(0),
            B_sharded_src.stride(1),
            C_local_iris.stride(0),
            C_local_iris.stride(1),
            BLOCK_SIZE_M=256,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=64,
            GROUP_SIZE_M=6,
            NUM_SMS=num_sms,
            NUM_XCDS=1,
            EVEN_K=True,
            heap_bases=iris_instance.get_heap_bases(),
            cur_rank=rank,
            world_size=world_size,
        )
        time_iris_ms = iris.do_bench(
            fn=fn_to_benchmark_iris, barrier_fn=iris_instance.barrier, n_warmup=100, n_repeat=500, return_mode="mean"
        )
        if rank == 0:
            print(f"  1. Iris Fused AG+GEMM Time: {time_iris_ms:.4f} ms")

        # --- 2. Benchmark RCCL AG + Triton GEMM ---
        C_local_triton = torch.empty((M, N), dtype=dtype, device="cuda")

        def rccl_ag_triton_gemm_full():
            dist.all_gather(all_a_shards, A_local_src)
            A_gathered = torch.cat(all_a_shards, dim=1)
            local_gemm_kernel[(num_sms,)](
                A_gathered,
                B_sharded_src,
                C_local_triton,
                M,
                N,
                K,
                A_gathered.stride(0),
                A_gathered.stride(1),
                B_sharded_src.stride(0),
                B_sharded_src.stride(1),
                C_local_triton.stride(0),
                C_local_triton.stride(1),
                BLOCK_SIZE_M=256,
                BLOCK_SIZE_N=64,
                BLOCK_SIZE_K=64,
                GROUP_SIZE_M=6,
                NUM_SMS=num_sms,
                NUM_XCDS=1,
                EVEN_K=True,
            )

        time_rccl_triton_ms = iris.do_bench(
            fn=rccl_ag_triton_gemm_full, barrier_fn=dist.barrier, n_warmup=100, n_repeat=500, return_mode="mean"
        )
        if rank == 0:
            print(f"  2. RCCL AG + Triton GEMM Time: {time_rccl_triton_ms:.4f} ms")

        # --- 3. Benchmark PyTorch/RCCL Baseline ---
        def rccl_allgather_matmul():
            dist.all_gather(all_a_shards, A_local_src)
            A_gathered = torch.cat(all_a_shards, dim=1)
            torch.matmul(A_gathered, B_sharded_src)

        time_rccl_ms = iris.do_bench(
            fn=rccl_allgather_matmul, barrier_fn=dist.barrier, n_warmup=100, n_repeat=500, return_mode="mean"
        )
        if rank == 0:
            print(f"  3. PyTorch/RCCL Time: {time_rccl_ms:.4f} ms")

        results.append(
            {
                "batch_size": b_size,
                "iris_ms": time_iris_ms,
                "rccl_triton_ms": time_rccl_triton_ms,
                "rccl_torch_ms": time_rccl_ms,
            }
        )

    if rank == 0:
        print(f"\n\n{'=' * 50} Performance Summary {'=' * 50}")
        headers = [
            "Batch Size",
            "Iris Fused (ms)",
            "RCCL+Triton (ms)",
            "RCCL+Torch (ms)",
            "Speedup (vs RCCL+Triton)",
            "Speedup (vs RCCL+Torch)",
        ]
        print(
            f"{headers[0]:<12} | {headers[1]:<18} | {headers[2]:<18} | {headers[3]:<18} | {headers[4]:<25} | {headers[5]:<25}"
        )
        print("-" * 125)
        for res in results:
            speedup1 = res["rccl_triton_ms"] / res["iris_ms"]
            speedup2 = res["rccl_torch_ms"] / res["iris_ms"]
            print(
                f"{res['batch_size']:<12} | {res['iris_ms']:<18.4f} | {res['rccl_triton_ms']:<18.4f} | {res['rccl_torch_ms']:<18.4f} | {speedup1:<25.2f}x | {speedup2:<25.2f}x"
            )
        print("-" * 125)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run correctness or performance tests for the distributed GEMM kernel."
    )
    parser.add_argument(
        "--test", type=str, default="performance", choices=["correctness", "performance"], help="Which test to run"
    )
    args = parser.parse_args()

    if args.test == "performance":
        test_performance()
    else:
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
        if rank == 0:
            print("Running correctness test...")
        test_correctness()
