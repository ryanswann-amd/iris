#!/usr/bin/env python3
"""
Test Iris V2 MoE implementation
"""

import contextlib
import os
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

from triton_kernels.distributed import make_expt_dict_uniform, make_expt_assignment, symm_mem_pool
from moe_iris_v2 import moe_iris_v2


def _get_free_tcp_port():
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _distributed_worker(rank, fn, world_size, kwargs):
    dev = f"cuda:{rank}"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=torch.device(dev))
    torch.cuda.set_device(dev)
    try:
        fn(rank=rank, world_size=world_size, **kwargs)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_worker(rank, world_size):
    """Test Iris V2 MoE"""
    torch.manual_seed(0)
    dev = torch.cuda.current_device()
    n_shards = world_size

    # Test params
    n_tokens = 256
    d_model = 2048
    n_expts_tot = 16
    n_expts_act = 2

    if rank == 0:
        print(f"Testing Iris V2 MoE on {world_size} GPUs...")
        print(f"Tokens: {n_tokens}, d_model: {d_model}, experts: {n_expts_tot}, top_k: {n_expts_act}")

    expt_dict = make_expt_dict_uniform(n_shards, n_expts_tot)
    expt_assignment = make_expt_assignment(n_shards, n_expts_tot, expt_dict, device=dev)

    # Create data
    x_global = torch.randn(n_tokens, d_model, device=dev, dtype=torch.bfloat16)
    l_global = torch.rand(n_tokens, n_expts_tot, device=dev, dtype=torch.float32)
    w_global = torch.randn((n_expts_tot, d_model, d_model), device=dev, dtype=torch.bfloat16)
    b_global = torch.randn((n_expts_tot, d_model), device=dev, dtype=torch.float32)

    # Shard
    n_tokens_local = n_tokens // n_shards
    start_idx = rank * n_tokens_local
    end_idx = (rank + 1) * n_tokens_local

    w_ep_local = w_global[expt_assignment.expt_boolmask[rank, :], :, :]
    b_ep_local = b_global[expt_assignment.expt_boolmask[rank, :], :]
    x_dp_local = x_global[start_idx:end_idx, :]
    l_dp_local = l_global[start_idx:end_idx, :]

    # Initialize Iris
    import iris
    shmem = iris.iris()

    # Initialize Triton symmetric memory
    symm_mem_pool.initialize_matmul(
        n_tokens_global=n_tokens,
        d_input=d_model,
        d_model=d_model,
        n_expts_act=n_expts_act,
        n_expts_tot=n_expts_tot,
        dtype=torch.bfloat16,
        n_ranks=world_size,
        group=dist.group.WORLD,
        device=dev,
    )

    # Run Iris V2
    try:
        y = moe_iris_v2(x_dp_local, l_dp_local, w_ep_local, b_ep_local, expt_assignment, n_expts_act, shmem)

        if rank == 0:
            print(f"✓ Iris V2: shape={y.shape}, mean={y.mean():.4f}, std={y.std():.4f}")
            print("✓ SUCCESS: Iris V2 works!")
    except Exception as e:
        if rank == 0:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
        raise


def benchmark_worker(rank, world_size):
    """Benchmark Iris V2 MoE"""
    torch.manual_seed(0)
    dev = torch.cuda.current_device()
    n_shards = world_size

    # Test params
    n_tokens = 256
    d_model = 2048
    n_expts_tot = 16
    n_expts_act = 2

    expt_dict = make_expt_dict_uniform(n_shards, n_expts_tot)
    expt_assignment = make_expt_assignment(n_shards, n_expts_tot, expt_dict, device=dev)

    x_global = torch.randn(n_tokens, d_model, device=dev, dtype=torch.bfloat16)
    l_global = torch.rand(n_tokens, n_expts_tot, device=dev, dtype=torch.float32)
    w_global = torch.randn((n_expts_tot, d_model, d_model), device=dev, dtype=torch.bfloat16)
    b_global = torch.randn((n_expts_tot, d_model), device=dev, dtype=torch.float32)

    n_tokens_local = n_tokens // n_shards
    start_idx = rank * n_tokens_local
    end_idx = (rank + 1) * n_tokens_local

    w_ep_local = w_global[expt_assignment.expt_boolmask[rank, :], :, :]
    b_ep_local = b_global[expt_assignment.expt_boolmask[rank, :], :]
    x_dp_local = x_global[start_idx:end_idx, :]
    l_dp_local = l_global[start_idx:end_idx, :]

    # Initialize Iris
    import iris
    shmem = iris.iris()

    symm_mem_pool.initialize_matmul(
        n_tokens_global=n_tokens,
        d_input=d_model,
        d_model=d_model,
        n_expts_act=n_expts_act,
        n_expts_tot=n_expts_tot,
        dtype=torch.bfloat16,
        n_ranks=world_size,
        group=dist.group.WORLD,
        device=dev,
    )

    # Warmup
    for _ in range(5):
        y = moe_iris_v2(x_dp_local, l_dp_local, w_ep_local, b_ep_local, expt_assignment, n_expts_act, shmem)
        torch.cuda.synchronize()

    # Benchmark
    n_runs = 20
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()

    for _ in range(n_runs):
        y = moe_iris_v2(x_dp_local, l_dp_local, w_ep_local, b_ep_local, expt_assignment, n_expts_act, shmem)
        torch.cuda.synchronize()

    dist.barrier()
    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000 / n_runs

    if rank == 0:
        print(f"\n{'='*80}")
        print("Iris V2 MoE Benchmark Results")
        print(f"{'='*80}")
        print(f"Tokens: {n_tokens}, d_model: {d_model}, experts: {n_expts_tot}, top_k: {n_expts_act}")
        print(f"GPUs: {world_size}")
        print(f"Time: {elapsed_ms:.2f} ms")
        print(f"Target (Triton): 0.85 ms")
        print(f"Speedup vs target: {0.85/elapsed_ms:.2f}x")
        print(f"{'='*80}\n")


def run_test(world_size=8):
    master_port = _get_free_tcp_port()
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port))

    print("="*80)
    print("Phase 1: Testing Iris V2...")
    print("="*80)
    mp.spawn(_distributed_worker, args=(test_worker, world_size, {}), nprocs=world_size, join=True)

    print("\n" + "="*80)
    print("Phase 2: Benchmarking Iris V2...")
    print("="*80)
    mp.spawn(_distributed_worker, args=(benchmark_worker, world_size, {}), nprocs=world_size, join=True)


if __name__ == "__main__":
    run_test(world_size=8)

