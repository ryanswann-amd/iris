#!/usr/bin/env python3
"""
Iris MoE Benchmark - Shape sweeps and performance profiling
Adapted from triton_kernels/bench/bench_mlp.py to use Iris
"""

from itertools import chain
from pathlib import Path
import triton.profiler as proton
import torch
import torch.distributed as dist
import argparse
import sys
import os
import time

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    import triton_kernels.roofline as roofline
except ImportError:
    roofline = None

from triton_kernels.matmul import matmul
from triton_kernels.target_info import get_cdna_version
from triton_kernels.distributed import make_expt_dict_uniform, make_expt_assignment, symm_mem_pool
from triton_kernels.topk import topk
from triton_kernels.tensor import make_ragged_tensor_metadata, remap_ragged_tensor_metadata
from triton_kernels.reduce import reduce

# Iris imports
import iris

# Import Iris MoE implementation
from moe_iris_v2 import moe_iris_v2

# Import benchmark utilities from reference
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reference"))
from bench_utils import prepare_mlp_numerics, resolve_x_dtype
import tempfile


def bench_iris_moe(batch_per_expt, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP):
    """
    Benchmark Iris MoE with given configuration
    """
    assert n_expts_tot % EP == 0
    assert dim2 % TP == 0

    # Get rank and world_size from distributed context
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    dev = f"cuda:{rank}"
    DP = world_size
    batch = batch_per_expt * n_expts_tot // n_expts_act

    assert n_expts_tot % EP == 0, f"{n_expts_tot=}, {EP=}, n_expts_tot must be divisible by EP"
    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"

    # Initialize Iris
    shmem = iris.iris()

    # -- init data --
    # weights
    wg = torch.randn((dim1, n_expts_tot), device=dev)
    if world_size > 1:
        dist.broadcast(wg, src=0)
    w1 = torch.randn((n_expts_tot // EP, dim1, dim2 // TP), device=dev)
    w2 = torch.randn((n_expts_tot // EP, dim2 // TP // 2, dim1), device=dev)
    # biases
    bg = torch.randn((n_expts_tot,), device=dev)
    if world_size > 1:
        dist.broadcast(bg, src=0)
    b1 = torch.randn((n_expts_tot // EP, dim2 // TP), device=dev)
    b2 = torch.randn((n_expts_tot // EP, dim1), device=dev)

    # -- numerics --
    numerics = prepare_mlp_numerics(batch, w_dtype, wg, w1, w2)
    wg, w1, w2 = numerics.wg, numerics.w1, numerics.w2
    pcg, pc1, pc2, act = numerics.pcg, numerics.pc1, numerics.pc2, numerics.activation

    # -- benchmark --
    x_dtype = resolve_x_dtype(x_dtype)

    input_x = torch.randn((batch // DP, dim1), device=dev).to(x_dtype)
    logits = torch.rand((batch // DP, n_expts_tot), device=dev, dtype=torch.float32)

    expt_dict = make_expt_dict_uniform(EP, n_expts_tot)
    expt_assignment = make_expt_assignment(EP, n_expts_tot, expt_dict, device=torch.device(dev))

    # Initialize symmetric memory
    symm_mem_pool.initialize_matmul(
        n_tokens_global=batch,
        d_input=dim1,
        d_model=dim2,
        n_expts_act=n_expts_act,
        n_expts_tot=n_expts_tot,
        n_ranks=world_size,
        dtype=input_x.dtype,
        group=dist.group.WORLD if dist.is_initialized() else None,
        device=torch.cuda.current_device(),
    )

    # Run Iris MoE layer
    def run_layer():
        # For simplicity, use w1 as the expert weight matrix
        # In a real scenario, you'd concatenate w1 and w2 properly
        expert_weights = w1
        expert_biases = b1 if b1 is not None else torch.zeros_like(w1[:, 0, :])

        output = moe_iris_v2(input_x, logits, expert_weights, expert_biases, expt_assignment, n_expts_act, shmem)
        return output

    # Warmup
    for _ in range(5):
        _ = run_layer()
        torch.cuda.synchronize()

    # Benchmark
    n_iters = 20
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = run_layer()
        torch.cuda.synchronize()

    if dist.is_initialized():
        dist.barrier()
    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000 / n_iters

    if rank == 0:
        print(f"\n{'=' * 80}")
        print("Iris MoE Benchmark Results")
        print(f"{'=' * 80}")
        print("Configuration:")
        print(f"  Batch per expert: {batch_per_expt}")
        print(f"  Total batch: {batch}")
        print(f"  Input dim (dim1): {dim1}")
        print(f"  Hidden dim (dim2): {dim2}")
        print(f"  Total experts: {n_expts_tot}")
        print(f"  Active experts (top-k): {n_expts_act}")
        print(f"  Expert Parallelism (EP): {EP}")
        print(f"  Tensor Parallelism (TP): {TP}")
        print(f"  Data Parallelism (DP): {DP}")
        print(f"  Input dtype: {x_dtype}")
        print(f"  Weight dtype: {w_dtype}")
        print("\nPerformance:")
        print(f"  Time per iteration: {elapsed_ms:.3f} ms")
        print(f"  Throughput: {batch / elapsed_ms * 1000:.1f} tokens/sec")
        print(f"{'=' * 80}\n")

    # Cleanup
    symm_mem_pool.release()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _worker_init_and_run(
    rank, world_size, batch_per_expt, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP
):
    """Worker function for multiprocessing"""
    # Initialize process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=torch.device(f"cuda:{rank}"))
    torch.cuda.set_device(rank)

    # Run benchmark
    bench_iris_moe(batch_per_expt, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Iris MoE with various configurations")

    # Problem size
    parser.add_argument("--batch-per-expt", type=int, default=8, help="Tokens per expert")
    parser.add_argument("--dim1", type=int, default=2048, help="Input/output dimension")
    parser.add_argument("--dim2", type=int, default=8192, help="Hidden dimension")
    parser.add_argument("--n-expts-tot", type=int, default=16, help="Total number of experts")
    parser.add_argument("--n-expts-act", type=int, default=2, help="Number of active experts (top-k)")

    # Parallelism
    parser.add_argument("--EP", type=int, default=8, help="Expert parallelism (GPU count)")
    parser.add_argument("--TP", type=int, default=1, help="Tensor parallelism")

    # Data types
    parser.add_argument("--x-dtype", type=str, default="bf16", choices=["bf16", "fp8", "fp16"], help="Input data type")
    parser.add_argument("--w-dtype", type=str, default="bf16", choices=["bf16", "fp8", "mx4"], help="Weight data type")

    args = parser.parse_args()

    if torch.cuda.device_count() < args.EP * args.TP:
        print(f"Error: Need {args.EP * args.TP} GPUs, but only {torch.cuda.device_count()} available")
        return

    # Setup distributed
    import torch.multiprocessing as mp

    world_size = args.EP * args.TP

    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12356")

    if world_size > 1:
        mp.spawn(
            _worker_init_and_run,
            args=(
                world_size,
                args.batch_per_expt,
                args.dim1,
                args.dim2,
                args.n_expts_tot,
                args.n_expts_act,
                args.x_dtype,
                args.w_dtype,
                args.TP,
                args.EP,
            ),
            nprocs=world_size,
            join=True,
        )
    else:
        bench_iris_moe(
            args.batch_per_expt,
            args.dim1,
            args.dim2,
            args.n_expts_tot,
            args.n_expts_act,
            args.x_dtype,
            args.w_dtype,
            args.TP,
            args.EP,
        )


if __name__ == "__main__":
    main()
