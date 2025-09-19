import torch
import torch.nn.functional as F
import numpy as np
import triton
import triton.language as tl
import iris
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

from fused_kernel_iris import ff_a16w16_fused_ungated_iris

def test_correctness_iris_fused(iris_instance):
    rank = iris_instance.get_rank()
    world_size = iris_instance.get_num_ranks()
    torch.cuda.set_device(rank)

    def dist_print(msg):
        if rank == 0:
            print(msg)

    dist_print("Starting Correctness Test (Fused Iris Kernel)...")

    M, K, N = 8, 4096, 14336
    dtype = torch.float16
    activation = "relu"
    activation_fn = F.relu

    torch.manual_seed(42)

    dist_print("Preparing and distributing tensors...")
    if rank == 0:
        x_full = torch.randn((M, K), dtype=dtype, device="cpu")
        w1_full = torch.randn((N, K), dtype=dtype, device="cpu")
        w2_full = torch.randn((K, N), dtype=dtype, device="cpu").T.contiguous()
    else:
        x_full = torch.empty((M, K), dtype=dtype, device="cpu")
        w1_full = torch.empty((N, K), dtype=dtype, device="cpu")
        w2_full = torch.empty((N, K), dtype=dtype, device="cpu")

    x_full = torch.from_numpy(iris_instance.broadcast_tensor(x_full.numpy())).cuda()
    w1_full = torch.from_numpy(iris_instance.broadcast_tensor(w1_full.numpy())).cuda()
    w2_full = torch.from_numpy(iris_instance.broadcast_tensor(w2_full.numpy())).cuda()

    w1_shard = torch.chunk(w1_full, world_size, dim=0)[rank].contiguous()
    w2_shard = torch.chunk(w2_full, world_size, dim=0)[rank].contiguous()

    w1_shard = w1_shard / (N**0.5)
    w2_shard = w2_shard / (K**0.5)

    y_output = iris_instance.zeros((M, K), dtype=dtype)

    iris_instance.barrier()

    dist_print("Running your fused iris kernel on each rank...")
    ff_a16w16_fused_ungated_iris(
        x=x_full,
        w_up=w1_shard,
        w_down=w2_shard,
        iris_instance=iris_instance,
        y=y_output,
        dtype=dtype,
        activation=activation
    )

    iris_instance.barrier()
    dist_print("Kernel execution and fused AllReduce are complete.")

    if rank == 0:
        print("Running reference PyTorch implementation on rank 0...")
        w1_full_scaled = w1_full / (N**0.5)
        w2_full_scaled = w2_full / (K**0.5)
        intermediate_out = F.linear(x_full, w1_full_scaled)
        intermediate_out = activation_fn(intermediate_out)
        ref_output = intermediate_out @ w2_full_scaled

        print("Comparing results...")
        try:
            torch.testing.assert_close(y_output, ref_output, rtol=5e-2, atol=5e-2)
            print("\nCorrectness Test Passed!")
        except AssertionError as e:
            print(f"\nCorrectness Test FAILED: \n{e}")

def test_performance_iris_fused(iris_instance):
    rank = iris_instance.get_rank()
    world_size = iris_instance.get_num_ranks()
    torch.cuda.set_device(rank)

    def dist_print(msg):
        if rank == 0:
            print(msg)

    dist_print("Starting Performance Test (Fused Iris Kernel)...")

    K, N = 4096, 14336
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    dtype = torch.bfloat16
    activation = "relu"

    results = []

    for M in batch_sizes:
        dist_print(f"\n--- Benchmarking @ Batch Size (M) = {M} ---")

        x_full = torch.randn((M, K), dtype=dtype, device="cuda")
        w1_full = torch.randn((N, K), dtype=dtype, device="cuda")
        w2_full = torch.randn((K, N), dtype=dtype, device="cuda").T.contiguous()

        w1_shard = torch.chunk(w1_full, world_size, dim=0)[rank].contiguous()
        w2_shard = torch.chunk(w2_full, world_size, dim=0)[rank].contiguous()

        w1_shard = w1_shard / (N**0.5)
        w2_shard = w2_shard / (K**0.5)

        y_output = iris_instance.zeros((M, K), dtype=dtype)

        iris_instance.barrier()

        fn_to_benchmark = lambda: ff_a16w16_fused_ungated_iris(
            x=x_full,
            w_up=w1_shard,
            w_down=w2_shard,
            iris_instance=iris_instance,
            y=y_output,
            dtype=dtype,
            activation=activation
        )

        dist_print("Running benchmark...")
        avg_time_ms = iris.do_bench(
            fn=fn_to_benchmark,
            barrier_fn=iris_instance.barrier,
            n_warmup=10,
            n_repeat=50,
            return_mode="mean"
        )

        if rank == 0:
            results.append((M, avg_time_ms))
            print(f"       Avg Time: {avg_time_ms:.4f} ms")

    if rank == 0:
        print("\n" + "="*50)
        print("Fused FFN Iris Performance Summary")
        print("="*50)
        print(f"{'Batch Size (M)':<20} | {'Avg Time (ms)':<20}")
        print("-"*50)
        for m_val, avg_time in results:
            print(f"{m_val:<20} | {avg_time:<20.4f}")
        print("="*50)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Fused FFN Iris Tests with torch.multiprocessing.spawn")
    parser.add_argument("-n", "--num_ranks", type=int, default=2, help="Number of processes (GPUs) to use.")
    parser.add_argument("--correctness", action="store_true", help="Run the correctness test.")
    parser.add_argument("--performance", action="store_true", help="Run the performance test.")
    return parser.parse_args()

def worker(rank, world_size, init_url, cli_args):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method=init_url, world_size=world_size, rank=rank)

    iris_instance = iris.iris()

    if cli_args.correctness:
        test_correctness_iris_fused(iris_instance)

    if cli_args.performance:
        test_performance_iris_fused(iris_instance)

    iris_instance.barrier()
    dist.destroy_process_group()

def main():
    args = parse_args()

    if not args.correctness and not args.performance:
        if args.num_ranks > 0:
            print("Neither --correctness nor --performance was specified. Defaulting to --correctness.")
        args.correctness = True

    num_ranks = args.num_ranks
    init_url = "tcp://127.0.0.1:29501"
    mp.spawn(
        fn=worker,
        args=(num_ranks, init_url, args),
        nprocs=num_ranks,
        join=True,
    )

if __name__ == "__main__":
    main()