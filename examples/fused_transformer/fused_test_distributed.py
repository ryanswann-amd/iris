import torch
import torch.nn.functional as F
import torch.distributed as dist
import argparse
import os
import sys

# This assumes your working kernel and wrapper are in a file named fused_kernel.py
from fused_kernel import ff_a16w16_fused_ungated


def test_correctness_distributed():
    """
    Tests the kernel in a distributed setting with enhanced debug printing
    for tensor sizes and output slices.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    def dist_print(msg):
        if rank == 0:
            print(msg)

    dist_print("🚀 Starting Correctness Test (Distributed, Llama3 8B dimensions)...")

    # --- Test Parameters for Llama3 8B ---
    M, K, N = 8, 4096, 14336
    dtype = torch.bfloat16
    activation = "relu"
    activation_fn = F.relu

    torch.manual_seed(42)

    # --- 1. Prepare, Scale, and Distribute Tensors ---
    dist_print("Preparing and distributing tensors...")
    if rank == 0:
        x_full = torch.randn((M, K), dtype=dtype, device="cuda")
        w1_full = torch.randn((N, K), dtype=dtype, device="cuda")
        w2_full = torch.randn((K, N), dtype=dtype, device="cuda").T.contiguous()

        # --- NEW: PRINT GLOBAL SIZES ---
        print("\n--- Global Tensor Sizes (on Rank 0) ---")
        print(f"x_full shape:    {x_full.shape}")
        print(f"w1_full shape:   {w1_full.shape}")
        print(f"w2_full shape:   {w2_full.shape}")
        print("----------------------------------------")

    else:
        x_full = torch.empty((M, K), dtype=dtype, device="cuda")
        w1_full = torch.empty((N, K), dtype=dtype, device="cuda")
        w2_full = torch.empty((N, K), dtype=dtype, device="cuda")

    dist.broadcast(x_full, src=0)
    dist.broadcast(w1_full, src=0)
    dist.broadcast(w2_full, src=0)

    # Shard weights for tensor parallelism
    w1_shard = torch.chunk(w1_full, world_size, dim=0)[rank].contiguous()
    w2_shard = torch.chunk(w2_full, world_size, dim=0)[rank].contiguous()

    # --- NEW: PRINT PER-RANK SIZES ---
    # This will print on every rank
    print(f"[Rank {rank}] Shard w1_shard shape: {w1_shard.shape}")
    print(f"[Rank {rank}] Shard w2_shard shape: {w2_shard.shape}")

    w1_shard = w1_shard / (N**0.5)
    w2_shard = w2_shard / (K**0.5)

    dist.barrier()

    # --- 2. Run Triton Kernel on Sharded Data and AllReduce ---
    dist_print("\nRunning your Triton kernel on each rank...")
    kernel_partial_output = ff_a16w16_fused_ungated(
        x=x_full, w_up=w1_shard, w_down=w2_shard, dtype=dtype, activation=activation
    )

    dist_print("Performing AllReduce on kernel outputs...")
    dist.all_reduce(kernel_partial_output, op=dist.ReduceOp.SUM)

    dist.barrier()

    # --- 3. Run Reference Calculation on Rank 0 ---
    if rank == 0:
        print("\nRunning reference PyTorch implementation on rank 0...")
        w1_full_scaled = w1_full / (N**0.5)
        w2_full_scaled = w2_full / (K**0.5)

        intermediate_out = F.linear(x_full, w1_full_scaled)
        intermediate_out = activation_fn(intermediate_out)
        ref_output = intermediate_out @ w2_full_scaled

        # --- 4. Compare the Results ---
        print("Comparing results...")

        # --- NEW: PRINT RANDOM SLICE ---
        row_idx = M // 2
        col_start_idx = K // 2
        print("\n--- Checking a Random Slice Post-AllReduce ---")
        print(f"Slice Details: [row={row_idx}, cols={col_start_idx}:{col_start_idx + 8}]")
        print(f"Actual (Kernel):   {kernel_partial_output[row_idx, col_start_idx : col_start_idx + 8].tolist()}")
        print(f"Expected (Reference): {ref_output[row_idx, col_start_idx : col_start_idx + 8].tolist()}")
        print("----------------------------------------------\n")

        try:
            torch.testing.assert_close(kernel_partial_output, ref_output, rtol=5e-2, atol=5e-2)
            print("✅ Correctness Test Passed!")
        except AssertionError as e:
            print(f"❌ Correctness Test FAILED: \n{e}")


def main():
    """
    Initializes distributed environment and runs the test.
    How to Run: torchrun --nproc_per_node=2 your_script_name.py
    """
    if "RANK" not in os.environ:
        print("This script must be run with `torchrun`.")
        sys.exit(1)

    dist.init_process_group(backend="nccl")

    test_correctness_distributed()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
