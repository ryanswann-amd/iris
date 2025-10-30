#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import os
import sys
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import time

import iris.experimental.iris_rdma as iris_rdma


@triton.jit
def producer_put_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    rank_id,
    dst_rank: tl.constexpr,
    device_ctx,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Producer kernel that generates data and enqueues RDMA put operations.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Generate data: rank_id * 1000 + offset
    data = (rank_id * 1000 + offsets).to(tl.float32)
    
    # src_ptr is a pointer to float32, adding offsets automatically scales by sizeof(float32)
    src_ptrs = src_ptr + offsets
    
    # dst_ptr is an integer address, need to manually calculate byte offsets
    dst_ptrs = dst_ptr + offsets * 4  # multiply by sizeof(float32) to get byte addresses
    
    # Enqueue RDMA put to remote rank
    iris_rdma.put(dst_ptrs, src_ptrs, data, dst_rank, device_ctx, mask)


@triton.jit
def consumer_kernel(
    input_ptr,
    result_ptr,
    n_elements,
    expected_rank_id,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Consumer kernel that verifies received data.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load received data
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Check if it matches expected pattern
    expected = (expected_rank_id * 1000 + offsets).to(tl.float32)
    is_correct = (data == expected).to(tl.float32)
    
    tl.store(result_ptr + offsets, is_correct, mask=mask)


def main():
    # Initialize distributed
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device_id = torch.device(f"cuda:{local_rank}")
    
    dist.init_process_group(
        backend='nccl',
        device_id=device_id
    )
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if world_size < 2:
        print("This example requires at least 2 ranks")
        sys.exit(1)
    
    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'
    
    print(f"[Rank {rank}/{world_size}] Initialized on {device}")
    
    # Create Iris RDMA context with queue
    heap_size = 1024 * 1024 * 8  # 8MB
    queue_size = 512
    ctx = iris_rdma.iris(heap_size=heap_size, queue_size=queue_size)
    
    print(f"[Rank {rank}] Iris RDMA initialized")
    print(f"[Rank {rank}]   - Heap base: {ctx.get_heap_base():#x}")
    print(f"[Rank {rank}]   - Queue ptr: {ctx.get_queue_ptr():#x}")
    
    # Get device context for Triton kernels
    device_ctx = ctx.get_device_context()
    
    # Allocate buffers in symmetric heap
    n_elements = 4096
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Allocate in symmetric heap (GPU memory for GPUDirect RDMA)
    local_buffer = ctx.zeros(n_elements, dtype=torch.float32)  # Already on GPU
    
    # Move device_ctx to GPU
    device_ctx_gpu = device_ctx.to(device)
    
    print(f"[Rank {rank}] Local buffer on device: {local_buffer.device}")
    
    ctx.barrier()
    
    # ============================================================
    # PRODUCER (Rank 0): Generate data and RDMA put to Rank 1
    # ============================================================
    if rank == 0:
        print(f"\n[Rank 0] === Producer: Generating and Sending Data ===")
        
        # Get remote heap address for rank 1
        dst_rank = 1
        remote_heap_base = ctx.remote_heap_bases[dst_rank]
        
        # local_buffer is already on GPU, just get its pointer
        # Create pointer tensors on GPU
        local_ptr = local_buffer.data_ptr()
        remote_ptr = remote_heap_base
        
        print(f"[Rank 0] Launching Triton producer kernel")
        print(f"[Rank 0]   - Local ptr:  {local_ptr:#x}")
        print(f"[Rank 0]   - Remote ptr: {remote_ptr:#x}")
        print(f"[Rank 0]   - Dst rank:   {dst_rank}")
        
        # Launch producer kernel
        # This will:
        # 1. Generate data
        # 2. Store locally (in registered GPU heap)
        # 3. Enqueue RDMA put operations to queue
        producer_put_kernel[grid](
            local_buffer,  # Pass tensor directly (pointer will be extracted in kernel)
            remote_ptr,
            n_elements,
            rank_id=0,
            dst_rank=dst_rank,
            device_ctx=device_ctx_gpu,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Wait for GPU to finish enqueueing
        torch.cuda.synchronize()
        print(f"[Rank 0] ✓ Triton kernel completed (operations enqueued to queue)")
        print(f"[Rank 0]   Grid size was: {triton.cdiv(n_elements, BLOCK_SIZE)} programs")
        print(f"[Rank 0]   Each program should enqueue 1 work item")
        
        # Show what we sent
        print(f"[Rank 0] Sent data first 10: {local_buffer[:10].tolist()}")
    
    # Barrier: waits for queue to drain AND all ranks to sync
    # This ensures all RDMA operations have completed before proceeding
    print(f"[Rank {rank}] Waiting at barrier for RDMA completion...")
    ctx.barrier()
    print(f"[Rank {rank}] ✓ Barrier complete, all RDMA operations finished")
    
    # ============================================================
    # CONSUMER (Rank 1): Verify received data
    # ============================================================
    if rank == 1:
        print(f"\n[Rank 1] === Consumer: Verifying Received Data ===")
        
        # Show received data (already on GPU)
        print(f"[Rank 1] Received data first 10: {local_buffer[:10].tolist()}")
        
        # Verify data (already on GPU)
        result_buffer = torch.zeros(n_elements, dtype=torch.float32, device=device)
        
        consumer_kernel[grid](
            local_buffer,  # Already on GPU
            result_buffer,
            n_elements,
            expected_rank_id=0,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        result_cpu = result_buffer.cpu()
        num_correct = result_cpu.sum().item()
        num_total = n_elements
        
        print(f"[Rank 1] Verified: {int(num_correct)}/{num_total}")
        
        if num_correct == num_total:
            print(f"\n" + "="*60)
            print(f"[Rank 1] ✓ SUCCESS! Data matches perfectly!")
        else:
            print(f"[Rank 1] ✗ FAILED - Data mismatch!")
            first_wrong_idx = (result_cpu == 0).nonzero(as_tuple=True)[0]
            if len(first_wrong_idx) > 0:
                idx = first_wrong_idx[0].item()
                print(f"[Rank 1]   First wrong at index {idx}")
                print(f"[Rank 1]   Expected: {0 * 1000 + idx}")
                print(f"[Rank 1]   Got: {local_buffer[idx].item()}")
            sys.exit(1)
    
    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

