#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import os
import sys
import torch
import torch.distributed as dist
import triton
import triton.language as tl

import iris.experimental.iris_rdma as iris_rdma


@triton.jit
def producer_kernel(
    output_ptr,
    n_elements,
    rank_id,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    data = (rank_id * 1000 + offsets).to(tl.float32)
    tl.store(output_ptr + offsets, data, mask=mask)


@triton.jit
def consumer_kernel(
    input_ptr,
    result_ptr,
    n_elements,
    expected_rank_id,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    expected = (expected_rank_id * 1000 + offsets).to(tl.float32)
    is_correct = (data == expected).to(tl.float32)
    
    tl.store(result_ptr + offsets, is_correct, mask=mask)


def main():
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
    
    heap_size = 1024 * 1024 * 8
    ctx = iris_rdma.iris(heap_size=heap_size)
    
    print(f"[Rank {rank}] Iris RDMA initialized")
    
    n_elements = 4096
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    local_buffer = ctx.zeros(n_elements, dtype=torch.float32)
    
    ctx.barrier()
    
    if rank == 0:
        print(f"\n[Rank 0] Producing data")
        
        gpu_buffer = local_buffer.to(device)
        
        producer_kernel[grid](
            gpu_buffer,
            n_elements,
            rank_id=0,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        local_buffer.copy_(gpu_buffer.cpu())
        
        print(f"[Rank 0] First 10: {local_buffer[:10].tolist()}")
        
        dst_rank = 1
        local_addr = local_buffer.data_ptr()
        remote_addr = ctx.remote_heap_bases[dst_rank]
        size = n_elements * 4
        
        print(f"[Rank 0] RDMA transfer to Rank {dst_rank}")
        
        ret = ctx.rdma_put(dst_rank, local_addr, remote_addr, size)
        
        if ret == 0:
            import time
            for attempt in range(100):
                n_comp = ctx.poll_completion(dst_rank)
                if n_comp > 0:
                    print(f"[Rank 0] RDMA completed")
                    break
                time.sleep(0.001)
    
    ctx.barrier()
    
    if rank == 1:
        print(f"\n[Rank 1] Consuming data")
        
        gpu_buffer = local_buffer.to(device)
        
        print(f"[Rank 1] Received first 10: {local_buffer[:10].tolist()}")
        
        result_buffer = torch.zeros(n_elements, dtype=torch.float32, device=device)
        
        consumer_kernel[grid](
            gpu_buffer,
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
            print(f"[Rank 1] SUCCESS!")
        else:
            print(f"[Rank 1] FAILED")
            sys.exit(1)
    
    ctx.barrier()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"RDMA Producer-Consumer Complete")
        print(f"{'='*60}")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

