from itertools import product
import json
import os
import pandas as pd
from datetime import datetime
import sys
import math

import torch
import torch.distributed as dist
import torch.profiler

# Add parent directory to path to import iris modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import iris
from gemm_all_scatter_bulk_synchronous import persistent_all_scatter


def benchmark(
    matmul_size,
    comm_size,
    matmul_stream,
    comm_stream,
    world_size,
    rank,
    shmem,
    comm_sms,
    BLK_M=256,
    BLK_N=64,
    gsize_m=6,
    warmup_steps=10,
    benchmark_steps=200,
):
    torch.cuda.empty_cache()
    num_xcds = iris.hip.get_num_xcc()
    
    with torch.device("cuda"):
        A = torch.randn(*matmul_size[0], dtype=torch.bfloat16)
        B = torch.randn(*matmul_size[1], dtype=torch.bfloat16)
        # For iris all_scatter, use shmem tensor
        comm_tensor = shmem.randn(*comm_size, device="cuda", dtype=torch.bfloat16)

    # warmup matmul
    for _ in range(warmup_steps):
        with torch.cuda.stream(matmul_stream):
            torch.matmul(A, B)
    torch.cuda.synchronize()

    # benchmark matmul
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(matmul_stream)
    for _ in range(benchmark_steps):
        with torch.cuda.stream(matmul_stream):
            torch.matmul(A, B)
    end_event.record(matmul_stream)
    torch.cuda.synchronize()

    matmul_time = start_event.elapsed_time(end_event) / benchmark_steps

    # warmup comm (iris persistent_all_scatter)
    for _ in range(warmup_steps):
        with torch.cuda.stream(comm_stream):
            persistent_all_scatter[(comm_sms,)](
                comm_tensor,
                comm_size[0],  # M
                comm_size[1],  # N
                comm_tensor.stride(0),
                comm_tensor.stride(1),
                BLK_M,
                BLK_N,
                gsize_m,
                comm_sms,
                num_xcds,
                shmem.get_heap_bases(),
                rank,
                world_size,
                False,  # COLLECT_TIMESTAMPS
                None,  # mm_begin_timestamp_ptr
                None,  # mm_end_timestamp_ptr
            )
    torch.cuda.synchronize()

    # benchmark comm
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(comm_stream)
    for _ in range(benchmark_steps):
        with torch.cuda.stream(comm_stream):
            persistent_all_scatter[(comm_sms,)](
                comm_tensor,
                comm_size[0],  # M
                comm_size[1],  # N
                comm_tensor.stride(0),
                comm_tensor.stride(1),
                BLK_M,
                BLK_N,
                gsize_m,
                comm_sms,
                num_xcds,
                shmem.get_heap_bases(),
                rank,
                world_size,
                False,  # COLLECT_TIMESTAMPS
                None,  # mm_begin_timestamp_ptr
                None,  # mm_end_timestamp_ptr
            )
    end_event.record(comm_stream)
    torch.cuda.synchronize()

    comm_time = start_event.elapsed_time(end_event) / benchmark_steps

    # warmup matmul-comm
    for _ in range(warmup_steps):
        with torch.cuda.stream(matmul_stream):
            torch.matmul(A, B)
        with torch.cuda.stream(comm_stream):
            persistent_all_scatter[(comm_sms,)](
                comm_tensor,
                comm_size[0],
                comm_size[1],
                comm_tensor.stride(0),
                comm_tensor.stride(1),
                BLK_M,
                BLK_N,
                gsize_m,
                comm_sms,
                num_xcds,
                shmem.get_heap_bases(),
                rank,
                world_size,
                False,
                None,
                None,
            )
    torch.cuda.synchronize()

    # benchmark matmul-comm
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_matmul_event = torch.cuda.Event(enable_timing=True)
    end_matmul_event = torch.cuda.Event(enable_timing=True)
    start_comm_event = torch.cuda.Event(enable_timing=True)
    end_comm_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    matmul_stream.wait_stream(torch.cuda.current_stream())
    comm_stream.wait_stream(torch.cuda.current_stream())

    start_matmul_event.record(matmul_stream)
    start_comm_event.record(comm_stream)

    for _ in range(benchmark_steps):
        with torch.cuda.stream(matmul_stream):
            torch.matmul(A, B)
        with torch.cuda.stream(comm_stream):
            persistent_all_scatter[(comm_sms,)](
                comm_tensor,
                comm_size[0],
                comm_size[1],
                comm_tensor.stride(0),
                comm_tensor.stride(1),
                BLK_M,
                BLK_N,
                gsize_m,
                comm_sms,
                num_xcds,
                shmem.get_heap_bases(),
                rank,
                world_size,
                False,
                None,
                None,
            )

    end_matmul_event.record(matmul_stream)
    end_comm_event.record(comm_stream)

    torch.cuda.current_stream().wait_stream(matmul_stream)
    torch.cuda.current_stream().wait_stream(comm_stream)
    end_event.record()
    
    torch.cuda.synchronize()

    matmul_comm_time = start_event.elapsed_time(end_event) / benchmark_steps
    overlapped_matmul_time = start_matmul_event.elapsed_time(end_matmul_event) / benchmark_steps
    overlapped_comm_time = start_comm_event.elapsed_time(end_comm_event) / benchmark_steps

    return matmul_time, comm_time, matmul_comm_time, overlapped_matmul_time, overlapped_comm_time


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Initialize iris shared memory
    heap_size = 1 << 33  # 8GB
    shmem = iris.iris(heap_size)
    
    # Get compute unit count and calculate comm_sms
    cu_count = torch.cuda.get_device_properties(local_rank).multi_processor_count
    comm_sms = 2 ** int(math.log2(cu_count)) if cu_count > 0 else 1
    
    # Tiling parameters
    BLK_M = 256
    BLK_N = 64
    gsize_m = 6

    matmul_stream = torch.cuda.Stream()
    comm_stream = torch.cuda.Stream()

    # Define matmul sizes as (size_A, size_B) for A @ B
    # A=(3840, 4352), B=(4352, 3840)
    matmul_sizes = [
        ((3840, 4352), (4352, 3840))
    ]
    
    # comm_sizes use the same shape as matrix A
    comm_sizes = [(3840, 4352)]
    
    if rank == 0:
        print(f"Using iris persistent_all_scatter with world_size={world_size}")
        print(f"comm_sms={comm_sms}, BLK_M={BLK_M}, BLK_N={BLK_N}")
        print(f"Matmul: A @ B where A={matmul_sizes[0][0]}, B={matmul_sizes[0][1]}")
        print(f"Comm size: {comm_sizes[0]}")
    results = []
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for matmul_size, comm_size in product(matmul_sizes, comm_sizes):
            matmul_time, comm_time, matmul_comm_time, overlapped_matmul_time, overlapped_comm_time = benchmark(
                matmul_size,
                comm_size,
                matmul_stream,
                comm_stream,
                world_size,
                rank,
                shmem,
                comm_sms,
                BLK_M,
                BLK_N,
                gsize_m,
            )
            # Get environment variables
            tensile_grid = os.environ.get("TENSILE_STREAMK_FIXED_GRID", "unset")
            nccl_channels = os.environ.get("NCCL_MAX_NCHANNELS", "unset")
            
            size_A, size_B = matmul_size
            results.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tensile_streamk_fixed_grid": tensile_grid,
                "nccl_max_nchannels": nccl_channels,
                "matmul_A_shape": f"{size_A[0]}x{size_A[1]}",
                "matmul_B_shape": f"{size_B[0]}x{size_B[1]}",
                "comm_shape": f"{comm_size[0]}x{comm_size[1]}",
                "matmul_time": matmul_time,
                "comm_time": comm_time,
                "matmul_comm_time": matmul_comm_time,
                "overlapped_matmul_time": overlapped_matmul_time,
                "overlapped_comm_time": overlapped_comm_time,
                "overlapped_matmul_time_ratio":  overlapped_matmul_time/matmul_time,
            })
            if rank == 0:
                print(
                    f"A: {size_A[0]}x{size_A[1]} @ B: {size_B[0]}x{size_B[1]}, comm: {comm_size[0]}x{comm_size[1]}",
                )
                print(f"  matmul alone:         {matmul_time:.4f}ms")
                print(f"  comm alone:           {comm_time:.4f}ms")
                print(f"  matmul + comm:        {matmul_comm_time:.4f}ms")
                print(f"  overlapped matmul:    {overlapped_matmul_time:.4f}ms")
                print(f"  overlapped comm:      {overlapped_comm_time:.4f}ms")
                print("-" * 60)

    if rank == 0:
        prof.export_chrome_trace(f"iris_allscatter_hipblaslt_trace_rank{rank}.json")
        print(f"Profiler trace saved to iris_allscatter_hipblaslt_trace_rank{rank}.json")
        
        with open("overlap_results.json", "w") as f:
            json.dump(results, f)
        
        # Save to Excel with append functionality
        excel_file = "overlap_results_200.xlsx"
        df = pd.DataFrame(results)
        
        # Check if Excel file exists
        if os.path.exists(excel_file):
            # Read existing data and append new results
            try:
                existing_df = pd.read_excel(excel_file)
                df = pd.concat([existing_df, df], ignore_index=True)
            except Exception as e:
                print(f"Warning: Could not read existing Excel file: {e}")
                print("Creating new Excel file...")
        
        # Save to Excel
        try:
            df.to_excel(excel_file, index=False)
            print(f"Results saved to {excel_file}")
        except Exception as e:
            print(f"Error saving to Excel: {e}")
            print("Results saved to JSON only.")

    shmem.barrier()
    dist.destroy_process_group()
