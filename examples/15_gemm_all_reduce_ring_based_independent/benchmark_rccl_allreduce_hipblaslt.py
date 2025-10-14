from itertools import product
import json
import os
import pandas as pd
from datetime import datetime

import torch
import torch.distributed as dist
import torch.profiler


def benchmark(
    matmul_size,
    comm_size,
    matmul_stream,
    comm_stream,
    world_size,
    warmup_steps=10,
    benchmark_steps=200,
):
    torch.cuda.empty_cache()
    with torch.device("cuda"):
        A = torch.randn(*matmul_size[0], dtype=torch.bfloat16)
        B = torch.randn(*matmul_size[1], dtype=torch.bfloat16)
        comm_tensor = torch.randn(*comm_size, dtype=torch.bfloat16)

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

    # warmup comm (all_reduce)
    for _ in range(warmup_steps):
        with torch.cuda.stream(comm_stream):
            dist.all_reduce(comm_tensor)
    torch.cuda.synchronize()

    # benchmark comm
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(comm_stream)
    for _ in range(benchmark_steps):
        with torch.cuda.stream(comm_stream):
            dist.all_reduce(comm_tensor)
    end_event.record(comm_stream)
    torch.cuda.synchronize()

    comm_time = start_event.elapsed_time(end_event) / benchmark_steps

    # warmup matmul-comm
    for _ in range(warmup_steps):
        with torch.cuda.stream(matmul_stream):
            torch.matmul(A, B)
        with torch.cuda.stream(comm_stream):
            dist.all_reduce(comm_tensor)
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
            dist.all_reduce(comm_tensor)

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

    matmul_stream = torch.cuda.Stream()
    comm_stream = torch.cuda.Stream()

    # Define matmul sizes as (size_A, size_B) for A @ B
    # A=(3840, 4352), B=(4352, 3840)
    matmul_sizes = [((3840, 4352), (4352, 3840))]

    # comm_sizes use the same shape as matrix A
    comm_sizes = [(3840, 4352)]

    if rank == 0:
        print(f"Using RCCL all_reduce with world_size={world_size}")
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
            )
            # Get environment variables
            tensile_grid = os.environ.get("TENSILE_STREAMK_FIXED_GRID", "unset")
            nccl_channels = os.environ.get("NCCL_MAX_NCHANNELS", "unset")

            size_A, size_B = matmul_size
            results.append(
                {
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
                    "overlapped_matmul_time_ratio": overlapped_matmul_time / matmul_time,
                }
            )
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
        prof.export_chrome_trace(f"rccl_allreduce_trace_rank{rank}.json")
        print(f"Profiler trace saved to rccl_allreduce_trace_rank{rank}.json")

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

    dist.destroy_process_group()
