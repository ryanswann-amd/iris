#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Comprehensive CCL latency/bandwidth sweep for iris collectives.
#
# Measures all_reduce, all_gather, reduce_scatter across 2/4/8 GPUs
# and ~35 message sizes from 1KB to 1GB.
#
# Architecture:
#   - Worker mode (--worker): launched via torchrun, benchmarks one collective
#     across all message sizes. Fresh iris context per invocation.
#   - Driver mode (default): launches workers via torchrun for each
#     (collective, gpu_count) combination, then post-processes results.
#   - Dry-run mode (--dry-run): prints config matrix, no GPU work.
#
# Usage:
#   python3 ccl_sweep.py                       # full sweep (driver)
#   python3 ccl_sweep.py --dry-run             # config matrix only
#   python3 ccl_sweep.py --worker --collective all_reduce  # worker mode
#   python3 ccl_sweep.py --output-dir /path/to/results     # custom output

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Message sizes: powers of 2 from 1KB to 1GB + midpoints
# ---------------------------------------------------------------------------
def _build_message_sizes() -> list[int]:
    """Build ~35 message sizes spanning 1KB to 1GB."""
    sizes = set()
    # Powers of 2: 2^10 (1KB) to 2^30 (1GB)
    for exp in range(10, 31):
        sizes.add(1 << exp)
    # Midpoints at ~1.5x each decade boundary
    midpoints = [
        3 * 1024, 6 * 1024, 12 * 1024,
        48 * 1024, 96 * 1024,
        384 * 1024, 768 * 1024,
        int(1.5 * 1024**2), 3 * 1024**2, 6 * 1024**2,
        48 * 1024**2, 96 * 1024**2,
        384 * 1024**2, 768 * 1024**2,
    ]
    sizes.update(midpoints)
    return sorted(sizes)


MSG_SIZES = _build_message_sizes()
COLLECTIVES = ["all_reduce", "all_gather", "reduce_scatter"]
GPU_COUNTS = [2, 4, 8]
DTYPE_NAME = "bfloat16"
ELEM_BYTES = 2
N_WARMUP = 10
N_REPEAT = 20
HEAP_SIZE = 1 << 34  # 16 GB

ALGORITHMS = {
    "all_reduce": "two_shot",
    "all_gather": "persistent",
    "reduce_scatter": "two_shot",
}

# Block sizes for iris kernel tiling
BLOCK_M = 32
BLOCK_N = 64


def _msg_size_to_mn(msg_bytes: int) -> tuple[int, int]:
    """Convert message bytes to (M, N) shape for bf16 tensors.

    iris CCL kernels tile by BLOCK_M x BLOCK_N. Both M and N must
    be multiples of those block sizes.
    """
    total_elements = msg_bytes // ELEM_BYTES

    if total_elements <= BLOCK_M * BLOCK_N:
        return BLOCK_M, BLOCK_N

    M = BLOCK_M
    N_raw = (total_elements + M - 1) // M
    N = ((N_raw + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    return M, N


def _actual_bytes(M: int, N: int) -> int:
    return M * N * ELEM_BYTES


def _fmt_bytes(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / (1024**3):.1f} GB"
    elif n >= 1024 ** 2:
        return f"{n / (1024**2):.1f} MB"
    elif n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _bus_bandwidth_bytes(collective: str, M: int, N: int, world_size: int) -> int:
    """Compute bus bandwidth factor for each collective."""
    data_size = M * N * ELEM_BYTES
    if collective == "all_reduce":
        return int(2 * (world_size - 1) / world_size * data_size)
    elif collective == "all_gather":
        return (world_size - 1) * data_size
    elif collective == "reduce_scatter":
        return int((world_size - 1) / world_size * data_size)
    return data_size


# ---------------------------------------------------------------------------
# Worker mode: launched via torchrun
# ---------------------------------------------------------------------------
def run_worker(args):
    """Run as a torchrun worker -- benchmark one collective across all sizes."""
    import torch
    import torch.distributed as dist
    import iris
    from iris.ccl import Config

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()

    collective = args.collective
    output_file = args.output

    ctx = iris.iris(HEAP_SIZE)
    dtype = torch.bfloat16

    results = []

    for msg_bytes in MSG_SIZES:
        M, N = _msg_size_to_mn(msg_bytes)
        actual = _actual_bytes(M, N)

        # Check memory: for all_gather, output is world_size * M * N * 2
        if collective == "all_gather":
            total_alloc = actual + world_size * actual
        else:
            total_alloc = 2 * actual  # inp + out

        # Skip if would exceed ~80% of heap
        max_alloc = int(HEAP_SIZE * 0.8)
        if total_alloc > max_alloc:
            if rank == 0:
                print(f"  SKIP {_fmt_bytes(msg_bytes)}: alloc {_fmt_bytes(total_alloc)} > {_fmt_bytes(max_alloc)}")
            continue

        try:
            # Allocate tensors
            inp = ctx.zeros((M, N), dtype=dtype)
            if collective == "all_gather":
                out = ctx.zeros((world_size * M, N), dtype=dtype)
            else:
                out = ctx.zeros((M, N), dtype=dtype)
            inp.fill_(float(rank + 1) * 0.1)

            # Build config
            config = Config()
            workspace = None
            if collective == "all_reduce":
                config = Config(all_reduce_variant="two_shot")
                workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)

            # Define the operation
            def run_op():
                if collective == "all_reduce":
                    ctx.ccl.all_reduce(out, inp, config=config, workspace=workspace)
                elif collective == "all_gather":
                    ctx.ccl.all_gather(out, inp, config=config)
                elif collective == "reduce_scatter":
                    ctx.ccl.reduce_scatter(out, inp, config=config)

            # Warmup
            for _ in range(N_WARMUP):
                out.zero_()
                run_op()
            torch.cuda.synchronize()
            dist.barrier()

            # Timed iterations with CUDA events
            times_ms = []
            for _ in range(N_REPEAT):
                out.zero_()
                torch.cuda.synchronize()
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()
                run_op()
                end_ev.record()
                torch.cuda.synchronize()
                times_ms.append(start_ev.elapsed_time(end_ev))

            dist.barrier()

            # Compute stats
            times_us = sorted([t * 1000.0 for t in times_ms])
            lat_min = min(times_us)
            lat_median = statistics.median(times_us)
            n = len(times_us)
            p99_idx = min(int(math.ceil(0.99 * n)) - 1, n - 1)
            lat_p99 = times_us[p99_idx]

            bus_bytes = _bus_bandwidth_bytes(collective, M, N, world_size)
            mean_ms = statistics.mean(times_ms)
            bw_gbps = (bus_bytes / 1e9) / (mean_ms * 1e-3) if mean_ms > 0 else 0.0

            if rank == 0:
                results.append({
                    "collective": collective,
                    "num_gpus": world_size,
                    "msg_bytes": msg_bytes,
                    "latency_us_min": round(lat_min, 2),
                    "latency_us_median": round(lat_median, 2),
                    "latency_us_p99": round(lat_p99, 2),
                    "bandwidth_GBps": round(bw_gbps, 2),
                    "algorithm": ALGORITHMS.get(collective, "default"),
                })
                print(f"  {_fmt_bytes(msg_bytes):>10s} | "
                      f"lat_med={lat_median:.1f}us | "
                      f"bw={bw_gbps:.1f} GB/s")

        except Exception as e:
            if rank == 0:
                print(f"  ERROR {_fmt_bytes(msg_bytes)}: {e}")
            try:
                dist.barrier()
            except Exception:
                pass
            # Clean up whatever was allocated
            for name in ['inp', 'out', 'workspace']:
                if name in dir():
                    try:
                        exec(f'del {name}')
                    except Exception:
                        pass
            gc.collect()
            torch.cuda.empty_cache()
            continue
        else:
            # Successful -- free tensors explicitly
            del inp, out
            workspace = None
            gc.collect()
            torch.cuda.empty_cache()

    # Write results from rank 0
    if rank == 0 and results and output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Wrote {len(results)} results to {output_file}")

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Driver mode: orchestrate torchrun launches
# ---------------------------------------------------------------------------
def run_driver(args):
    """Launch workers for each (collective, gpu_count) combination."""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).resolve().parent
        output_dir = script_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    final_csv = output_dir / "ccl_sweep_results.csv"
    final_json = output_dir / "ccl_sweep_results.json"
    summary_md = output_dir / "sweep_summary.md"

    all_results = []
    script_path = str(Path(__file__).resolve())

    # Run 8-GPU first (freshest system state, avoids thread exhaustion),
    # then 4, then 2.
    ordered_gpus = sorted(GPU_COUNTS, reverse=True)

    for ngpu in ordered_gpus:
        for collective in COLLECTIVES:
            part_file = output_dir / f"part_{collective}_{ngpu}gpu.json"

            # Skip if we already have results from a previous run
            if part_file.exists():
                with open(part_file) as f:
                    partial = json.load(f)
                if len(partial) > 0:
                    all_results.extend(partial)
                    print(f"\n[cached] {collective} {ngpu}GPU: {len(partial)} results")
                    continue

            print(f"\n{'='*60}")
            print(f"Running {collective} with {ngpu} GPUs")
            print(f"{'='*60}")

            env = os.environ.copy()
            env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
            # Limit threading to avoid pthread_create failures with many ranks
            env["OMP_NUM_THREADS"] = "1"
            env["TRITON_NUM_COMPILER_THREADS"] = "1"
            env["NCCL_NTHREADS"] = "64"

            cmd = [
                sys.executable, "-m", "torch.distributed.run",
                "--standalone",
                "--nproc_per_node", str(ngpu),
                script_path,
                "--worker",
                "--collective", collective,
                "--output", str(part_file),
            ]

            # Try up to 2 attempts
            for attempt in range(2):
                result = subprocess.run(
                    cmd,
                    timeout=600,  # 10 min per combo
                    capture_output=False,
                    env=env,
                )

                if result.returncode == 0 and part_file.exists():
                    break

                if attempt == 0:
                    print(f"  Attempt 1 failed (rc={result.returncode}), retrying...")
                    time.sleep(5)  # Brief pause to let resources clean up

            if result.returncode != 0:
                print(f"WARNING: {collective} {ngpu}GPU failed after 2 attempts (rc={result.returncode})")
                continue

            # Load partial results
            if part_file.exists():
                with open(part_file) as f:
                    partial = json.load(f)
                all_results.extend(partial)
                print(f"Collected {len(partial)} results for {collective} {ngpu}GPU")

    # Sort results
    all_results.sort(key=lambda r: (r["collective"], r["num_gpus"], r["msg_bytes"]))

    # Write CSV
    fieldnames = [
        "collective", "num_gpus", "msg_bytes",
        "latency_us_min", "latency_us_median", "latency_us_p99",
        "bandwidth_GBps", "algorithm",
    ]
    with open(final_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Write JSON
    with open(final_json, "w") as f:
        json.dump(all_results, f, indent=2)
        f.write("\n")

    # Generate summary
    _generate_summary(all_results, str(summary_md))

    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Total data points: {len(all_results)}")
    print(f"CSV: {final_csv}")
    print(f"JSON: {final_json}")
    print(f"Summary: {summary_md}")

    # Coverage report
    collectives = set(r["collective"] for r in all_results)
    gpu_counts = set(r["num_gpus"] for r in all_results)
    msg_sizes = set(r["msg_bytes"] for r in all_results)
    print(f"\nCoverage:")
    print(f"  Collectives: {sorted(collectives)}")
    print(f"  GPU counts: {sorted(gpu_counts)}")
    print(f"  Message sizes: {len(msg_sizes)} unique")
    print(f"  Total configs: {len(all_results)}")
    print(f"  Pass (>= 270): {len(all_results) >= 270}")


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------
def _generate_summary(rows: list[dict], summary_path: str) -> None:
    collectives = sorted(set(r["collective"] for r in rows))
    gpu_counts = sorted(set(r["num_gpus"] for r in rows))
    msg_sizes = sorted(set(r["msg_bytes"] for r in rows))

    lines = [
        "# CCL Sweep Results Summary",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**GPU**: MI300X",
        f"**Dtype**: bfloat16",
        "",
        "## Coverage",
        "",
        f"- **Collectives**: {', '.join(collectives)}",
        f"- **GPU counts**: {', '.join(str(g) for g in gpu_counts)}",
        f"- **Message sizes**: {len(msg_sizes)} unique sizes ({_fmt_bytes(min(msg_sizes))} to {_fmt_bytes(max(msg_sizes))})",
        f"- **Total configurations**: {len(rows)}",
        "",
        "## Message Size Grid",
        "",
    ]
    for sz in msg_sizes:
        lines.append(f"- {_fmt_bytes(sz)} ({sz:,} bytes)")

    for coll in collectives:
        lines.append("")
        lines.append(f"## {coll}")
        lines.append("")
        lines.append("| GPUs | Msg Size | Latency min (us) | Latency median (us) | Latency p99 (us) | BW (GB/s) |")
        lines.append("|------|----------|-------------------|---------------------|-------------------|-----------|")
        coll_rows = [r for r in rows if r["collective"] == coll]
        for r in coll_rows:
            lines.append(
                f"| {r['num_gpus']} | {_fmt_bytes(r['msg_bytes'])} "
                f"| {r['latency_us_min']:.1f} | {r['latency_us_median']:.1f} "
                f"| {r['latency_us_p99']:.1f} | {r['bandwidth_GBps']:.1f} |"
            )

    lines.extend(["", "## Key Observations", ""])
    for coll in collectives:
        coll_rows = [r for r in rows if r["collective"] == coll and r["num_gpus"] == 8]
        if coll_rows:
            best = max(coll_rows, key=lambda r: r["bandwidth_GBps"])
            smallest = min(coll_rows, key=lambda r: r["msg_bytes"])
            lines.append(
                f"- **{coll} (8 GPU)**: Peak BW = {best['bandwidth_GBps']:.1f} GB/s "
                f"at {_fmt_bytes(best['msg_bytes'])}; "
                f"latency floor = {smallest['latency_us_median']:.1f} us "
                f"at {_fmt_bytes(smallest['msg_bytes'])}"
            )
    lines.append("")

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------
def _dry_run():
    print("=" * 60)
    print("CCL Sweep -- Dry Run")
    print("=" * 60)
    print(f"\nCollectives: {COLLECTIVES}")
    print(f"GPU counts: {GPU_COUNTS}")
    print(f"Message sizes ({len(MSG_SIZES)}):")
    for sz in MSG_SIZES:
        M, N = _msg_size_to_mn(sz)
        actual = _actual_bytes(M, N)
        print(f"  {_fmt_bytes(sz):>10s} ({sz:>12,} bytes) -> shape ({M}, {N}), actual {actual:,} bytes")

    total = len(COLLECTIVES) * len(GPU_COUNTS) * len(MSG_SIZES)
    print(f"\nTotal configurations: {total}")
    print(f"Minimum required: 270")
    print(f"Coverage: {'PASS' if total >= 270 else 'FAIL'} ({total} >= 270)")
    print(f"\nDtype: {DTYPE_NAME}")
    print(f"Warmup: {N_WARMUP} iterations")
    print(f"Repeat: {N_REPEAT} timed iterations")
    print("=" * 60)
    print("\nDry run complete -- no GPU operations performed.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CCL latency/bandwidth sweep")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true",
                        help="Worker mode (launched via torchrun)")
    parser.add_argument("--collective", type=str,
                        choices=COLLECTIVES)
    parser.add_argument("--output", type=str,
                        help="Output JSON file (worker mode)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (driver mode)")
    args = parser.parse_args()

    if args.dry_run:
        sys.exit(_dry_run())
    elif args.worker:
        if not args.collective:
            print("ERROR: --collective required in worker mode", file=sys.stderr)
            sys.exit(1)
        run_worker(args)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
