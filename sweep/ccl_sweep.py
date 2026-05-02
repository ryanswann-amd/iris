#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
CCL parameter sweep for cost model construction.

Sweeps collective operations across message sizes, GPU counts, comm_sms values,
block sizes, and algorithm variants. Produces a CSV dataset suitable for fitting
a latency/bandwidth cost model.

Supports:
  --dry-run   Print the parameter grid without executing benchmarks
  --resume    Resume from an existing partial CSV (checkpoint)
  --subset    Run a fraction of the grid for testing
"""

import argparse
import csv
import itertools
import json
import math
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Parameter space definition
# ---------------------------------------------------------------------------

# Message sizes: tensor shapes that yield ~1KB to ~1GB in FP16 (2 bytes/elem)
# For small messages: M=32 (min block_size_m), vary N from 16 upward
# For large messages: M=8192, vary N
MSG_SHAPES = []
for log_total in range(9, 30):  # 2^9=512 elems (~1KB) to 2^29=512M elems (~1GB)
    total_elems = 1 << log_total
    # Try to make roughly square tensors, but respect minimum block sizes
    # M must be >= 32 (block_size_m), N must be >= 16 (min block_size_n)
    for m_candidate in [32, 128, 512, 2048, 8192]:
        if m_candidate > total_elems:
            continue
        n_candidate = total_elems // m_candidate
        if n_candidate < 16:
            continue
        MSG_SHAPES.append((m_candidate, n_candidate))
        break  # take first valid shape per total size

# De-duplicate and sort
MSG_SHAPES = sorted(set(MSG_SHAPES))

NUM_GPUS_LIST = [2, 4, 8]
COMM_SMS_LIST = [8, 16, 32, 64, 96, 128]
BLOCK_SIZE_N_LIST = [16, 32, 64, 128]
BLOCK_SIZE_M = 32  # Fixed - standard value
SWIZZLE_SIZE = 4   # Fixed - standard value
DTYPE = "fp16"

# Operation-specific variants
ALL_REDUCE_VARIANTS = ["two_shot", "atomic", "one_shot"]
# Ring excluded from default sweep: requires block_size_n divisible by world_size
# and ring_slice_n constraints make it fragile in automated sweeps

ALL_GATHER_VARIANTS = ["persistent", "partitioned"]

# Operations with no variant selection
ALL_TO_ALL_VARIANTS = [None]
REDUCE_SCATTER_VARIANTS = [None]


@dataclass
class SweepPoint:
    """A single parameter combination to benchmark."""
    op: str
    m: int
    n: int
    num_gpus: int
    comm_sms: int
    block_size_m: int
    block_size_n: int
    swizzle_size: int
    dtype: str
    variant: Optional[str] = None
    distribution: Optional[int] = None

    @property
    def msg_bytes(self) -> int:
        elem_size = 2 if self.dtype == "fp16" else 4 if self.dtype == "fp32" else 2
        return self.m * self.n * elem_size

    @property
    def key(self) -> str:
        return f"{self.op}|{self.m}|{self.n}|{self.num_gpus}|{self.comm_sms}|{self.block_size_m}|{self.block_size_n}|{self.variant}|{self.distribution}"

    def to_csv_row(self, latency_ms: float, bandwidth_gbps: float,
                   success: bool = True) -> dict:
        return {
            "op": self.op,
            "m": self.m,
            "n": self.n,
            "msg_bytes": self.msg_bytes,
            "num_gpus": self.num_gpus,
            "comm_sms": self.comm_sms,
            "block_size_m": self.block_size_m,
            "block_size_n": self.block_size_n,
            "swizzle_size": self.swizzle_size,
            "dtype": self.dtype,
            "variant": self.variant or "",
            "distribution": self.distribution if self.distribution is not None else "",
            "latency_ms": f"{latency_ms:.6f}",
            "bandwidth_gbps": f"{bandwidth_gbps:.3f}",
            "success": str(success),
        }


def generate_sweep_points() -> list[SweepPoint]:
    """Generate the full parameter grid."""
    points = []

    for (m, n), num_gpus, comm_sms, bsn in itertools.product(
        MSG_SHAPES, NUM_GPUS_LIST, COMM_SMS_LIST, BLOCK_SIZE_N_LIST
    ):
        # Constraint: block_size_n must divide n (or be <= n for partial tiles)
        if bsn > n:
            continue
        # Constraint: block_size_m must divide m
        if BLOCK_SIZE_M > m:
            continue

        # all_reduce
        for variant in ALL_REDUCE_VARIANTS:
            for dist_mode in [0, 1] if variant == "two_shot" else [None]:
                points.append(SweepPoint(
                    op="all_reduce", m=m, n=n, num_gpus=num_gpus,
                    comm_sms=comm_sms, block_size_m=BLOCK_SIZE_M,
                    block_size_n=bsn, swizzle_size=SWIZZLE_SIZE,
                    dtype=DTYPE, variant=variant,
                    distribution=dist_mode,
                ))

        # all_gather
        for variant in ALL_GATHER_VARIANTS:
            points.append(SweepPoint(
                op="all_gather", m=m, n=n, num_gpus=num_gpus,
                comm_sms=comm_sms, block_size_m=BLOCK_SIZE_M,
                block_size_n=bsn, swizzle_size=SWIZZLE_SIZE,
                dtype=DTYPE, variant=variant,
            ))

        # all_to_all
        # Constraint: n must be divisible by num_gpus for all_to_all
        if n % num_gpus == 0:
            points.append(SweepPoint(
                op="all_to_all", m=m, n=n, num_gpus=num_gpus,
                comm_sms=comm_sms, block_size_m=BLOCK_SIZE_M,
                block_size_n=bsn, swizzle_size=SWIZZLE_SIZE,
                dtype=DTYPE,
            ))

        # reduce_scatter
        points.append(SweepPoint(
            op="reduce_scatter", m=m, n=n, num_gpus=num_gpus,
            comm_sms=comm_sms, block_size_m=BLOCK_SIZE_M,
            block_size_n=bsn, swizzle_size=SWIZZLE_SIZE,
            dtype=DTYPE, variant="two_shot",
        ))

    return points


def load_completed_keys(csv_path: str) -> set[str]:
    """Load keys of already-completed sweep points from checkpoint CSV."""
    keys = set()
    if not os.path.exists(csv_path):
        return keys
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (f"{row['op']}|{row['m']}|{row['n']}|{row['num_gpus']}|"
                   f"{row['comm_sms']}|{row['block_size_m']}|{row['block_size_n']}|"
                   f"{row['variant']}|{row['distribution']}")
            keys.add(key)
    return keys


CSV_FIELDS = [
    "op", "m", "n", "msg_bytes", "num_gpus", "comm_sms",
    "block_size_m", "block_size_n", "swizzle_size", "dtype",
    "variant", "distribution", "latency_ms", "bandwidth_gbps", "success",
]


def run_benchmark_point(point: SweepPoint, iris_root: str,
                        heap_size: int = 1 << 34) -> Optional[dict]:
    """Run a single benchmark point as a subprocess."""
    script_map = {
        "all_gather": "benchmark/ccl/all_gather/benchmark.py",
        "all_reduce": "benchmark/ccl/all_reduce/benchmark.py",
        "all_to_all": "benchmark/ccl/all_to_all/benchmark.py",
        "reduce_scatter": "benchmark/ccl/reduce_scatter/benchmark.py",
    }

    script_path = os.path.join(iris_root, script_map[point.op])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_output = f.name

    cmd = [
        sys.executable, script_path,
        "-m", str(point.m),
        "-n", str(point.n),
        "--datatype", point.dtype,
        "--comm_sms", str(point.comm_sms),
        "-r", str(point.num_gpus),
        "--heap_size", str(heap_size),
        "--benchmark",
        "--output_file", temp_output,
        "--block_size_m", str(point.block_size_m),
        "--block_size_n", str(point.block_size_n),
        "--swizzle_size", str(point.swizzle_size),
    ]

    # Operation-specific args
    if point.op == "all_reduce":
        cmd.extend(["--variant", point.variant])
        if point.variant == "two_shot" and point.distribution is not None:
            cmd.extend(["--distribution", str(point.distribution)])
    elif point.op == "all_gather":
        if point.variant:
            cmd.extend(["--variant", point.variant])
    elif point.op == "reduce_scatter":
        if point.distribution is not None:
            cmd.extend(["--all_reduce_distribution", str(point.distribution)])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            cwd=iris_root,
        )
        if result.returncode != 0:
            print(f"  FAIL ({point.op} M={point.m} N={point.n} "
                  f"gpus={point.num_gpus} sms={point.comm_sms}): "
                  f"exit {result.returncode}", flush=True)
            if os.path.exists(temp_output):
                os.unlink(temp_output)
            return None

        with open(temp_output, "r") as f:
            data = json.load(f)
        os.unlink(temp_output)
        return data
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT ({point.op} M={point.m} N={point.n})", flush=True)
        if os.path.exists(temp_output):
            os.unlink(temp_output)
        return None
    except Exception as e:
        print(f"  ERROR ({point.op}): {e}", flush=True)
        if os.path.exists(temp_output):
            os.unlink(temp_output)
        return None


def extract_results(point: SweepPoint, data: dict) -> dict:
    """Extract latency and bandwidth from benchmark output JSON."""
    op_key = f"{point.op}_ms"
    latency_ms = data.get(op_key, data.get("total_ms", 0.0))
    bandwidth_gbps = data.get("bandwidth_gbps", 0.0)
    success = data.get("success", True)
    return point.to_csv_row(latency_ms, bandwidth_gbps, success)


def main():
    parser = argparse.ArgumentParser(
        description="CCL parameter sweep for cost model construction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print parameter grid without executing")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint CSV")
    parser.add_argument("--subset", type=float, default=1.0,
                        help="Fraction of grid to run (0.0-1.0)")
    parser.add_argument("--output", type=str,
                        default="sweep/results/ccl_sweep_results.csv",
                        help="Output CSV path (relative to iris root)")
    parser.add_argument("--heap-size", type=int, default=1 << 34,
                        help="Iris heap size in bytes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subset sampling")
    args = parser.parse_args()

    # Determine iris root
    iris_root = str(Path(__file__).resolve().parent.parent)
    output_path = os.path.join(iris_root, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Generate full grid
    all_points = generate_sweep_points()
    print(f"Full parameter grid: {len(all_points)} combinations")

    # Show grid summary in dry-run mode
    if args.dry_run:
        ops = {}
        for p in all_points:
            ops.setdefault(p.op, 0)
            ops[p.op] += 1
        print(f"\nGrid breakdown by operation:")
        for op, count in sorted(ops.items()):
            print(f"  {op}: {count}")

        msg_sizes = sorted(set(p.msg_bytes for p in all_points))
        print(f"\nMessage sizes: {len(msg_sizes)} unique")
        print(f"  Min: {msg_sizes[0]:,} bytes ({msg_sizes[0]/1024:.0f} KB)")
        print(f"  Max: {msg_sizes[-1]:,} bytes ({msg_sizes[-1]/(1024**3):.2f} GB)")

        print(f"\nGPU counts: {sorted(set(p.num_gpus for p in all_points))}")
        print(f"comm_sms values: {sorted(set(p.comm_sms for p in all_points))}")
        print(f"block_size_n values: {sorted(set(p.block_size_n for p in all_points))}")

        variants = {}
        for p in all_points:
            if p.variant:
                variants.setdefault(p.op, set()).add(p.variant)
        print(f"\nVariants:")
        for op, vs in sorted(variants.items()):
            print(f"  {op}: {sorted(vs)}")

        print(f"\nTotal unique combinations: {len(all_points)}")
        print(f"\nCSV columns: {', '.join(CSV_FIELDS)}")
        print(f"\nOutput would be written to: {output_path}")
        return

    # Resume: filter out already-completed points
    completed_keys = set()
    if args.resume:
        completed_keys = load_completed_keys(output_path)
        print(f"Resuming: {len(completed_keys)} points already completed")

    pending = [p for p in all_points if p.key not in completed_keys]

    # Subset sampling
    if args.subset < 1.0:
        random.seed(args.seed)
        k = max(1, int(len(pending) * args.subset))
        pending = random.sample(pending, k)
        print(f"Subset mode: running {len(pending)} of {len(all_points)} points")

    print(f"Points to run: {len(pending)}")

    # Open CSV in append mode
    write_header = not os.path.exists(output_path) or not args.resume
    csvfile = open(output_path, "a" if args.resume else "w", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    # Handle graceful shutdown
    shutdown = False
    def handle_signal(sig, frame):
        nonlocal shutdown
        print("\nShutdown requested, finishing current point...")
        shutdown = True
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    completed = 0
    failed = 0
    start_time = time.time()

    for i, point in enumerate(pending):
        if shutdown:
            break

        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (len(pending) - i) / rate if rate > 0 else 0

        print(f"\n[{i+1}/{len(pending)}] {point.op} M={point.m} N={point.n} "
              f"gpus={point.num_gpus} sms={point.comm_sms} bsn={point.block_size_n} "
              f"var={point.variant or '-'} "
              f"(ETA: {eta/60:.0f}m)", flush=True)

        data = run_benchmark_point(point, iris_root, args.heap_size)
        if data is not None:
            row = extract_results(point, data)
            writer.writerow(row)
            csvfile.flush()
            completed += 1
            print(f"  OK: {row['latency_ms']}ms, {row['bandwidth_gbps']} GB/s",
                  flush=True)
        else:
            # Record failure with zero metrics
            row = point.to_csv_row(0.0, 0.0, success=False)
            writer.writerow(row)
            csvfile.flush()
            failed += 1

    csvfile.close()

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Sweep complete: {completed} succeeded, {failed} failed")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
