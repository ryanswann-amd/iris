#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Validate sweep coverage: checks that the CSV dataset contains at least
--min-combos unique parameter combinations across required dimensions.
"""

import argparse
import csv
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Validate CCL sweep coverage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--min-combos", type=int, default=500,
                        help="Minimum number of unique parameter combinations")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to sweep CSV (default: sweep/results/ccl_sweep_results.csv)")
    args = parser.parse_args()

    # Default CSV path
    if args.csv is None:
        iris_root = str(Path(__file__).resolve().parent.parent)
        csv_path = os.path.join(iris_root, "sweep", "results", "ccl_sweep_results.csv")
    else:
        csv_path = args.csv

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    # Read CSV
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("ERROR: CSV file is empty")
        sys.exit(1)

    # Count unique combinations using the key dimensions
    # Key: (op, m, n, num_gpus, comm_sms, block_size_n, variant, distribution)
    unique_combos = set()
    for row in rows:
        key = (
            row.get("op", ""),
            row.get("m", ""),
            row.get("n", ""),
            row.get("num_gpus", ""),
            row.get("comm_sms", ""),
            row.get("block_size_n", ""),
            row.get("variant", ""),
            row.get("distribution", ""),
        )
        unique_combos.add(key)

    n_combos = len(unique_combos)

    # Compute coverage statistics
    ops = set(row.get("op", "") for row in rows)
    msg_sizes = set(int(row.get("msg_bytes", 0)) for row in rows)
    gpu_counts = set(int(row.get("num_gpus", 0)) for row in rows)
    comm_sms_vals = set(int(row.get("comm_sms", 0)) for row in rows)
    bsn_vals = set(int(row.get("block_size_n", 0)) for row in rows)
    successful = sum(1 for row in rows if row.get("success", "True") == "True")

    print(f"CCL Sweep Coverage Report")
    print(f"{'='*50}")
    print(f"Total rows:              {len(rows)}")
    print(f"Unique combinations:     {n_combos}")
    print(f"Successful benchmarks:   {successful}")
    print(f"Failed benchmarks:       {len(rows) - successful}")
    print(f"")
    print(f"Operations:              {sorted(ops)}")
    print(f"Message sizes:           {len(msg_sizes)} unique")
    print(f"GPU counts:              {sorted(gpu_counts)}")
    print(f"comm_sms values:         {sorted(comm_sms_vals)}")
    print(f"block_size_n values:     {sorted(bsn_vals)}")
    print(f"")

    if n_combos >= args.min_combos:
        print(f"PASS: {n_combos} >= {args.min_combos} unique combinations")
        sys.exit(0)
    else:
        print(f"FAIL: {n_combos} < {args.min_combos} unique combinations")
        sys.exit(1)


if __name__ == "__main__":
    main()
