#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Validate CCL sweep coverage and data quality.
#
# Reads ccl_sweep_results.csv and checks:
#   1. Full matrix coverage (>= 270 unique configs)
#   2. All three collectives present
#   3. All three GPU counts present (2, 4, 8)
#   4. Bandwidth values non-zero for messages >= 64KB
#   5. Latency values are positive and sensible

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load_results(csv_path: str) -> list[dict]:
    """Load the CSV and parse numeric fields."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "collective": row["collective"],
                    "num_gpus": int(row["num_gpus"]),
                    "msg_bytes": int(row["msg_bytes"]),
                    "latency_us_min": float(row["latency_us_min"]),
                    "latency_us_median": float(row["latency_us_median"]),
                    "latency_us_p99": float(row["latency_us_p99"]),
                    "bandwidth_GBps": float(row["bandwidth_GBps"]),
                    "algorithm": row["algorithm"],
                }
            )
    return rows


def validate_coverage(rows: list[dict], min_configs: int) -> tuple[bool, list[str]]:
    """Check that the matrix has sufficient coverage."""
    errors = []

    # Unique configs
    configs = set((r["collective"], r["num_gpus"], r["msg_bytes"]) for r in rows)
    n_configs = len(configs)

    if n_configs < min_configs:
        errors.append(f"FAIL: Only {n_configs} unique configs, need >= {min_configs}")
    else:
        print(f"PASS: {n_configs} unique configs >= {min_configs}")

    # Check all three collectives
    collectives = set(r["collective"] for r in rows)
    expected_colls = {"all_reduce", "all_gather", "reduce_scatter"}
    missing_colls = expected_colls - collectives
    if missing_colls:
        errors.append(f"FAIL: Missing collectives: {missing_colls}")
    else:
        print(f"PASS: All 3 collectives present: {sorted(collectives)}")

    # Check GPU counts
    gpu_counts = set(r["num_gpus"] for r in rows)
    expected_gpus = {2, 4, 8}
    missing_gpus = expected_gpus - gpu_counts
    if missing_gpus:
        errors.append(f"FAIL: Missing GPU counts: {missing_gpus}")
    else:
        print(f"PASS: All 3 GPU counts present: {sorted(gpu_counts)}")

    # Check message sizes
    msg_sizes = sorted(set(r["msg_bytes"] for r in rows))
    print(f"INFO: {len(msg_sizes)} unique message sizes")
    if len(msg_sizes) < 30:
        errors.append(f"FAIL: Only {len(msg_sizes)} message sizes, need >= 30")
    else:
        print(f"PASS: {len(msg_sizes)} message sizes >= 30")

    # Check per-collective coverage
    for coll in expected_colls:
        coll_rows = [r for r in rows if r["collective"] == coll]
        coll_gpus = set(r["num_gpus"] for r in coll_rows)
        coll_sizes = set(r["msg_bytes"] for r in coll_rows)
        print(f"  {coll}: {len(coll_rows)} configs, gpus={sorted(coll_gpus)}, sizes={len(coll_sizes)}")

    return len(errors) == 0, errors


def validate_bandwidth(rows: list[dict]) -> tuple[bool, list[str]]:
    """Check bandwidth values are non-zero for messages >= 64KB."""
    errors = []
    threshold = 64 * 1024  # 64 KB

    large_rows = [r for r in rows if r["msg_bytes"] >= threshold]
    if not large_rows:
        errors.append("FAIL: No rows with msg_bytes >= 64KB")
        return False, errors

    zero_bw = [r for r in large_rows if r["bandwidth_GBps"] <= 0.0]
    if zero_bw:
        errors.append(f"FAIL: {len(zero_bw)} rows with msg_bytes >= 64KB have zero bandwidth")
        for r in zero_bw[:5]:
            errors.append(f"  {r['collective']} gpus={r['num_gpus']} msg={r['msg_bytes']} bw={r['bandwidth_GBps']}")
    else:
        print(f"PASS: All {len(large_rows)} rows with msg_bytes >= 64KB have positive bandwidth")

    # Sanity check: bandwidth should be <= 1000 GB/s (MI300X theoretical max ~800 GB/s per link)
    insane_bw = [r for r in large_rows if r["bandwidth_GBps"] > 1000]
    if insane_bw:
        print(f"WARNING: {len(insane_bw)} rows have bandwidth > 1000 GB/s (may indicate measurement error)")

    return len(errors) == 0, errors


def validate_latency(rows: list[dict]) -> tuple[bool, list[str]]:
    """Check latency values are positive and sensible."""
    errors = []

    for field_name in ["latency_us_min", "latency_us_median", "latency_us_p99"]:
        negative = [r for r in rows if r[field_name] <= 0]
        if negative:
            errors.append(f"FAIL: {len(negative)} rows have non-positive {field_name}")
        else:
            print(f"PASS: All {field_name} values are positive")

    # Check ordering: min <= median <= p99
    ordering_violations = [
        r for r in rows if not (r["latency_us_min"] <= r["latency_us_median"] <= r["latency_us_p99"])
    ]
    if ordering_violations:
        errors.append(f"FAIL: {len(ordering_violations)} rows violate min <= median <= p99 ordering")
    else:
        print("PASS: All rows satisfy min <= median <= p99")

    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Validate CCL sweep results coverage and data quality")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to ccl_sweep_results.csv (auto-detected if not set)",
    )
    parser.add_argument(
        "--min-configs",
        type=int,
        default=270,
        help="Minimum number of unique (collective, gpu_count, msg_size) configs",
    )
    parser.add_argument(
        "--check-bandwidth",
        action="store_true",
        help="Also validate bandwidth column for large messages",
    )
    args = parser.parse_args()

    # Find CSV
    if args.csv:
        csv_path = args.csv
    else:
        # Auto-detect relative to this script
        script_dir = Path(__file__).resolve().parent
        # Check for results/ subdir first (repo layout), then parent/output (workspace)
        candidates = [
            script_dir / "results" / "ccl_sweep_results.csv",
            script_dir.parent / "output" / "ccl_sweep_results.csv",
        ]
        csv_path = None
        for c in candidates:
            if c.exists():
                csv_path = str(c)
                break
        if csv_path is None:
            csv_path = str(candidates[0])  # Will fail with clear message

    if not Path(csv_path).exists():
        print(f"ERROR: CSV not found at {csv_path}")
        sys.exit(1)

    print(f"Validating: {csv_path}")
    print("=" * 60)

    rows = load_results(csv_path)
    print(f"Loaded {len(rows)} rows\n")

    all_pass = True

    # Coverage check
    print("--- Coverage Check ---")
    ok, errs = validate_coverage(rows, args.min_configs)
    if not ok:
        all_pass = False
        for e in errs:
            print(e)

    # Latency check
    print("\n--- Latency Check ---")
    ok, errs = validate_latency(rows)
    if not ok:
        all_pass = False
        for e in errs:
            print(e)

    # Bandwidth check (always run, but report separately)
    print("\n--- Bandwidth Check ---")
    ok, errs = validate_bandwidth(rows)
    if not ok:
        if args.check_bandwidth:
            all_pass = False
        for e in errs:
            print(e)

    print("\n" + "=" * 60)
    if all_pass:
        print("OVERALL: ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("OVERALL: SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
