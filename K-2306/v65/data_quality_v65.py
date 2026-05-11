#!/usr/bin/env python3
"""K-2306 data quality validator (v65 schema, matches K-2303 J4 + K-2289 F4).

Schema (per-rep CSV columns):
  run_id, primitive_id, primitive_name, sem, scope, block_size, dtype,
  src_rank, dst_rank, buffer_bytes, n_elements, rep_idx, time_ms,
  bandwidth_gibps, world_size, gpu_arch, hostname, ts_unix

Grid: 3 scopes x 3 block_sizes x 2 dtypes x 64 (src,dst) pairs = 1,152 cells.
Reps: 25 per cell. Total expected rows: 28,800.

PASS:
  - row count >= 28,000 (>=97% of 28,800)
  - 0% null bandwidth_gibps
  - 0% zero-time / zero-bw rows
  - 0% negative bandwidth_gibps
  - 1,152 (scope, block_size, dtype, src_rank, dst_rank) cells present
  - bandwidth_gibps in [0.001, 2000.0]
  - all rows have primitive_id == J4 and sem == release

Usage: python3 data_quality_v65.py <corpus.csv ...>
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

EXPECTED_CELLS = 1152
EXPECTED_REPS = 25
EXPECTED_ROWS = EXPECTED_CELLS * EXPECTED_REPS  # 28,800


def _load_many(paths):
    dfs = []
    for p in paths:
        p = Path(p)
        if p.suffix == ".parquet":
            dfs.append(pd.read_parquet(p))
        else:
            dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="+")
    args = ap.parse_args()

    df = _load_many(args.corpus)
    fails = []

    # row count
    rc = len(df)
    if rc < int(EXPECTED_ROWS * 0.97):
        fails.append(f"row count {rc} < 97% of expected {EXPECTED_ROWS}")

    # null bw
    null_pct = df["bandwidth_gibps"].isna().mean() * 100
    if null_pct > 0:
        fails.append(f"null bandwidth_gibps {null_pct:.2f}% (expected 0%)")

    # zero-time
    zero = ((df["time_ms"] == 0) & (df["bandwidth_gibps"] == 0)).mean() * 100
    if zero > 0:
        fails.append(f"zero-time/zero-bw rows {zero:.2f}% (expected 0%)")

    # negative bw
    neg_pct = (df["bandwidth_gibps"] < 0).mean() * 100
    if neg_pct > 0:
        fails.append(f"negative bandwidth_gibps {neg_pct:.2f}% (expected 0%)")

    # bw range
    bw_min, bw_max = df["bandwidth_gibps"].min(), df["bandwidth_gibps"].max()
    if bw_min < 0.001 or bw_max > 2000.0:
        fails.append(f"bandwidth_gibps range [{bw_min:.4f}, {bw_max:.2f}] outside [0.001, 2000.0]")

    # cell coverage (scope x bs x dtype x src x dst)
    cells = df.groupby(["scope", "block_size", "dtype", "src_rank", "dst_rank"]).size().shape[0]
    if cells != EXPECTED_CELLS:
        fails.append(f"cell coverage {cells} != expected {EXPECTED_CELLS}")

    # primitive id / sem
    if not (df["primitive_id"] == "J4").all():
        fails.append(f"primitive_id not all J4: {df['primitive_id'].unique().tolist()}")
    if not (df["sem"] == "release").all():
        fails.append(f"sem not all release: {df['sem'].unique().tolist()}")

    # rep coverage per cell
    rep_counts = df.groupby(["scope", "block_size", "dtype", "src_rank", "dst_rank"]).size()
    bad_reps = (rep_counts != EXPECTED_REPS).sum()
    if bad_reps > 0:
        fails.append(f"{bad_reps} cells have rep count != {EXPECTED_REPS}")

    # gpu_arch sanity
    archs = df["gpu_arch"].unique().tolist()
    nodes = df["hostname"].nunique()
    print(f"[INFO] rows={rc} cells={cells} nodes={nodes} archs={archs}")
    print(f"[INFO] bw_GiBps median={df['bandwidth_gibps'].median():.2f} min={bw_min:.4f} max={bw_max:.2f}")
    print(f"[INFO] time_ms median={df['time_ms'].median():.4f}")

    if fails:
        print("[FAIL] data_quality_v65 — issues:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("[PASS] data_quality_v65 — all checks green")


if __name__ == "__main__":
    main()
