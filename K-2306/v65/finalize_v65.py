#!/usr/bin/env python3
"""K-2306 finalizer: aggregates per-rank CSVs into combined parquet/CSV
and produces the summary plots required by the task spec.

Usage: python3 finalize_v65.py --in-dir <dir with rank*.csv> --out-dir <dir>
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="v65_J4_baseline")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rank_csvs = sorted(in_dir.glob(f"{args.prefix}.rank*.csv"))
    if not rank_csvs:
        raise SystemExit(f"no rank csvs found under {in_dir}/{args.prefix}.rank*.csv")
    df = pd.concat([pd.read_csv(p) for p in rank_csvs], ignore_index=True)
    print(f"[finalize] loaded {len(df)} rows from {len(rank_csvs)} ranks")

    # Combined CSV + parquet
    df.to_csv(out_dir / f"{args.prefix}.csv", index=False)
    try:
        df.to_parquet(out_dir / f"{args.prefix}.parquet", index=False)
    except Exception as e:
        print(f"[finalize] parquet skipped: {e}")

    # Aggregate by (scope, block_size, dtype) — median/p99 across pairs+reps
    agg = df.groupby(["scope", "block_size", "dtype"]).agg(
        time_ms_median=("time_ms", "median"),
        time_ms_p99=("time_ms", lambda x: np.percentile(x, 99)),
        bw_gibps_median=("bandwidth_gibps", "median"),
        bw_gibps_p99=("bandwidth_gibps", lambda x: np.percentile(x, 99)),
        n_rows=("rep_idx", "size"),
    ).reset_index()
    agg.to_csv(out_dir / f"{args.prefix}_agg_by_scope_bs_dtype.csv", index=False)

    # Local vs remote (same_gpu)
    df["same_gpu"] = df["src_rank"] == df["dst_rank"]
    agg_lr = df.groupby(["same_gpu", "scope", "dtype"]).agg(
        time_ms_median=("time_ms", "median"),
        bw_gibps_median=("bandwidth_gibps", "median"),
        n_rows=("rep_idx", "size"),
    ).reset_index()
    agg_lr.to_csv(out_dir / f"{args.prefix}_agg_by_local_remote.csv", index=False)

    # PLOT 1 — Time by scope/block (boxplot)
    fig, ax = plt.subplots(figsize=(10, 5))
    grouped = []
    labels = []
    for sc in ("cta", "gpu", "sys"):
        for bs in (256, 1024, 4096):
            grouped.append(df[(df.scope == sc) & (df.block_size == bs)]["time_ms"].values)
            labels.append(f"{sc}\nBS={bs}")
    ax.boxplot(grouped, labels=labels, showfliers=False)
    ax.set_ylabel("time (ms)")
    ax.set_title(f"K-2306 J4 ATOMIC_MAX_RELEASE — time by scope×block_size")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "p1_time_by_scope_bs.png", dpi=120)
    plt.close(fig)

    # PLOT 2 — BW by scope/block (boxplot)
    fig, ax = plt.subplots(figsize=(10, 5))
    grouped = []
    labels = []
    for sc in ("cta", "gpu", "sys"):
        for bs in (256, 1024, 4096):
            grouped.append(df[(df.scope == sc) & (df.block_size == bs)]["bandwidth_gibps"].values)
            labels.append(f"{sc}\nBS={bs}")
    ax.boxplot(grouped, labels=labels, showfliers=False)
    ax.set_ylabel("bandwidth (GiB/s)")
    ax.set_title(f"K-2306 J4 ATOMIC_MAX_RELEASE — bandwidth by scope×block_size")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "p2_bw_by_scope_bs.png", dpi=120)
    plt.close(fig)

    # PLOT 3 — pair heatmap (median time, scope=gpu, bs=1024, dtype=int32)
    sub = df[(df.scope == "gpu") & (df.block_size == 1024) & (df.dtype == "int32")]
    pivot = sub.groupby(["src_rank", "dst_rank"])["time_ms"].median().unstack(fill_value=np.nan)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(8)); ax.set_xticklabels(range(8))
    ax.set_yticks(range(8)); ax.set_yticklabels(range(8))
    ax.set_xlabel("dst_rank"); ax.set_ylabel("src_rank")
    ax.set_title("K-2306 J4 — pair median time (ms), scope=gpu BS=1024 int32")
    plt.colorbar(im, ax=ax, label="time (ms)")
    plt.tight_layout()
    fig.savefig(out_dir / "p3_pair_heatmap.png", dpi=120)
    plt.close(fig)

    # Manifest
    manifest = {
        "task": "K-2306",
        "primitive": "J4_ATOMIC_MAX_RELEASE",
        "version": "v65",
        "n_rows": int(len(df)),
        "n_cells": int(df.groupby(["scope", "block_size", "dtype", "src_rank", "dst_rank"]).ngroups),
        "n_ranks": int(df["world_size"].iloc[0]),
        "gpu_arch": df["gpu_arch"].iloc[0],
        "hostname": df["hostname"].unique().tolist(),
        "run_id": df["run_id"].iloc[0],
        "bandwidth_gibps_median": float(df["bandwidth_gibps"].median()),
        "time_ms_median": float(df["time_ms"].median()),
        "outputs": {
            "csv": f"{args.prefix}.csv",
            "parquet": f"{args.prefix}.parquet",
            "agg_scope_bs_dtype": f"{args.prefix}_agg_by_scope_bs_dtype.csv",
            "agg_local_remote": f"{args.prefix}_agg_by_local_remote.csv",
            "plots": ["p1_time_by_scope_bs.png", "p2_bw_by_scope_bs.png", "p3_pair_heatmap.png"],
        },
    }
    (out_dir / "v65_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[finalize] wrote outputs to {out_dir}")
    print(f"[finalize] median bw = {manifest['bandwidth_gibps_median']:.2f} GiB/s, "
          f"median lat = {manifest['time_ms_median']:.4f} ms")


if __name__ == "__main__":
    main()
