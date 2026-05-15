#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Plot SP-001 sprint evidence: iris.ccl.all_reduce vs RCCL on MI300X (fp16, ws8).

Reads the comprehensive_sweep.py CSV format and emits a single PNG with two
panels: bus-bandwidth GB/s (log-x) and the iris/rccl latency ratio (log-x,
horizontal line at 1.10 = 10% acceptance gate).
"""

import csv
import sys
from collections import defaultdict

import matplotlib.pyplot as plt


def load(csv_path):
    rows = defaultdict(dict)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (row["collective"], row["dtype"], int(row["total_bytes"]))
            rows[key][row["impl"]] = {
                "mean_ms": float(row["mean_ms"]),
                "bus_gbps": float(row["bus_gbps"]),
                "correct": row.get("correct", ""),
                "max_abs_err": row.get("max_abs_err", ""),
            }
    return rows


def main():
    if len(sys.argv) != 3:
        print("usage: plot_sp001_evidence.py <input.csv> <output.png>")
        sys.exit(2)
    csv_path, png_path = sys.argv[1], sys.argv[2]
    data = load(csv_path)

    sizes, iris_bw, rccl_bw, ratio, correct_flag = [], [], [], [], []
    for (_coll, _dt, nbytes), impls in sorted(data.items(), key=lambda kv: kv[0][2]):
        if "iris" not in impls or "rccl" not in impls:
            continue
        sizes.append(nbytes)
        iris_bw.append(impls["iris"]["bus_gbps"])
        rccl_bw.append(impls["rccl"]["bus_gbps"])
        ratio.append(impls["iris"]["mean_ms"] / impls["rccl"]["mean_ms"])
        correct_flag.append(impls["iris"]["correct"] == "true")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(sizes, rccl_bw, "o-", label="RCCL", color="C1")
    ax1.plot(sizes, iris_bw, "o-", label="iris.ccl default", color="C0")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("message bytes (log2)")
    ax1.set_ylabel("bus bandwidth (GB/s, log)")
    ax1.set_title("all_reduce fp16 ws=8 MI300X — bus bandwidth")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(sizes, ratio, "o-", color="C3", label="iris / rccl latency")
    ax2.axhline(1.10, color="k", linestyle="--", linewidth=1, label="10% gate")
    for x, r, ok in zip(sizes, ratio, correct_flag):
        if not ok:
            ax2.annotate("X", (x, r), color="red", fontsize=12, ha="center")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("message bytes (log2)")
    ax2.set_ylabel("iris_latency / rccl_latency (log)")
    ax2.set_title("Acceptance gate: <= 1.10 (red X = correctness fail)")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
