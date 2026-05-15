#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Plot the full SP-001 4-collective x 2-dtype sweep on MI300X (ws=8).

Reads the comprehensive_sweep.py CSV (rows keyed by collective/impl/dtype/size)
and emits a single 8-panel PNG: one panel per (collective, dtype) showing the
iris/rccl latency ratio against message size with the 10% acceptance gate
overlaid as a horizontal reference line.
"""

import csv
import sys
from collections import defaultdict

import matplotlib.pyplot as plt


_COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")
_DTYPES = ("fp16", "bf16")


def load(csv_path):
    """Group sweep CSV rows by (collective, dtype, size) -> impl -> stats."""
    rows = defaultdict(dict)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (row["collective"], row["dtype"], int(row["total_bytes"]))
            rows[key][row["impl"]] = {
                "mean_ms": float(row["mean_ms"]),
                "bus_gbps": float(row["bus_gbps"]),
            }
    return rows


def main():
    if len(sys.argv) != 3:
        print("usage: plot_sp001_full_sweep.py <input.csv> <output.png>")
        sys.exit(2)
    csv_path, png_path = sys.argv[1], sys.argv[2]
    data = load(csv_path)

    fig, axes = plt.subplots(len(_COLLECTIVES), len(_DTYPES), figsize=(12, 14), sharex=False)
    for i, coll in enumerate(_COLLECTIVES):
        for j, dt in enumerate(_DTYPES):
            ax = axes[i][j]
            sizes, ratios = [], []
            for (c, d, n), impls in sorted(data.items(), key=lambda kv: kv[0][2]):
                if c != coll or d != dt:
                    continue
                if "iris" not in impls or "rccl" not in impls:
                    continue
                sizes.append(n)
                ratios.append(impls["iris"]["mean_ms"] / impls["rccl"]["mean_ms"])
            if not sizes:
                ax.set_title(f"{coll}/{dt} (no data)")
                ax.set_axis_off()
                continue
            n_fail = sum(1 for r in ratios if r > 1.10)
            ax.plot(sizes, ratios, "o-", color="C3")
            ax.axhline(1.10, color="k", linestyle="--", linewidth=1)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(f"{coll}/{dt}  fail={n_fail}/{len(ratios)}  worst={max(ratios):.2f}x")
            ax.set_xlabel("message bytes (log2)")
            ax.set_ylabel("iris / rccl latency")
            ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("SP-001 full sweep — iris.ccl default Config vs RCCL on MI300X (ws=8)", y=0.995)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
