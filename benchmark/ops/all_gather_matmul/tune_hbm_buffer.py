#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Parameter tuning script for HBM-buffered all_gather_matmul.

Sweeps parameters around a baseline configuration, collecting traces, TFLOPs,
PyTorch baseline, and validation for every configuration.

This script does NOT modify benchmark_hbm_buffer.py — it invokes it via
``torchrun`` as a subprocess for each parameter set.

Usage:
    # Default one-at-a-time sweep (each param varied independently):
    python benchmark/ops/all_gather_matmul/tune_hbm_buffer.py

    # Custom matrix size:
    python benchmark/ops/all_gather_matmul/tune_hbm_buffer.py -m 8192 -n 4096 -k 131072

    # Only sweep specific parameters:
    python benchmark/ops/all_gather_matmul/tune_hbm_buffer.py --params num_fetch_sms k_per_flag

    # Full cartesian product (warning: combinatorial explosion):
    python benchmark/ops/all_gather_matmul/tune_hbm_buffer.py --mode full

    # Dry run — just print what would be tested:
    python benchmark/ops/all_gather_matmul/tune_hbm_buffer.py --dry_run
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Baseline configuration — the centre point of every sweep.
# Edit these to match your current best-known config.
# ─────────────────────────────────────────────────────────────────────────────
BASELINE = {
    "block_size_m": 256,
    "block_size_n": 256,
    "block_size_k": 64,
    "group_size_m": 4,
    "num_fetch_sms": 64,
    "k_per_flag": 64,
    "num_warps": 8,
    "num_fetch_stages": 4,
    "first_stage_fetch_sms": 304,
}

# ─────────────────────────────────────────────────────────────────────────────
# Sweep ranges — values to try for each parameter.
# In ``oneatatime`` mode only one parameter deviates from the baseline at a
# time; in ``full`` mode the cartesian product is taken (use with care).
# ─────────────────────────────────────────────────────────────────────────────
SWEEP_RANGES = {
    "block_size_m":          [64, 128, 256],
    "block_size_n":          [64, 128, 256],
    "block_size_k":          [64],
    "group_size_m":          [1, 2, 4, 8],
    "num_fetch_sms":         [64, 128, 192, 256],
    "k_per_flag":            [16, 32, 64, 128],
    "num_warps":             [4, 8],
    "num_fetch_stages":      [2, 4, 8],
    "first_stage_fetch_sms": [128, 192, 256, 304],
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_label(cfg):
    """Short human-readable label for a config."""
    parts = [
        f"bm{cfg['block_size_m']}",
        f"bn{cfg['block_size_n']}",
        f"bk{cfg['block_size_k']}",
        f"gm{cfg['group_size_m']}",
        f"nf{cfg['num_fetch_sms']}",
        f"kpf{cfg['k_per_flag']}",
        f"nw{cfg['num_warps']}",
        f"fs{cfg['num_fetch_stages']}",
    ]
    if cfg["num_fetch_stages"] > 1:
        parts.append(f"fsf{cfg['first_stage_fetch_sms']}")
    return "_".join(parts)


def validate_config(cfg, M, N, K, world_size=8):
    """Return a list of error strings; empty list means valid."""
    errors = []
    K_local = K // world_size
    bm, bn, bk = cfg["block_size_m"], cfg["block_size_n"], cfg["block_size_k"]
    kpf = cfg["k_per_flag"]

    if M % bm != 0:
        errors.append(f"M={M} not divisible by block_size_m={bm}")
    if N % bn != 0:
        errors.append(f"N={N} not divisible by block_size_n={bn}")
    if K % bk != 0:
        errors.append(f"K={K} not divisible by block_size_k={bk}")
    if K_local % bk != 0:
        errors.append(f"K_local={K_local} not divisible by block_size_k={bk}")

    num_k_blocks = K // bk
    if num_k_blocks % kpf != 0:
        errors.append(f"num_k_blocks={num_k_blocks} not divisible by k_per_flag={kpf}")

    if cfg["num_warps"] not in (1, 2, 4, 8, 16):
        errors.append(f"num_warps={cfg['num_warps']} must be a power of 2 in [1..16]")

    return errors


def build_command(cfg, M, N, K, trace_path, nproc=8,
                  validate=True, benchmark=True, benchmark_pytorch=False):
    """Build the ``torchrun`` CLI for one configuration."""
    cmd = [
        "torchrun", "--nproc_per_node", str(nproc),
        "benchmark/ops/all_gather_matmul/benchmark_hbm_buffer.py",
        "-m", str(M),
        "-n", str(N),
        "-k", str(K),
        "--block_size_m", str(cfg["block_size_m"]),
        "--block_size_n", str(cfg["block_size_n"]),
        "--block_size_k", str(cfg["block_size_k"]),
        "--group_size_m", str(cfg["group_size_m"]),
        "--num_fetch_sms", str(cfg["num_fetch_sms"]),
        "--k_per_flag", str(cfg["k_per_flag"]),
        "--num_warps", str(cfg["num_warps"]),
        "--num_fetch_stages", str(cfg["num_fetch_stages"]),
    ]

    if cfg["num_fetch_stages"] > 1 and cfg.get("first_stage_fetch_sms") is not None:
        cmd.extend(["--first_stage_fetch_sms", str(cfg["first_stage_fetch_sms"])])

    if validate:
        cmd.append("-v")
    if benchmark:
        cmd.append("-b")
    if benchmark_pytorch:
        cmd.append("--benchmark_pytorch")

    cmd.extend(["--trace", "--trace_output", trace_path])
    return cmd


# ── Output parsing ────────────────────────────────────────────────────────────

_RE_IRIS = re.compile(
    r"HBM-Buffer\s*\([^)]*\):\s*([\d.]+)\s*ms,\s*([\d.]+)\s*TFLOPS,\s*([\d.]+)\s*GB/s"
)
_RE_PYTORCH = re.compile(
    r"PyTorch\s*\([^)]*\):\s*([\d.]+)\s*ms,\s*([\d.]+)\s*TFLOPS,\s*([\d.]+)\s*GB/s"
)
_RE_SPEEDUP = re.compile(r"Speedup.*?:\s*([\d.]+)x")
_RE_VALID_FAIL = re.compile(r"Validation FAILED.*?max diff:\s*([\d.eE+-]+)")


def parse_output(output):
    """Extract metrics from benchmark stdout+stderr."""
    result = {
        "iris_ms": None,
        "iris_tflops": None,
        "iris_bw_gbps": None,
        "pytorch_ms": None,
        "pytorch_tflops": None,
        "pytorch_bw_gbps": None,
        "validation": None,
        "speedup": None,
    }

    m = _RE_IRIS.search(output)
    if m:
        result["iris_ms"] = float(m.group(1))
        result["iris_tflops"] = float(m.group(2))
        result["iris_bw_gbps"] = float(m.group(3))

    m = _RE_PYTORCH.search(output)
    if m:
        result["pytorch_ms"] = float(m.group(1))
        result["pytorch_tflops"] = float(m.group(2))
        result["pytorch_bw_gbps"] = float(m.group(3))

    if "Validation PASSED" in output:
        result["validation"] = "PASSED"
    elif "Validation FAILED" in output:
        fm = _RE_VALID_FAIL.search(output)
        result["validation"] = f"FAILED (diff={fm.group(1)})" if fm else "FAILED"

    m = _RE_SPEEDUP.search(output)
    if m:
        result["speedup"] = float(m.group(1))

    return result


# ── Sweep generation ──────────────────────────────────────────────────────────

def generate_configs(baseline, sweep_ranges, mode="oneatatime", params=None):
    """
    Generate the list of configs to evaluate.

    Args:
        baseline:     dict of default values
        sweep_ranges: dict mapping param name -> list of values
        mode:         "oneatatime" or "full"
        params:       optional list of param names to sweep (None = all)
    """
    configs = []
    seen = set()

    def _add(cfg):
        label = make_label(cfg)
        if label not in seen:
            configs.append(dict(cfg))
            seen.add(label)

    # Always include baseline first
    _add(baseline)

    active_params = params if params else list(sweep_ranges.keys())

    if mode == "oneatatime":
        for param in active_params:
            if param not in sweep_ranges:
                print(f"  WARNING: unknown param '{param}', skipping")
                continue
            for val in sweep_ranges[param]:
                cfg = dict(baseline)
                cfg[param] = val
                # When num_fetch_stages == 1, first_stage_fetch_sms is irrelevant
                if cfg["num_fetch_stages"] == 1:
                    cfg["first_stage_fetch_sms"] = cfg["num_fetch_sms"]
                _add(cfg)

    elif mode == "full":
        active_ranges = {p: sweep_ranges[p] for p in active_params if p in sweep_ranges}
        names = list(active_ranges.keys())
        values = [active_ranges[n] for n in names]
        for combo in product(*values):
            cfg = dict(baseline)
            for n, v in zip(names, combo):
                cfg[n] = v
            if cfg["num_fetch_stages"] == 1:
                cfg["first_stage_fetch_sms"] = cfg["num_fetch_sms"]
            _add(cfg)

    return configs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parameter tuning for HBM-buffered all_gather_matmul.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── Matrix dimensions ────────────────────────────────────────────────
    parser.add_argument("-m", type=int, default=16384, help="M dimension")
    parser.add_argument("-n", type=int, default=2048, help="N dimension")
    parser.add_argument("-k", type=int, default=131072, help="K dimension (total)")
    parser.add_argument("--nproc", type=int, default=8, help="Number of GPUs")

    # ── Baseline overrides (non-swept params use these values) ────────
    parser.add_argument("--block_size_m", type=int, default=None,
                        help=f"Baseline block_size_m (default: {BASELINE['block_size_m']})")
    parser.add_argument("--block_size_n", type=int, default=None,
                        help=f"Baseline block_size_n (default: {BASELINE['block_size_n']})")
    parser.add_argument("--block_size_k", type=int, default=None,
                        help=f"Baseline block_size_k (default: {BASELINE['block_size_k']})")
    parser.add_argument("--group_size_m", type=int, default=None,
                        help=f"Baseline group_size_m (default: {BASELINE['group_size_m']})")
    parser.add_argument("--num_fetch_sms", type=int, default=None,
                        help=f"Baseline num_fetch_sms (default: {BASELINE['num_fetch_sms']})")
    parser.add_argument("--k_per_flag", type=int, default=None,
                        help=f"Baseline k_per_flag (default: {BASELINE['k_per_flag']})")
    parser.add_argument("--num_warps", type=int, default=None,
                        help=f"Baseline num_warps (default: {BASELINE['num_warps']})")
    parser.add_argument("--num_fetch_stages", type=int, default=None,
                        help=f"Baseline num_fetch_stages (default: {BASELINE['num_fetch_stages']})")
    parser.add_argument("--first_stage_fetch_sms", type=int, default=None,
                        help=f"Baseline first_stage_fetch_sms (default: {BASELINE['first_stage_fetch_sms']})")

    # ── Sweep control ─────────────────────────────────────────────────
    parser.add_argument(
        "--mode", choices=["oneatatime", "full"], default="oneatatime",
        help="'oneatatime' varies one param at a time; 'full' = cartesian product",
    )
    parser.add_argument(
        "--params", nargs="+", default=None,
        help="Only sweep these parameters (default: all). "
             "Choices: " + ", ".join(SWEEP_RANGES.keys()),
    )
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (auto-generated if unset)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print configs and exit without running")
    parser.add_argument("--skip_validation", action="store_true",
                        help="Skip validation (faster, no correctness check)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-config timeout in seconds (default: 600)")

    args = parser.parse_args()
    M, N, K = args.m, args.n, args.k

    # Apply any CLI baseline overrides
    baseline = dict(BASELINE)
    for key in baseline:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            baseline[key] = cli_val

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"benchmark/ops/all_gather_matmul/tune_results_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(exist_ok=True)

    # Generate configs
    configs = generate_configs(baseline, SWEEP_RANGES,
                               mode=args.mode, params=args.params)

    # Pre-validate all configs
    valid_configs = []
    skipped = []
    for cfg in configs:
        errs = validate_config(cfg, M, N, K, world_size=args.nproc)
        if errs:
            skipped.append((cfg, errs))
        else:
            valid_configs.append(cfg)

    # Banner
    print(f"\n{'='*100}")
    print(f"  HBM-Buffer All-Gather MatMul  —  Parameter Tuning")
    print(f"  M={M}  N={N}  K={K}  nproc={args.nproc}  mode={args.mode}")
    print(f"  Baseline: {make_label(baseline)}")
    print(f"  Configs to run: {len(valid_configs)}  (skipped: {len(skipped)})")
    print(f"  Output dir:     {output_dir}")
    print(f"  Validation:     {'OFF' if args.skip_validation else 'ON'}")
    print(f"{'='*100}")

    if skipped:
        print(f"\n  Skipped (invalid for M={M}, N={N}, K={K}):")
        for cfg, errs in skipped:
            print(f"    {make_label(cfg)}: {'; '.join(errs)}")

    if args.dry_run:
        print(f"\n  Configs that would be run:")
        for i, cfg in enumerate(valid_configs):
            label = make_label(cfg)
            is_baseline = (cfg == baseline)
            tag = " [BASELINE]" if is_baseline else ""
            print(f"    [{i+1:>3}] {label}{tag}")
        print(f"\n  Total: {len(valid_configs)} configs")
        return

    # ── Run sweep ─────────────────────────────────────────────────────────
    results = []
    pytorch_baseline = None
    env = os.environ.copy()
    env["HSA_NO_SCRATCH_RECLAIM"] = "1"

    total_start = time.time()

    for i, cfg in enumerate(valid_configs):
        label = make_label(cfg)
        trace_path = str(trace_dir / f"trace_{label}.png")
        is_first = (i == 0)

        sep = "-" * 80
        print(f"\n{sep}")
        print(f"[{i+1}/{len(valid_configs)}] {label}")
        if is_first:
            print(f"  (includes PyTorch baseline benchmark)")
        print(sep)

        cmd = build_command(
            cfg, M, N, K, trace_path,
            nproc=args.nproc,
            validate=not args.skip_validation,
            benchmark=True,
            benchmark_pytorch=is_first,
        )
        cmd_str = " ".join(cmd)
        print(f"  $ HSA_NO_SCRATCH_RECLAIM=1 {cmd_str}")

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, env=env,
                capture_output=True, text=True,
                timeout=args.timeout,
            )
            elapsed = time.time() - t0
            full_output = proc.stdout + "\n" + proc.stderr

            parsed = parse_output(full_output)

            # Capture PyTorch baseline on first run
            if is_first and parsed["pytorch_tflops"] is not None:
                pytorch_baseline = {
                    "ms": parsed["pytorch_ms"],
                    "tflops": parsed["pytorch_tflops"],
                    "bw_gbps": parsed["pytorch_bw_gbps"],
                }

            trace_exists = os.path.exists(trace_path)
            results.append({
                "label": label,
                "config": cfg,
                "iris_ms": parsed["iris_ms"],
                "iris_tflops": parsed["iris_tflops"],
                "iris_bw_gbps": parsed["iris_bw_gbps"],
                "validation": parsed["validation"],
                "trace_path": trace_path if trace_exists else None,
                "elapsed_s": round(elapsed, 1),
                "returncode": proc.returncode,
            })

            # Print summary line
            parts = []
            if parsed["iris_tflops"] is not None:
                parts.append(f"{parsed['iris_tflops']:.2f} TFLOPS")
                parts.append(f"{parsed['iris_ms']:.3f} ms")
            if parsed["iris_bw_gbps"] is not None:
                parts.append(f"{parsed['iris_bw_gbps']:.1f} GB/s")
            if parsed["validation"]:
                parts.append(f"valid={parsed['validation']}")
            if trace_exists:
                parts.append(f"trace=OK")
            else:
                parts.append(f"trace=MISSING")
            if proc.returncode != 0:
                parts.append(f"EXIT={proc.returncode}")
            print(f"  => {' | '.join(parts)}  ({elapsed:.0f}s)")

            if is_first and pytorch_baseline:
                print(f"  => PyTorch baseline: {pytorch_baseline['tflops']:.2f} TFLOPS"
                      f"  {pytorch_baseline['ms']:.3f} ms")

            # Save full log for debugging
            log_path = output_dir / f"log_{label}.txt"
            with open(log_path, "w") as f:
                f.write(f"COMMAND: HSA_NO_SCRATCH_RECLAIM=1 {cmd_str}\n")
                f.write(f"EXIT CODE: {proc.returncode}\n")
                f.write(f"ELAPSED: {elapsed:.1f}s\n\n")
                f.write("=== STDOUT ===\n")
                f.write(proc.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(proc.stderr)

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            results.append({
                "label": label,
                "config": cfg,
                "iris_ms": None,
                "iris_tflops": None,
                "iris_bw_gbps": None,
                "validation": "TIMEOUT",
                "trace_path": None,
                "elapsed_s": round(elapsed, 1),
                "returncode": -1,
            })
            print(f"  => TIMEOUT after {args.timeout}s")

        except Exception as e:
            elapsed = time.time() - t0
            results.append({
                "label": label,
                "config": cfg,
                "iris_ms": None,
                "iris_tflops": None,
                "iris_bw_gbps": None,
                "validation": f"ERROR: {e}",
                "trace_path": None,
                "elapsed_s": round(elapsed, 1),
                "returncode": -1,
            })
            print(f"  => ERROR: {e}")

    total_elapsed = time.time() - total_start

    # ── Summary table ─────────────────────────────────────────────────────
    W = 130
    print(f"\n\n{'='*W}")
    print(f"  TUNING RESULTS  |  M={M}  N={N}  K={K}  |  nproc={args.nproc}  |  "
          f"{len(valid_configs)} configs in {total_elapsed:.0f}s")
    if pytorch_baseline:
        print(f"  PyTorch baseline: {pytorch_baseline['ms']:.3f} ms  |  "
              f"{pytorch_baseline['tflops']:.2f} TFLOPS  |  "
              f"{pytorch_baseline['bw_gbps']:.1f} GB/s")
    print(f"{'='*W}")

    col_label_w = 65
    print(f"  {'#':>3}  {'Configuration':<{col_label_w}}  {'ms':>8}  {'TFLOPS':>8}  "
          f"{'vs PT':>7}  {'Valid':>8}  {'Trace':>5}")
    print(f"  {'-'*(W-4)}")

    for i, r in enumerate(results):
        ms_s = f"{r['iris_ms']:.3f}" if r["iris_ms"] is not None else "--"
        tf_s = f"{r['iris_tflops']:.2f}" if r["iris_tflops"] is not None else "--"

        if pytorch_baseline and r["iris_tflops"] is not None and pytorch_baseline["tflops"] > 0:
            vs_pt = f"{r['iris_tflops'] / pytorch_baseline['tflops']:.2f}x"
        else:
            vs_pt = "--"

        valid_s = (r["validation"] or "--")[:8]
        trace_s = "Y" if r.get("trace_path") else "N"

        tag = " *" if (r["iris_tflops"] is not None and
                       r["iris_tflops"] == max((x["iris_tflops"] for x in results
                                                if x["iris_tflops"] is not None), default=0)) else ""

        print(f"  {i+1:>3}  {r['label']:<{col_label_w}}  {ms_s:>8}  {tf_s:>8}  "
              f"{vs_pt:>7}  {valid_s:>8}  {trace_s:>5}{tag}")

    # Best config
    valid_results = [r for r in results if r["iris_tflops"] is not None]
    if valid_results:
        best = max(valid_results, key=lambda r: r["iris_tflops"])
        worst = min(valid_results, key=lambda r: r["iris_tflops"])
        print(f"\n  {'BEST':>6}: {best['label']}")
        print(f"          {best['iris_ms']:.3f} ms  |  {best['iris_tflops']:.2f} TFLOPS  |  "
              f"valid={best['validation']}")
        if pytorch_baseline and pytorch_baseline["tflops"] > 0:
            print(f"          {best['iris_tflops'] / pytorch_baseline['tflops']:.2f}x vs PyTorch")
        if best.get("trace_path"):
            print(f"          trace: {best['trace_path']}")
        print(f"  {'WORST':>6}: {worst['label']}")
        print(f"          {worst['iris_ms']:.3f} ms  |  {worst['iris_tflops']:.2f} TFLOPS")
        if best["iris_tflops"] > 0 and worst["iris_tflops"] > 0:
            print(f"  SPREAD: {best['iris_tflops'] / worst['iris_tflops']:.2f}x "
                  f"({worst['iris_tflops']:.2f} → {best['iris_tflops']:.2f} TFLOPS)")

    print(f"{'='*W}")

    # ── Save results JSON ─────────────────────────────────────────────────
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "meta": {
                "M": M, "N": N, "K": K,
                "nproc": args.nproc,
                "mode": args.mode,
                "baseline": baseline,
                "sweep_ranges": SWEEP_RANGES,
                "timestamp": datetime.now().isoformat(),
                "total_elapsed_s": round(total_elapsed, 1),
                "pytorch_baseline": pytorch_baseline,
            },
            "results": results,
        }, f, indent=2, default=str)

    print(f"\n  Results JSON : {results_path}")
    print(f"  Trace PNGs   : {trace_dir}/")
    print(f"  Per-run logs : {output_dir}/log_*.txt")
    print()


if __name__ == "__main__":
    main()
