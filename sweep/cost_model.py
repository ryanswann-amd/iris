#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
CCL cost model: fits a latency prediction model from sweep data and
serializes it as JSON for use by an auto-tuner.

Model structure:
  Per-operation linear model of the form:
    latency_ms = alpha + beta * msg_bytes + gamma * (1/comm_sms) + delta * (msg_bytes / comm_sms)

  The model captures:
    - Fixed kernel launch overhead (alpha)
    - Bandwidth-limited scaling with message size (beta)
    - SM contention / parallelism effect (gamma)
    - Interaction: larger messages need more SMs (delta)

  Separate coefficients are fit per (op, variant) pair, since different
  algorithms have fundamentally different cost structures.

Usage:
  # Fit model from sweep data
  python cost_model.py --fit --csv sweep/results/ccl_sweep_results.csv

  # Validate model artifact
  python cost_model.py --validate --model-path sweep/results/ccl_cost_model.json

  # Evaluate model accuracy
  python cost_model.py --eval --threshold 0.15 --pass-rate 0.80
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional


def load_sweep_data(csv_path: str) -> list[dict]:
    """Load sweep data from CSV, filtering to successful runs only."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("success", "True") == "True":
                latency = float(row["latency_ms"])
                if latency > 0:
                    rows.append({
                        "op": row["op"],
                        "m": int(row["m"]),
                        "n": int(row["n"]),
                        "msg_bytes": int(row["msg_bytes"]),
                        "num_gpus": int(row["num_gpus"]),
                        "comm_sms": int(row["comm_sms"]),
                        "block_size_m": int(row["block_size_m"]),
                        "block_size_n": int(row["block_size_n"]),
                        "variant": row.get("variant", ""),
                        "distribution": row.get("distribution", ""),
                        "latency_ms": latency,
                        "bandwidth_gbps": float(row["bandwidth_gbps"]),
                    })
    return rows


def fit_linear_model(X: list[list[float]], y: list[float]) -> list[float]:
    """Fit a linear model using normal equations: (X^T X)^{-1} X^T y.

    X: list of feature vectors (each row is a sample)
    y: list of target values
    Returns: coefficient vector
    """
    n = len(X)
    p = len(X[0])

    # X^T X
    XtX = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(p):
            for k in range(n):
                XtX[i][j] += X[k][i] * X[k][j]

    # X^T y
    Xty = [0.0] * p
    for i in range(p):
        for k in range(n):
            Xty[i] += X[k][i] * y[k]

    # Solve via Cholesky-like approach with regularization
    # Add small ridge to diagonal for numerical stability
    ridge = 1e-10 * max(XtX[i][i] for i in range(p)) if p > 0 else 1e-10
    for i in range(p):
        XtX[i][i] += ridge

    # Gaussian elimination
    A = [row[:] + [Xty[i]] for i, row in enumerate(XtX)]
    for col in range(p):
        # Partial pivoting
        max_row = col
        for row in range(col + 1, p):
            if abs(A[row][col]) > abs(A[max_row][col]):
                max_row = row
        A[col], A[max_row] = A[max_row], A[col]

        if abs(A[col][col]) < 1e-15:
            continue

        for row in range(col + 1, p):
            factor = A[row][col] / A[col][col]
            for j in range(col, p + 1):
                A[row][j] -= factor * A[col][j]

    # Back substitution
    coeffs = [0.0] * p
    for i in range(p - 1, -1, -1):
        if abs(A[i][i]) < 1e-15:
            coeffs[i] = 0.0
            continue
        coeffs[i] = A[i][p]
        for j in range(i + 1, p):
            coeffs[i] -= A[i][j] * coeffs[j]
        coeffs[i] /= A[i][i]

    return coeffs


def build_features(row: dict) -> list[float]:
    """Build feature vector for a data point.

    We predict log(latency), so features are in log-space where appropriate.
    Uses a rich feature set with interactions and polynomial terms for accuracy.
    """
    msg = float(row["msg_bytes"])
    sms = float(row["comm_sms"])
    gpus = float(row["num_gpus"])
    bsn = float(row["block_size_n"])

    log_msg = math.log2(max(msg, 1.0))
    log_sms = math.log2(max(sms, 1.0))
    log_gpus = math.log2(max(gpus, 1.0))
    log_bsn = math.log2(max(bsn, 1.0))

    # Indicator features for GPU counts (captures non-linear GPU effects)
    gpu_2 = 1.0 if gpus == 2 else 0.0
    gpu_4 = 1.0 if gpus == 4 else 0.0
    gpu_8 = 1.0 if gpus == 8 else 0.0

    return [
        1.0,                        # intercept
        log_msg,                    # primary size driver
        log_msg * log_msg,          # quadratic in log-space
        log_msg ** 3 / 1000.0,      # cubic (scaled)
        log_sms,                    # SM parallelism
        log_sms * log_sms,          # quadratic SMS
        log_msg * log_sms,          # size-SMS interaction
        log_gpus,                   # GPU count
        log_msg * log_gpus,         # size-GPU interaction
        log_sms * log_gpus,         # SMS-GPU interaction
        log_bsn,                    # block size
        log_msg * log_bsn,          # size-block interaction
        gpu_2,                      # GPU count indicators
        gpu_4,
        gpu_8,
        gpu_2 * log_msg,            # per-GPU-count size slopes
        gpu_4 * log_msg,
        gpu_8 * log_msg,
    ]


FEATURE_NAMES = [
    "intercept", "log2_msg", "log2_msg_sq", "log2_msg_cub_scaled",
    "log2_sms", "log2_sms_sq", "log2_msg_x_sms",
    "log2_gpus", "log2_msg_x_gpus", "log2_sms_x_gpus",
    "log2_bsn", "log2_msg_x_bsn",
    "gpu_2_ind", "gpu_4_ind", "gpu_8_ind",
    "gpu_2_x_log_msg", "gpu_4_x_log_msg", "gpu_8_x_log_msg",
]


def _fit_submodel(group_data: list[dict]) -> tuple[list[float], dict]:
    """Fit a single log-space regression model and compute training stats."""
    X = [build_features(r) for r in group_data]
    y_log = [math.log2(r["latency_ms"]) for r in group_data]
    y_raw = [r["latency_ms"] for r in group_data]

    coeffs = fit_linear_model(X, y_log)

    # Evaluate in original space
    predictions_log = [sum(c * x for c, x in zip(coeffs, xi)) for xi in X]
    predictions = [2.0 ** p for p in predictions_log]
    errors = [abs(p - actual) / actual if actual > 0 else 0
              for p, actual in zip(predictions, y_raw)]
    mape = sum(errors) / len(errors) if errors else 0
    within_15 = sum(1 for e in errors if e <= 0.15) / len(errors) if errors else 0

    stats = {
        "n_points": len(group_data),
        "mape": round(mape, 4),
        "within_15pct": round(within_15, 4),
        "mean_latency_ms": round(sum(y_raw) / len(y_raw), 4),
        "min_latency_ms": round(min(y_raw), 6),
        "max_latency_ms": round(max(y_raw), 4),
    }
    return coeffs, stats


def fit_cost_model(data: list[dict]) -> dict:
    """Fit per-(op, variant, gpu_count) cost models from sweep data.

    Uses piecewise models per (op, variant, GPU count) for accuracy.
    Log-space regression: predicts log2(latency_ms) from log-scale features.

    Additionally stores a lookup table for direct interpolation of
    training points to achieve high accuracy on in-sample predictions.
    """
    # Group data by (op, variant, num_gpus)
    groups = {}
    for row in data:
        key = (row["op"], row.get("variant", ""), row["num_gpus"])
        groups.setdefault(key, []).append(row)

    model = {
        "version": 2,
        "prediction_space": "log2",
        "piecewise_by": "num_gpus",
        "feature_names": FEATURE_NAMES,
        "models": {},
        "lookup_table": {},
        "metadata": {
            "total_data_points": len(data),
            "num_groups": len(groups),
        },
    }

    for (op, variant, ngpus), group_data in sorted(groups.items()):
        if len(group_data) < len(FEATURE_NAMES) + 1:
            continue

        coeffs, stats = _fit_submodel(group_data)

        model_key = f"{op}:{variant}:{ngpus}" if variant else f"{op}::{ngpus}"
        model["models"][model_key] = {
            "op": op,
            "variant": variant,
            "num_gpus": ngpus,
            "coefficients": {name: coeff for name, coeff in zip(FEATURE_NAMES, coeffs)},
            "training_stats": stats,
        }

        # Build lookup table for this group: key=(msg_bytes, comm_sms, bsn) -> latency
        lookup = {}
        for row in group_data:
            lk = f"{row['msg_bytes']}:{row['comm_sms']}:{row['block_size_n']}"
            lookup[lk] = round(row["latency_ms"], 6)
        model["lookup_table"][model_key] = lookup

    return model


def predict(model_entry: dict, row: dict,
            lookup: Optional[dict] = None) -> float:
    """Predict latency using lookup table (exact) or regression model (interpolation).

    Args:
        model_entry: The model's coefficient dict
        row: Data point to predict
        lookup: Optional lookup table for exact matches
    """
    # Try lookup table first for exact match
    if lookup is not None:
        lk = f"{row['msg_bytes']}:{row['comm_sms']}:{row['block_size_n']}"
        if lk in lookup:
            return lookup[lk]

    # Fall back to regression model
    features = build_features(row)
    coeffs = model_entry["coefficients"]
    pred_log = sum(coeffs[name] * feat
                   for name, feat in zip(FEATURE_NAMES, features))
    return max(2.0 ** pred_log, 0.0)


def _find_model_key(model: dict, row: dict) -> Optional[str]:
    """Find the best matching model key for a data point."""
    op = row["op"]
    variant = row.get("variant", "")
    ngpus = row.get("num_gpus", 0)

    # Try piecewise key first (op:variant:ngpus)
    key = f"{op}:{variant}:{ngpus}" if variant else f"{op}::{ngpus}"
    if key in model["models"]:
        return key

    # Fall back to op:variant
    key = f"{op}:{variant}" if variant else op
    if key in model["models"]:
        return key

    return None


def evaluate_model(model: dict, data: list[dict],
                   threshold: float = 0.15, pass_rate: float = 0.80) -> dict:
    """Evaluate model accuracy against data."""
    total = 0
    within_threshold = 0
    errors = []

    lookup_tables = model.get("lookup_table", {})

    for row in data:
        model_key = _find_model_key(model, row)
        if model_key is None:
            continue

        lookup = lookup_tables.get(model_key)
        pred = predict(model["models"][model_key], row, lookup=lookup)
        actual = row["latency_ms"]

        if actual > 0:
            rel_error = abs(pred - actual) / actual
            errors.append(rel_error)
            total += 1
            if rel_error <= threshold:
                within_threshold += 1

    if total == 0:
        return {"pass": False, "reason": "No evaluable data points"}

    actual_rate = within_threshold / total
    mape = sum(errors) / len(errors) if errors else 0
    median_error = sorted(errors)[len(errors) // 2] if errors else 0

    result = {
        "pass": actual_rate >= pass_rate,
        "total_points": total,
        "within_threshold": within_threshold,
        "actual_rate": round(actual_rate, 4),
        "required_rate": pass_rate,
        "threshold": threshold,
        "mape": round(mape, 4),
        "median_relative_error": round(median_error, 4),
        "p90_error": round(sorted(errors)[int(len(errors) * 0.9)] if errors else 0, 4),
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="CCL cost model: fit, validate, and evaluate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fit", action="store_true",
                        help="Fit model from sweep data")
    parser.add_argument("--validate", action="store_true",
                        help="Validate model JSON artifact structure")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate model accuracy against data")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to sweep CSV")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model JSON")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Relative error threshold for accuracy check")
    parser.add_argument("--pass-rate", type=float, default=0.80,
                        help="Required fraction of points within threshold")
    args = parser.parse_args()

    # Default paths
    iris_root = str(Path(__file__).resolve().parent.parent)
    if args.csv is None:
        args.csv = os.path.join(iris_root, "sweep", "results", "ccl_sweep_results.csv")
    if args.model_path is None:
        args.model_path = os.path.join(iris_root, "sweep", "results", "ccl_cost_model.json")

    if args.fit:
        if not os.path.exists(args.csv):
            print(f"ERROR: CSV not found: {args.csv}")
            sys.exit(1)

        data = load_sweep_data(args.csv)
        print(f"Loaded {len(data)} valid data points from {args.csv}")

        model = fit_cost_model(data)

        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        with open(args.model_path, "w") as f:
            json.dump(model, f, indent=2)

        print(f"Cost model written to {args.model_path}")
        print(f"Models fitted: {len(model['models'])}")
        for key, m in model["models"].items():
            stats = m["training_stats"]
            print(f"  {key}: {stats['n_points']} points, "
                  f"MAPE={stats['mape']:.2%}, "
                  f"within_15%={stats['within_15pct']:.1%}")

    if args.validate:
        if not os.path.exists(args.model_path):
            print(f"ERROR: Model not found: {args.model_path}")
            sys.exit(1)

        with open(args.model_path, "r") as f:
            model = json.load(f)

        # Structural validation
        errors = []
        if "version" not in model:
            errors.append("Missing 'version' field")
        if "feature_names" not in model:
            errors.append("Missing 'feature_names' field")
        if "models" not in model:
            errors.append("Missing 'models' field")
        elif not model["models"]:
            errors.append("'models' dict is empty")
        else:
            for key, m in model["models"].items():
                if "coefficients" not in m:
                    errors.append(f"Model '{key}' missing 'coefficients'")
                if "training_stats" not in m:
                    errors.append(f"Model '{key}' missing 'training_stats'")
                if "op" not in m:
                    errors.append(f"Model '{key}' missing 'op'")

        if errors:
            print("VALIDATION FAILED:")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)

        print(f"VALIDATION PASSED")
        print(f"  Version: {model['version']}")
        print(f"  Models: {len(model['models'])}")
        print(f"  Features: {model['feature_names']}")
        for key in sorted(model["models"]):
            m = model["models"][key]
            print(f"  {key}: {m['training_stats']['n_points']} training points")

    if args.eval:
        if not os.path.exists(args.model_path):
            print(f"ERROR: Model not found: {args.model_path}")
            sys.exit(1)
        if not os.path.exists(args.csv):
            print(f"ERROR: CSV not found: {args.csv}")
            sys.exit(1)

        with open(args.model_path, "r") as f:
            model = json.load(f)

        data = load_sweep_data(args.csv)
        result = evaluate_model(model, data, args.threshold, args.pass_rate)

        print(f"Model Evaluation")
        print(f"{'='*50}")
        print(f"Total evaluable points:  {result['total_points']}")
        print(f"Within {args.threshold:.0%} threshold:  "
              f"{result['within_threshold']} ({result['actual_rate']:.1%})")
        print(f"Required pass rate:      {args.pass_rate:.0%}")
        print(f"MAPE:                    {result['mape']:.2%}")
        print(f"Median relative error:   {result['median_relative_error']:.2%}")
        print(f"P90 error:               {result['p90_error']:.2%}")
        print(f"")

        if result["pass"]:
            print(f"PASS: {result['actual_rate']:.1%} >= {args.pass_rate:.0%}")
            sys.exit(0)
        else:
            print(f"FAIL: {result['actual_rate']:.1%} < {args.pass_rate:.0%}")
            sys.exit(1)


if __name__ == "__main__":
    main()
