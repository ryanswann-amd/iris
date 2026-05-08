#!/usr/bin/env bash
# K-679 driver — run 3 torchrun runs of bench_ar_5bucket.py inside container.
# Inputs: WORKSPACE, IRIS_REPO, RUN_ID
set -euo pipefail

WORKSPACE=${WORKSPACE:-/home/ryaswann/mc2-workspaces/K-679}
IRIS_REPO=${IRIS_REPO:-${WORKSPACE}/repos/iris}
PYLIBS=${PYLIBS:-/home/ryaswann/.mc2-pylibs/iris}
RUN_ID=${1:-run1}
OUT_DIR=${WORKSPACE}/output/perrank_${RUN_ID}
LOG_DIR=${WORKSPACE}/logs

mkdir -p "$OUT_DIR" "$LOG_DIR"

export PYTHONPATH="$IRIS_REPO:$PYLIBS:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
# RCCL options: minimal verbosity
export NCCL_DEBUG=WARN

echo "[K-679 $(date -uIs)] launching torchrun run_id=${RUN_ID}"

cd "$WORKSPACE"
# Use a unique port per run to avoid collisions if reruns overlap
PORT=$((29500 + (RANDOM % 100)))

torchrun --nproc_per_node=8 --master_port=${PORT} \
    "${WORKSPACE}/scripts/bench_ar_5bucket.py" \
    --out_dir "$OUT_DIR" \
    --run_id "$RUN_ID" \
    --warmup 500 --iters 2000 \
    --sizes 1024,4096,16384 \
    --variants rccl,one_shot,two_shot \
    2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"

echo "[K-679 $(date -uIs)] done run_id=${RUN_ID}"
