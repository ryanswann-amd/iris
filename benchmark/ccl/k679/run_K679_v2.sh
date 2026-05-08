#!/usr/bin/env bash
# K-679 v2 driver — runs the cleaned-up bench (no amdsmi, pre-aggregated
# JSONL) inside the container.
set -euo pipefail

WORKSPACE=${WORKSPACE:-/home/ryaswann/mc2/K-679}
IRIS_REPO=${IRIS_REPO:-${WORKSPACE}/repos/iris}
RUN_ID=${1:-run1}
OUT_DIR=${WORKSPACE}/output/perrank_v2_${RUN_ID}
LOG_DIR=${WORKSPACE}/logs

mkdir -p "$OUT_DIR" "$LOG_DIR"

export PYTHONPATH="$IRIS_REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

echo "[K-679 v2 $(date -uIs)] launching torchrun run_id=${RUN_ID}"
cd "$WORKSPACE"
PORT=$((29500 + (RANDOM % 100)))
torchrun --nproc_per_node=8 --master_port=${PORT} \
    "${WORKSPACE}/scripts/bench_ar_5bucket_v2.py" \
    --out_dir "$OUT_DIR" \
    --run_id "$RUN_ID" \
    --warmup 500 --iters 2000 \
    --sizes 1024,4096,16384 \
    --variants rccl,one_shot,two_shot \
    2>&1 | tee "${LOG_DIR}/v2_${RUN_ID}.log"

echo "[K-679 v2 $(date -uIs)] done run_id=${RUN_ID}"
