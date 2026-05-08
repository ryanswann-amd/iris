#!/usr/bin/env bash
# K-679 — rocprofv3 --kernel-trace pass.
# Runs each (variant, size) cell under rocprofv3 and saves per-rank kernels.csv
# into output/ktrace/<variant>_<size>/rank<i>/.
set -euo pipefail

WORKSPACE=${WORKSPACE:-/home/ryaswann/mc2/K-679}
IRIS_REPO=${IRIS_REPO:-${WORKSPACE}/repos/iris}
KT_DIR=${WORKSPACE}/output/ktrace
LOG_DIR=${WORKSPACE}/logs

mkdir -p "$KT_DIR" "$LOG_DIR"
export PYTHONPATH="$IRIS_REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
chmod +x ${WORKSPACE}/scripts/_rocprof_run.sh

ITERS=${ITERS:-50}
WARMUP=${WARMUP:-20}
LOG=${LOG_DIR}/ktrace.log
: > "$LOG"

cd "$WORKSPACE"
for variant in rccl one_shot two_shot; do
    for size in 1024 4096 16384; do
        cell=${variant}_${size}
        export OUT_BASE=${KT_DIR}/${cell}
        rm -rf "$OUT_BASE"
        mkdir -p "$OUT_BASE"
        echo "[ktrace $(date -uIs)] cell=${cell}" | tee -a "$LOG"
        PORT=$((29700 + (RANDOM % 100)))
        torchrun --nproc_per_node=8 --master_port=${PORT} \
            ${WORKSPACE}/scripts/_rocprof_run.sh \
            python3 ${WORKSPACE}/scripts/bench_kernel_trace.py \
                --variant ${variant} --size ${size} \
                --warmup ${WARMUP} --iters ${ITERS} \
            >>"$LOG" 2>&1
    done
done
echo "[ktrace $(date -uIs)] DONE" | tee -a "$LOG"
