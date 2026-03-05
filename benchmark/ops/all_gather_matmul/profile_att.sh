#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

# ATT (Advanced Thread Trace) Profiling Script for all_gather_matmul benchmark
# Uses rocprofv3 with thread trace to profile the benchmark at ISA instruction level.
#
# Usage:
#   ./profile_att.sh [OPTIONS]
#
# Options:
#   -r, --ranks NUM_RANKS       Number of ranks/GPUs (default: 8)
#   -m, --m-dim M               M dimension (default: 2048)
#   -n, --n-dim N               N dimension (default: 16384)
#   -k, --k-dim K               K dimension (default: 131072)
#   -v, --variant VARIANT       Variant: pull, chunked, push, pipelined_pull (default: pull)
#   --block-m SIZE              Block size for M dimension (default: 256)
#   --block-n SIZE              Block size for N dimension (default: 256)
#   --block-k SIZE              Block size for K dimension (default: 64)
#   --group-m SIZE              Group size for M dimension tiling (default: 1)
#   --num-xcds NUM              Number of XCDs (default: 8)
#   --validate                  Enable validation mode
#   --benchmark-pytorch         Also benchmark PyTorch for comparison
#   -o, --output-dir DIR        Base output directory (default: ./att_profiles)
#   --att-target-cu CU          Target CU for thread trace (default: 1)
#   --att-buffer-size SIZE      Trace buffer size in hex (default: 0x6000000 = 96MB)
#   --att-activity LEVEL        Perfcounter streaming level 1-16 (default: 8)
#   --kernel-regex REGEX        Kernel name regex filter (optional)
#   --single-run                Run only one iteration (no warmup, no repeat)
#   --k-contiguous              Use K-contiguous layout for both A and B matrices
#                               (default A is row-major/K-contiguous, adds --b_col_major)
#   --a-col-major               Store A matrix in column-major order (M-contiguous)
#   --b-col-major               Store B matrix in column-major order (K-contiguous)
#   -h, --help                  Show this help message

set -e

# Default values
NUM_RANKS=8
M_DIM=2048
N_DIM=16384
K_DIM=131072
VARIANT="pull"
BASE_OUTPUT_DIR="./att_profiles"
ATT_TARGET_CU=1
ATT_BUFFER_SIZE="0x6000000"  # 96MB
ATT_ACTIVITY=8
KERNEL_REGEX=""
SINGLE_RUN=true
K_CONTIGUOUS=true  # Default to K-contiguous layout for both matrices
A_COL_MAJOR=false
B_COL_MAJOR=false
BLOCK_M=256
BLOCK_N=256
BLOCK_K=64
GROUP_M=1
NUM_XCDS=8
VALIDATE=true
BENCHMARK_PYTORCH=true

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/benchmark_torchrun.py"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--ranks)
            NUM_RANKS="$2"
            shift 2
            ;;
        -m|--m-dim)
            M_DIM="$2"
            shift 2
            ;;
        -n|--n-dim)
            N_DIM="$2"
            shift 2
            ;;
        -k|--k-dim)
            K_DIM="$2"
            shift 2
            ;;
        -v|--variant)
            VARIANT="$2"
            shift 2
            ;;
        -o|--output-dir)
            BASE_OUTPUT_DIR="$2"
            shift 2
            ;;
        --att-target-cu)
            ATT_TARGET_CU="$2"
            shift 2
            ;;
        --att-buffer-size)
            ATT_BUFFER_SIZE="$2"
            shift 2
            ;;
        --att-activity)
            ATT_ACTIVITY="$2"
            shift 2
            ;;
        --kernel-regex)
            KERNEL_REGEX="$2"
            shift 2
            ;;
        --single-run)
            SINGLE_RUN=true
            shift
            ;;
        --k-contiguous)
            K_CONTIGUOUS=true
            shift
            ;;
        --a-col-major)
            A_COL_MAJOR=true
            shift
            ;;
        --b-col-major)
            B_COL_MAJOR=true
            shift
            ;;
        --block-m)
            BLOCK_M="$2"
            shift 2
            ;;
        --block-n)
            BLOCK_N="$2"
            shift 2
            ;;
        --block-k)
            BLOCK_K="$2"
            shift 2
            ;;
        --group-m)
            GROUP_M="$2"
            shift 2
            ;;
        --num-xcds)
            NUM_XCDS="$2"
            shift 2
            ;;
        --validate)
            VALIDATE=true
            shift
            ;;
        --no-validate)
            VALIDATE=false
            shift
            ;;
        --benchmark-pytorch)
            BENCHMARK_PYTORCH=true
            shift
            ;;
        --no-benchmark-pytorch)
            BENCHMARK_PYTORCH=false
            shift
            ;;
        -h|--help)
            head -30 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Generate timestamp for output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${BASE_OUTPUT_DIR}/att_${VARIANT}_m${M_DIM}_n${N_DIM}_k${K_DIM}_${TIMESTAMP}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Log file with timestamp
LOG_FILE="${OUTPUT_DIR}/profile_${TIMESTAMP}.log"

echo "=============================================="  | tee "${LOG_FILE}"
echo "ATT Profiling for all_gather_matmul benchmark" | tee -a "${LOG_FILE}"
echo "=============================================="  | tee -a "${LOG_FILE}"
echo "Timestamp: $(date)" | tee -a "${LOG_FILE}"
echo "Output directory: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Configuration:" | tee -a "${LOG_FILE}"
echo "  NUM_RANKS: ${NUM_RANKS}" | tee -a "${LOG_FILE}"
echo "  M: ${M_DIM}" | tee -a "${LOG_FILE}"
echo "  N: ${N_DIM}" | tee -a "${LOG_FILE}"
echo "  K: ${K_DIM}" | tee -a "${LOG_FILE}"
echo "  Variant: ${VARIANT}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "ATT Parameters:" | tee -a "${LOG_FILE}"
echo "  att-target-cu: ${ATT_TARGET_CU}" | tee -a "${LOG_FILE}"
echo "  att-buffer-size: ${ATT_BUFFER_SIZE}" | tee -a "${LOG_FILE}"
echo "  att-activity: ${ATT_ACTIVITY}" | tee -a "${LOG_FILE}"
if [[ -n "${KERNEL_REGEX}" ]]; then
    echo "  kernel-include-regex: ${KERNEL_REGEX}" | tee -a "${LOG_FILE}"
fi
echo "  single-run: ${SINGLE_RUN}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Matrix Layout:" | tee -a "${LOG_FILE}"
echo "  k-contiguous: ${K_CONTIGUOUS}" | tee -a "${LOG_FILE}"
echo "  a-col-major: ${A_COL_MAJOR}" | tee -a "${LOG_FILE}"
echo "  b-col-major: ${B_COL_MAJOR}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Block Sizes:" | tee -a "${LOG_FILE}"
echo "  block-m: ${BLOCK_M}" | tee -a "${LOG_FILE}"
echo "  block-n: ${BLOCK_N}" | tee -a "${LOG_FILE}"
echo "  block-k: ${BLOCK_K}" | tee -a "${LOG_FILE}"
echo "  group-m: ${GROUP_M}" | tee -a "${LOG_FILE}"
echo "  num-xcds: ${NUM_XCDS}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Benchmark Options:" | tee -a "${LOG_FILE}"
echo "  validate: ${VALIDATE}" | tee -a "${LOG_FILE}"
echo "  benchmark-pytorch: ${BENCHMARK_PYTORCH}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Build rocprofv3 ATT options
ROCPROF_OPTS="--att"
ROCPROF_OPTS="${ROCPROF_OPTS} --att-target-cu ${ATT_TARGET_CU}"
ROCPROF_OPTS="${ROCPROF_OPTS} --att-buffer-size ${ATT_BUFFER_SIZE}"
ROCPROF_OPTS="${ROCPROF_OPTS} --att-activity ${ATT_ACTIVITY}"

if [[ -n "${KERNEL_REGEX}" ]]; then
    ROCPROF_OPTS="${ROCPROF_OPTS} --kernel-include-regex \"${KERNEL_REGEX}\""
fi

# Build benchmark args
BENCH_ARGS="-m ${M_DIM} -n ${N_DIM} -k ${K_DIM} --variant ${VARIANT} --benchmark -r ${NUM_RANKS}"
BENCH_ARGS="${BENCH_ARGS} --block_size_m ${BLOCK_M} --block_size_n ${BLOCK_N} --block_size_k ${BLOCK_K}"
BENCH_ARGS="${BENCH_ARGS} --group_size_m ${GROUP_M} --num_xcds ${NUM_XCDS}"

if [[ "${SINGLE_RUN}" == "true" ]]; then
    BENCH_ARGS="${BENCH_ARGS} --single-run"
fi

if [[ "${VALIDATE}" == "true" ]]; then
    BENCH_ARGS="${BENCH_ARGS} -v"
fi

if [[ "${BENCHMARK_PYTORCH}" == "true" ]]; then
    BENCH_ARGS="${BENCH_ARGS} --benchmark_pytorch"
fi

# Add K-contiguous layout options
# --k-contiguous: Both A and B become K-contiguous
#   - A is already K-contiguous in default row-major layout
#   - B needs --b_col_major to become K-contiguous
if [[ "${K_CONTIGUOUS}" == "true" ]]; then
    BENCH_ARGS="${BENCH_ARGS} --b_col_major"
fi

# Individual matrix layout overrides
if [[ "${A_COL_MAJOR}" == "true" ]]; then
    BENCH_ARGS="${BENCH_ARGS} --a_col_major"
fi
if [[ "${B_COL_MAJOR}" == "true" ]]; then
    BENCH_ARGS="${BENCH_ARGS} --b_col_major"
fi

# Full command
# rocprofv3 wraps the entire torchrun command, not the other way around
# HSA_NO_SCRATCH_RECLAIM=1 prevents scratch memory reclaim issues
FULL_CMD="HSA_NO_SCRATCH_RECLAIM=1 rocprofv3 ${ROCPROF_OPTS} -d ${OUTPUT_DIR} -- torchrun --nproc_per_node=${NUM_RANKS} ${BENCHMARK_SCRIPT} ${BENCH_ARGS}"

echo "Command:" | tee -a "${LOG_FILE}"
echo "${FULL_CMD}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Save configuration to JSON for reference
cat > "${OUTPUT_DIR}/config.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "num_ranks": ${NUM_RANKS},
    "m_dim": ${M_DIM},
    "n_dim": ${N_DIM},
    "k_dim": ${K_DIM},
    "variant": "${VARIANT}",
    "att_target_cu": ${ATT_TARGET_CU},
    "att_buffer_size": "${ATT_BUFFER_SIZE}",
    "att_activity": ${ATT_ACTIVITY},
    "kernel_regex": "${KERNEL_REGEX}",
    "single_run": ${SINGLE_RUN},
    "k_contiguous": ${K_CONTIGUOUS},
    "a_col_major": ${A_COL_MAJOR},
    "b_col_major": ${B_COL_MAJOR},
    "block_m": ${BLOCK_M},
    "block_n": ${BLOCK_N},
    "block_k": ${BLOCK_K},
    "group_m": ${GROUP_M},
    "num_xcds": ${NUM_XCDS},
    "validate": ${VALIDATE},
    "benchmark_pytorch": ${BENCHMARK_PYTORCH},
    "command": "${FULL_CMD}"
}
EOF

echo "Starting profiling..." | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Run the profiling command
START_TIME=$(date +%s)

# Execute the command and capture output
eval "${FULL_CMD}" 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "" | tee -a "${LOG_FILE}"
echo "=============================================="  | tee -a "${LOG_FILE}"
echo "Profiling completed" | tee -a "${LOG_FILE}"
echo "Exit code: ${EXIT_CODE}" | tee -a "${LOG_FILE}"
echo "Duration: ${DURATION} seconds" | tee -a "${LOG_FILE}"
echo "End time: $(date)" | tee -a "${LOG_FILE}"
echo "=============================================="  | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# List output files
echo "Output files:" | tee -a "${LOG_FILE}"
ls -la "${OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"

# Check for stats CSV files
if ls "${OUTPUT_DIR}"/stats_*.csv 1> /dev/null 2>&1; then
    echo "" | tee -a "${LOG_FILE}"
    echo "Stats CSV files found:" | tee -a "${LOG_FILE}"
    ls -la "${OUTPUT_DIR}"/stats_*.csv 2>&1 | tee -a "${LOG_FILE}"
fi

# Check for ui_output directories (ROCprof Compute Viewer compatible)
if ls -d "${OUTPUT_DIR}"/ui_output_* 1> /dev/null 2>&1; then
    echo "" | tee -a "${LOG_FILE}"
    echo "UI output directories (for ROCprof Compute Viewer):" | tee -a "${LOG_FILE}"
    ls -d "${OUTPUT_DIR}"/ui_output_* 2>&1 | tee -a "${LOG_FILE}"
fi

echo "" | tee -a "${LOG_FILE}"
echo "Profile output saved to: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"

exit ${EXIT_CODE}
