#!/bin/bash
#SBATCH --job-name=k058-hbm-test-ws4
#SBATCH --partition=mi300x
#SBATCH --gres=gpu:mi300:4
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=/home/ryaswann/k058/hbm_test_ws4_%j.log
#SBATCH --error=/home/ryaswann/k058/hbm_test_ws4_%j.err

set -euo pipefail

IRIS_DIR="/home/ryaswann/iris"
VENV_DIR="/home/ryaswann/envs/iris_venv"

mkdir -p /home/ryaswann/k058

echo "=== K-058 HBM Buffer Matmul Tests (ws=4) ==="
echo "Node: $(hostname)"
echo "Date: $(date -u)"
echo "ROCm: $(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"

if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
fi

rm -rf ~/.triton/cache/* 2>/dev/null || true

cd "${IRIS_DIR}"

pip install -e . --quiet 2>&1 | tail -3

export HSA_NO_SCRATCH_RECLAIM=1
export PYTHONPATH="${IRIS_DIR}:${PYTHONPATH:-}"

echo "=== Running tests (ws=4, excluding slow) ==="
python -m torch.distributed.run \
    --nproc_per_node=4 \
    --standalone \
    tests/run_tests_distributed.py \
    tests/ops/test_all_gather_matmul.py \
    -v --tb=short -m "not slow"

echo "=== Done (exit $?) ==="
