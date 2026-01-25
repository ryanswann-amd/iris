#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

set -e

SCRIPT_DIR=$(dirname "$(realpath "$0")")
VENV_DIR="${SCRIPT_DIR}/venv"

echo "[INFO] Setting up baremetal Python venv at ${VENV_DIR}"

# Create virtual environment if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
    echo "[INFO] Creating new Python virtual environment..."
    python3 -m venv --system-site-packages "${VENV_DIR}"
    echo "[INFO] Virtual environment created successfully"
else
    echo "[INFO] Using existing virtual environment at ${VENV_DIR}"
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Upgrade pip
echo "[INFO] Upgrading pip..."
pip install --upgrade pip

# Install basic dependencies similar to Docker/Apptainer images
# Note: Using latest versions for simplicity. For reproducible builds,
# consider creating a requirements.txt with pinned versions.
echo "[INFO] Installing base dependencies..."
pip install wheel jupyter

echo "[INFO] Baremetal environment setup completed successfully"
echo "[INFO] Virtual environment location: ${VENV_DIR}"
echo "[INFO] To activate manually: source ${VENV_DIR}/bin/activate"
