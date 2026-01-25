#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

SCRIPT_DIR=$(dirname "$(realpath "$0")")
VENV_DIR="${SCRIPT_DIR}/venv"

# Check if virtual environment exists
if [ ! -d "${VENV_DIR}" ]; then
    echo "[ERROR] Virtual environment not found at ${VENV_DIR}"
    echo "[ERROR] Please run baremetal/build.sh first to create the environment"
    exit 1
fi

echo "[INFO] Using baremetal environment at ${VENV_DIR}"

# Activate virtual environment and start interactive bash
source "${VENV_DIR}/bin/activate"

# If arguments provided, execute them; otherwise start interactive bash
if [ $# -eq 0 ]; then
    exec bash
else
    exec "$@"
fi
