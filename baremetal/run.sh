#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Note: This script is designed for Unix-like systems (Linux, macOS).
# Windows users should use WSL or container-based backends.

SCRIPT_DIR=$(dirname "$(realpath "$0")")
VENV_DIR="${SCRIPT_DIR}/venv"

# Check if virtual environment exists, if not, build it
if [ ! -d "${VENV_DIR}" ]; then
    echo "[INFO] Virtual environment not found at ${VENV_DIR}, building it now..."
    if ! bash "${SCRIPT_DIR}/build.sh"; then
        echo "[ERROR] Failed to build virtual environment"
        exit 1
    fi
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
