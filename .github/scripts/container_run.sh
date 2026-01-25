#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Universal container run script that works with Apptainer, Docker, or Baremetal

set -e

# Check if CONTAINER_RUNTIME is already set (e.g., from CI environment)
# If not set, auto-detect based on available tools
if [ -z "$CONTAINER_RUNTIME" ]; then
    if command -v apptainer &> /dev/null; then
        CONTAINER_RUNTIME="apptainer"
        echo "[INFO] Auto-detected Apptainer"
    elif command -v docker &> /dev/null; then
        CONTAINER_RUNTIME="docker"
        echo "[INFO] Auto-detected Docker"
    else
        # Fallback to baremetal (Python venv)
        CONTAINER_RUNTIME="baremetal"
        echo "[INFO] Auto-detected Baremetal (Python venv)"
    fi
else
    echo "[INFO] Using CONTAINER_RUNTIME from environment: $CONTAINER_RUNTIME"
fi

# Run based on detected runtime
if [ "$CONTAINER_RUNTIME" = "apptainer" ]; then
    echo "[INFO] Running with Apptainer..."
    bash apptainer/run.sh "$@"
elif [ "$CONTAINER_RUNTIME" = "docker" ]; then
    echo "[INFO] Running with Docker..."
    # Use GitHub variable if set, otherwise default to iris-dev
    IMAGE_NAME=${1:-${DOCKER_IMAGE_NAME:-"iris-dev"}}
    WORKSPACE_DIR=${2:-"$(pwd)"}
    bash docker/run.sh "$IMAGE_NAME" "$WORKSPACE_DIR"
elif [ "$CONTAINER_RUNTIME" = "baremetal" ]; then
    echo "[INFO] Running with Baremetal..."
    bash baremetal/run.sh "$@"
fi

