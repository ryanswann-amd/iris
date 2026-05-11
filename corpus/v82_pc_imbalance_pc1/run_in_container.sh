#!/bin/bash
# Wrapper to set up env inside container then run a python command.
set -e
export PYLIBS=/home/ryaswann/.mc2-pylibs/iris
export PYTHONPATH=$PYLIBS:$PYTHONPATH
export PYTHONUNBUFFERED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}
export TORCH_NCCL_TRACE_BUFFER_SIZE=0
export TORCH_NCCL_COORD_CHECK_MILSEC=600000
export NCCL_HEARTBEAT_TIMEOUT_MS=${NCCL_HEARTBEAT_TIMEOUT_MS:-1800000}
exec "$@"
