#!/usr/bin/env bash
# Per-rank rocprofv3 wrapper: invoked by torchrun. Profiles ONLY rank 0
# to avoid 8-way profiler-instance contention on the shared GPU
# partitions (which deadlocks the elastic agent in rocm/pytorch:rocm7
# with rocprofiler-sdk 1.1.0). Other ranks run unwrapped.
set -e
RANK=${RANK:-0}
LRANK=${LOCAL_RANK:-$RANK}
if [ "$LRANK" = "0" ]; then
    OUT=${OUT_BASE:?need OUT_BASE}/rank${RANK}
    mkdir -p "$OUT"
    exec rocprofv3 --kernel-trace --output-format csv \
        --output-file kernels --output-directory "$OUT" \
        -- "$@"
else
    exec "$@"
fi
