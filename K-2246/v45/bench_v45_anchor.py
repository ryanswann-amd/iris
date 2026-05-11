#!/usr/bin/env python3
"""
K-2246 retry — v45 anchor sweep.

Reviewer (Skeptic) feedback: the prior v45 ladder compared P3 (b21u01, v45) against
F3=34.36 µs (different host, v44).  This is a cross-version + cross-host delta.

This script re-measures the FULL CAS ladder {M3, N3, O3, F3, P3} on b21u01 in
the v45 environment, at the canonical cell K=4 N_PROD=4 N_OPS=32, with REPS=50
(2x the corpus REPS for a tighter anchor).

Uses CUDA events for the timing gate (event.record on default stream → kernel →
event.record → event.synchronize → elapsed_time) so the timing measures GPU work
specifically, not host wall-clock.  Same memory ordering and grid as the K-2246
baseline kernel.

Output: anchor CSV with one row per (prim, rep), schema-compatible with the
v45_baseline.csv (corpus_version='v45', prim, K=4, N_PROD=4, N_OPS=32, ...).
"""
from __future__ import annotations
import argparse, csv, os, sys, time, socket
from datetime import datetime, timezone

import torch
import triton
import triton.language as tl


# -------- ladder kernels (one per memory-ordering combination) --------

@triton.jit
def m3_kernel(ptr, n_ops: tl.constexpr, n_prod: tl.constexpr):
    """M3: relaxed-load + relaxed-CAS."""
    pid = tl.program_id(0); slot = pid % n_prod
    for _ in range(n_ops):
        cur = tl.atomic_add(ptr + slot, 0, sem='relaxed', scope='sys')
        tl.atomic_cas(ptr + slot, cur, cur + 1, sem='relaxed', scope='sys')


@triton.jit
def n3_kernel(ptr, n_ops: tl.constexpr, n_prod: tl.constexpr):
    """N3: acquire-load + acquire-CAS."""
    pid = tl.program_id(0); slot = pid % n_prod
    for _ in range(n_ops):
        cur = tl.atomic_add(ptr + slot, 0, sem='acquire', scope='sys')
        tl.atomic_cas(ptr + slot, cur, cur + 1, sem='acquire', scope='sys')


@triton.jit
def o3_kernel(ptr, n_ops: tl.constexpr, n_prod: tl.constexpr):
    """O3: relaxed-load + release-CAS."""
    pid = tl.program_id(0); slot = pid % n_prod
    for _ in range(n_ops):
        cur = tl.atomic_add(ptr + slot, 0, sem='relaxed', scope='sys')
        tl.atomic_cas(ptr + slot, cur, cur + 1, sem='release', scope='sys')


@triton.jit
def f3_kernel(ptr, n_ops: tl.constexpr, n_prod: tl.constexpr):
    """F3: acquire-load + acq_rel-CAS."""
    pid = tl.program_id(0); slot = pid % n_prod
    for _ in range(n_ops):
        cur = tl.atomic_add(ptr + slot, 0, sem='acquire', scope='sys')
        tl.atomic_cas(ptr + slot, cur, cur + 1, sem='acq_rel', scope='sys')


@triton.jit
def p3_kernel(ptr, n_ops: tl.constexpr, n_prod: tl.constexpr):
    """P3: acq_rel-load + acq_rel-CAS  (the K-2246 focal primitive)."""
    pid = tl.program_id(0); slot = pid % n_prod
    for _ in range(n_ops):
        cur = tl.atomic_add(ptr + slot, 0, sem='acq_rel', scope='sys')
        tl.atomic_cas(ptr + slot, cur, cur + 1, sem='acq_rel', scope='sys')


KERNELS = {'M3': m3_kernel, 'N3': n3_kernel, 'O3': o3_kernel,
           'F3': f3_kernel, 'P3': p3_kernel}

K_CANONICAL, NP_CANONICAL, NO_CANONICAL = 4, 4, 32
WARMUP = 5


def time_one(kernel, K: int, n_prod: int, n_ops: int, reps: int) -> list[float]:
    """Time `reps` launches with CUDA events (GPU-only timing)."""
    n_ctas = K * n_prod
    slots = torch.zeros(n_prod, dtype=torch.int32, device='cuda')
    # Warmup
    for _ in range(WARMUP):
        kernel[(n_ctas,)](slots, n_ops=n_ops, n_prod=n_prod)
    torch.cuda.synchronize()
    samples = []
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts[i].record()
        kernel[(n_ctas,)](slots, n_ops=n_ops, n_prod=n_prod)
        ends[i].record()
    torch.cuda.synchronize()
    for i in range(reps):
        # elapsed_time returns ms — convert to µs.
        samples.append(starts[i].elapsed_time(ends[i]) * 1000.0)
    return samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--reps', type=int, default=50)
    ap.add_argument('--corpus-version', default='v45')
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.stderr.write('CUDA/HIP unavailable\n'); return 3
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
    node = socket.gethostname()
    fields = ['corpus_version', 'prim', 'K', 'N_PROD', 'N_OPS', 'rep', 'us',
              'gpu_arch', 'node', 'ts_utc', 'timer']
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sys.stderr.write(f'[anchor] {len(KERNELS)} prims x {args.reps} reps at canonical cell on {node}/{arch}\n')
    written = 0
    with open(args.out, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=fields); wr.writeheader()
        for name, kernel in KERNELS.items():
            t0 = time.time()
            samples = time_one(kernel, K_CANONICAL, NP_CANONICAL, NO_CANONICAL, args.reps)
            ts = datetime.now(timezone.utc).isoformat()
            for rep, us in enumerate(samples):
                wr.writerow(dict(corpus_version=args.corpus_version, prim=name,
                                 K=K_CANONICAL, N_PROD=NP_CANONICAL, N_OPS=NO_CANONICAL,
                                 rep=rep, us=f'{us:.4f}', gpu_arch=arch, node=node,
                                 ts_utc=ts, timer='cuda_event'))
                written += 1
            samples_sorted = sorted(samples)
            med = samples_sorted[len(samples_sorted)//2]
            sys.stderr.write(f'[anchor] {name}: median={med:.3f} µs  (n={len(samples)}, t={time.time()-t0:.1f}s)\n')
    sys.stderr.write(f'[anchor] DONE: {written} rows -> {args.out}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
