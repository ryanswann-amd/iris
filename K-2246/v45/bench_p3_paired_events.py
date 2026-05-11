#!/usr/bin/env python3
"""
K-2246 retry — v45 paired-interference re-measure with FOCAL-STREAM CUDA EVENTS.

Reviewer (Skeptic) feedback on the prior v45 paired CSV: timing wrapped
`time.perf_counter()` around BOTH streams + a single `torch.cuda.synchronize()`,
so the recorded latency was max(focal_done, interferer_done) — not the focal
P3 latency under contention.  This script fixes that.

Methodology (per rep):
    e_start = Event(stream=focal)
    e_end   = Event(stream=focal)
    inter_kernel[grid](...) on stream B
    e_start.record(stream=focal)
    p3_kernel[grid](...) on stream A
    e_end.record(stream=focal)
    cudaEventSynchronize(e_end)   # waits only for the focal end-event
    sample_us = e_start.elapsed_time(e_end) * 1000

Because CUDA events `record()` on a stream serialize with that stream's queue,
e_start records AFTER any prior work on the focal stream completes (we ensure
no prior work is queued by syncing the device once at rep boundary), and
e_end records when the focal P3 launch finishes — so e_start→e_end is exactly
the focal kernel's GPU time WHILE the interferer is in flight on stream B.

This pairs P3 latency under contention with the interferer kernel running
concurrently, but isolates the FOCAL latency from the interferer's runtime.

Output: paired CSV with columns matching v45_paired.csv plus a 'timer' column.
Restricted to the canonical cell K=4 N_PROD=4 N_OPS=32 for a tight, defensible
fix at the canonical cell only (the FULL paired CSV from the prior attempt is
retained as v45_paired.csv; this is v45_paired_canonical_event.csv).
"""
from __future__ import annotations
import argparse, csv, os, sys, time, socket
from datetime import datetime, timezone

import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench_v45_anchor import p3_kernel
from interferer_kernels import KERNELS

PAIRED_INTERFERERS = ['Y', 'F', 'P', 'G', 'H', 'R2', 'D3', 'J3', 'K3', 'L3', 'E3',
                      'G3', 'H3', 'I3', 'M3', 'N3', 'O3']

K_CANONICAL, NP_CANONICAL, NO_CANONICAL = 4, 4, 32
WARMUP = 5


def time_paired_focal(K: int, n_prod: int, n_ops: int, inter_kernel, reps: int) -> list[float]:
    n_ctas = K * n_prod
    slots_p3 = torch.zeros(n_prod, dtype=torch.int32, device='cuda')
    slots_int_int = torch.zeros(n_prod, dtype=torch.int32, device='cuda')
    slots_int_fp = torch.zeros(n_prod, dtype=torch.float32, device='cuda')
    slots_int_misc = torch.zeros(n_prod, dtype=torch.int32, device='cuda')

    s_focal = torch.cuda.Stream(); s_inter = torch.cuda.Stream()

    # warmup both streams
    for _ in range(WARMUP):
        with torch.cuda.stream(s_focal):
            p3_kernel[(n_ctas,)](slots_p3, n_ops=n_ops, n_prod=n_prod)
        with torch.cuda.stream(s_inter):
            inter_kernel[(n_ctas,)](slots_int_int, slots_int_fp, slots_int_misc,
                                    n_ops=n_ops, n_prod=n_prod)
    torch.cuda.synchronize()

    samples = []
    for _ in range(reps):
        # Drain any prior queued work on both streams; rep starts from quiescent.
        torch.cuda.synchronize()
        e_start = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)

        # Launch interferer FIRST so it's already in flight when focal starts.
        with torch.cuda.stream(s_inter):
            inter_kernel[(n_ctas,)](slots_int_int, slots_int_fp, slots_int_misc,
                                    n_ops=n_ops, n_prod=n_prod)
        # Now record start on focal stream and launch focal.
        with torch.cuda.stream(s_focal):
            e_start.record(s_focal)
            p3_kernel[(n_ctas,)](slots_p3, n_ops=n_ops, n_prod=n_prod)
            e_end.record(s_focal)
        # Wait only for focal end event.  cudaEventSynchronize blocks the host
        # until the GPU reaches the event; the interferer may still be running,
        # which is exactly what we want — it provides the contention.
        e_end.synchronize()
        samples.append(e_start.elapsed_time(e_end) * 1000.0)
        # Drain interferer before the next rep so we restart from quiescent.
        torch.cuda.synchronize()
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
    fields = ['corpus_version', 'focal', 'interferer', 'K', 'N_PROD', 'N_OPS',
              'rep', 'us', 'gpu_arch', 'node', 'ts_utc', 'timer']
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_int = sum(1 for k in PAIRED_INTERFERERS if k in KERNELS)
    sys.stderr.write(f'[paired-event] {n_int} interferers x {args.reps} reps at canonical cell on {node}/{arch}\n')
    written = 0
    with open(args.out, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=fields); wr.writeheader()
        # Also include the unpaired focal P3 as the contention-free baseline.
        sys.stderr.write('[paired-event] focal-only baseline ...\n')
        from bench_v45_anchor import time_one
        baseline_samples = time_one(p3_kernel, K_CANONICAL, NP_CANONICAL, NO_CANONICAL, args.reps)
        ts = datetime.now(timezone.utc).isoformat()
        for rep, us in enumerate(baseline_samples):
            wr.writerow(dict(corpus_version=args.corpus_version, focal='P3', interferer='__none__',
                             K=K_CANONICAL, N_PROD=NP_CANONICAL, N_OPS=NO_CANONICAL,
                             rep=rep, us=f'{us:.4f}', gpu_arch=arch, node=node,
                             ts_utc=ts, timer='cuda_event'))
            written += 1
        sb = sorted(baseline_samples)
        sys.stderr.write(f'[paired-event] focal-only median = {sb[len(sb)//2]:.3f} µs (n={len(sb)})\n')

        for inter in PAIRED_INTERFERERS:
            ik = KERNELS.get(inter)
            if ik is None:
                sys.stderr.write(f'[paired-event] WARN no kernel for {inter}; skip\n'); continue
            t0 = time.time()
            samples = time_paired_focal(K_CANONICAL, NP_CANONICAL, NO_CANONICAL, ik, args.reps)
            ts = datetime.now(timezone.utc).isoformat()
            for rep, us in enumerate(samples):
                wr.writerow(dict(corpus_version=args.corpus_version, focal='P3', interferer=inter,
                                 K=K_CANONICAL, N_PROD=NP_CANONICAL, N_OPS=NO_CANONICAL,
                                 rep=rep, us=f'{us:.4f}', gpu_arch=arch, node=node,
                                 ts_utc=ts, timer='cuda_event'))
                written += 1
            ss = sorted(samples)
            sys.stderr.write(f'[paired-event] {inter}: median={ss[len(ss)//2]:.3f} µs  (t={time.time()-t0:.1f}s)\n')
            fh.flush()
    sys.stderr.write(f'[paired-event] DONE: {written} rows -> {args.out}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
