"""K-2380 sweep driver.

5 duty cycles {100, 50, 25, 10, 5} × 4 ordering classes × 6 (wgp_count, block_size) × 25 reps
= 3,000 launches. Writes a tidy CSV: one row per rep with per-rep latency_ms.
"""
import os, sys, time, json, socket, csv, argparse, statistics
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from duty_cycle_kernel import (
    KERNELS, calibrate_active_cycles, launch, sleep_reps_for_duty
)

DUTY_CYCLES = [100, 50, 25, 10, 5]
OPS = ['XCHG_ACQREL', 'MAX_ACQREL', 'CAS_ACQREL', 'FADD_RELEASE']
# 6 (wgp_count, block_size) cells — span K-2317 canonical and K-2348 wgp ranges
CELLS = [
    (4,   64),
    (4,  256),
    (16,  64),
    (16, 256),
    (32, 256),
    (64, 256),
]
N_REPS = 25
N_WARMUP = 5
BATCHES_PER_PGM = 8


def percentile(vals, p):
    s = sorted(vals)
    if not s:
        return float('nan')
    k = (len(s) - 1) * p / 100.0
    f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def run(out_path, smoke=False):
    device = 'cuda'
    assert torch.cuda.is_available(), "no CUDA/HIP device"
    gpu_name = torch.cuda.get_device_name(0)
    host = socket.gethostname()
    print(f"[K-2380] host={host} gpu={gpu_name} writing -> {out_path}", flush=True)

    # 1) calibration: per (op, batch_size) measure active cycles per batch
    cal = {}
    for op in OPS:
        for _, bs in set([(0, c[1]) for c in CELLS]):
            try:
                cyc = calibrate_active_cycles(op, bs)
                cal[(op, bs)] = cyc
                print(f"  cal[{op:13s} bs={bs:4d}] = {cyc:8.1f} cycles/batch", flush=True)
            except Exception as e:
                print(f"  cal[{op:13s} bs={bs:4d}] FAILED: {e}", flush=True)
                cal[(op, bs)] = float('nan')

    # 2) corpus
    duties = [100] if smoke else DUTY_CYCLES
    cells = CELLS[:2] if smoke else CELLS
    ops_run = OPS[:2] if smoke else OPS
    n_reps = 5 if smoke else N_REPS

    rows = []
    n_total = len(duties) * len(ops_run) * len(cells)
    n_done = 0
    t0 = time.time()
    for duty in duties:
        for op in ops_run:
            for (wgp, bs) in cells:
                cyc = cal.get((op, bs), float('nan'))
                if cyc != cyc:  # nan
                    n_done += 1; continue
                fn, expected, sleep_reps = launch(op, wgp, bs, BATCHES_PER_PGM, duty, cyc)
                # warmup
                for _ in range(N_WARMUP):
                    fn()
                torch.cuda.synchronize()
                # timed reps
                lat_ms = []
                for r in range(n_reps):
                    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                    s.record(); fn(); e.record()
                    torch.cuda.synchronize()
                    lat_ms.append(s.elapsed_time(e))
                for r, lm in enumerate(lat_ms):
                    rows.append({
                        'duty_pct': duty, 'op_class': op, 'wgp_count': wgp,
                        'block_size': bs, 'rep': r, 'latency_ms': lm,
                        'expected_atoms': expected, 'sleep_reps': sleep_reps,
                        'active_cycles_per_batch': cyc,
                        'host': host, 'gpu': gpu_name,
                        'ts': time.time(),
                    })
                n_done += 1
                med = percentile(lat_ms, 50)
                p99 = percentile(lat_ms, 99)
                print(f"  [{n_done:3d}/{n_total}] duty={duty:3d}% op={op:13s} wgp={wgp:3d} bs={bs:4d} "
                      f"sleep_reps={sleep_reps:3d} med={med:.4f}ms p99={p99:.4f}ms", flush=True)

    # write CSV
    fields = list(rows[0].keys()) if rows else []
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[K-2380] wrote {len(rows)} rows in {time.time()-t0:.1f}s -> {out_path}", flush=True)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    run(a.out, smoke=a.smoke)
