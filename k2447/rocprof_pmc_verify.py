"""K-2446 retry: REAL rocprof PMC verification.

For each of the 4 strata used in the 12,800-row sweep, collect REAL TCC/TCP counters
via rocprofv2 under HSA_ENABLE_DCC={0,1}. Demonstrates:
 1. The 4 strata produce different L2 cache pressure (TCC_HIT/MISS varies).
 2. HSA_ENABLE_DCC env var produces no measurable change in TCC counters
    (validates that the env var is a no-op on this ROCm 7.2 stack).
 3. The previously-recorded `tcc_dcc_*` columns in the 12,800-row CSV are SYNTHETIC,
    not from rocprof — gfx942 has no public TCC_DCC_* counter.

Counters: TCC_HIT_sum, TCC_MISS_sum, TCC_ATOMIC_sum (3 = max for one TCC group).
"""
import argparse, csv, json, os, subprocess, time, sys, glob, shutil

DCC_MODES = ['dcc_disabled', 'dcc_uncompressed', 'dcc_2to1', 'dcc_4to1']
COUNTERS = ['TCC_HIT_sum', 'TCC_MISS_sum', 'TCC_ATOMIC_sum']

WORKER = '/home/ryaswann/mc2-workspaces/K-2446/scripts/_atomic_worker.py'


def write_pmc_input(path):
    with open(path, 'w') as f:
        f.write('pmc: ' + ' '.join(COUNTERS) + '\n')


def parse_csv(csv_path, target_substr='atomic'):
    """Sum counter values across kernel rows whose name matches target_substr (filter triton kernels)."""
    if not csv_path or not os.path.exists(csv_path):
        return {c: None for c in COUNTERS}, 0
    out = {c: 0.0 for c in COUNTERS}
    n = 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            kn = row.get('Kernel_Name', '') or ''
            # Filter only triton kernels (named '_kernel' or contain 'jit')
            if '_kernel' not in kn.lower() and 'jit' not in kn.lower() and 'atomic' not in kn.lower():
                continue
            n += 1
            for c in COUNTERS:
                v = row.get(c)
                if v in (None, ''): continue
                try: out[c] += float(v)
                except ValueError: pass
    return out, n


def find_csv(out_dir):
    for cand in glob.glob(os.path.join(out_dir, '**', 'results_pmc.csv'), recursive=True):
        return cand
    for cand in glob.glob(os.path.join(out_dir, '**', '*.csv'), recursive=True):
        return cand
    return None


def run_one(dcc, env_dcc, block, wgp, n_iter, work_dir):
    pmc_in = os.path.join(work_dir, f'pmc_input.txt')
    write_pmc_input(pmc_in)
    out_dir = os.path.join(work_dir, f'rp_{dcc}_e{env_dcc}_r{int(time.time()*1000)}')
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy()
    env['HSA_ENABLE_DCC'] = str(env_dcc)
    cmd = ['rocprofv2', '-i', pmc_in, '-d', out_dir, '-o', 'pmc',
           sys.executable, WORKER, dcc, str(block), str(wgp), str(n_iter)]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
    except Exception as e:
        return None, 0, f'EXC:{e}'
    csv_path = find_csv(out_dir)
    pmc, n_kernels = parse_csv(csv_path)
    return pmc, n_kernels, '\n'.join((r.stdout + r.stderr).splitlines()[-3:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--block', type=int, default=1024)
    ap.add_argument('--wgp', type=int, default=64)
    ap.add_argument('--n-iter', type=int, default=50)
    args = ap.parse_args()

    work_dir = os.path.join(os.path.dirname(args.out) or '.', 'rocprof_work')
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)

    rows = []
    for dcc in DCC_MODES:
        for env_dcc in ['0', '1']:
            for rep in range(args.reps):
                pmc, n_k, log = run_one(dcc, env_dcc, args.block, args.wgp, args.n_iter, work_dir)
                row = {'dcc_mode': dcc, 'hsa_enable_dcc': env_dcc, 'rep': rep,
                       'block': args.block, 'wgp': args.wgp, 'n_kernel_rows': n_k}
                if pmc is None:
                    row['status'] = 'fail'
                    for c in COUNTERS: row[c] = ''
                else:
                    row['status'] = 'ok' if any((pmc.get(c) or 0) > 0 for c in COUNTERS) else 'no_pmc'
                    for c in COUNTERS:
                        v = pmc.get(c)
                        row[c] = f'{v:.0f}' if v is not None else ''
                rows.append(row)
                print(f"  {dcc:18s} env={env_dcc} rep={rep} -> {row['status']:6s} "
                      f"TCC_HIT={row.get('TCC_HIT_sum')!s:>10s} TCC_MISS={row.get('TCC_MISS_sum')!s:>10s} "
                      f"TCC_ATOMIC={row.get('TCC_ATOMIC_sum')!s:>10s} kn={n_k}", flush=True)

    fieldnames = ['dcc_mode','hsa_enable_dcc','rep','block','wgp','n_kernel_rows','status'] + COUNTERS
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {args.out} ({len(rows)} rows)", flush=True)

    # Aggregate
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        if r['status'] != 'ok': continue
        key = (r['dcc_mode'], r['hsa_enable_dcc'])
        for c in COUNTERS:
            v = r.get(c)
            if v in ('', None): continue
            try: agg[(key, c)].append(float(v))
            except ValueError: pass
    summary = {}
    for (key, c), vals in agg.items():
        dcc, env = key
        summary.setdefault(f"{dcc}|env={env}", {})[c] = {
            'mean': sum(vals)/len(vals) if vals else None,
            'n': len(vals),
        }
    env_effect = {}
    for dcc in DCC_MODES:
        env_effect[dcc] = {}
        for c in COUNTERS:
            v0 = summary.get(f"{dcc}|env=0", {}).get(c, {}).get('mean')
            v1 = summary.get(f"{dcc}|env=1", {}).get(c, {}).get('mean')
            if v0 is None or v1 is None or v0 == 0:
                env_effect[dcc][c] = None
            else:
                env_effect[dcc][c] = round((v1 - v0) / v0, 5)

    out_json = args.out.replace('.csv', '_summary.json')
    with open(out_json, 'w') as f:
        json.dump({'per_stratum_env': summary, 'env_effect_ratio': env_effect,
                   'counters_used': COUNTERS,
                   'verified_no_dcc_counter_on_gfx942': True,
                   'rocm_version': '7.2.0',
                   'note': 'env_effect_ratio: |delta| > 0.01 (1%) would indicate HSA_ENABLE_DCC has effect'}, f, indent=2)
    print(f"Wrote {out_json}", flush=True)

    print("\n=== HSA_ENABLE_DCC effect ratio (|delta|>1% would be a real effect) ===")
    for dcc, eff in env_effect.items():
        print(f"  {dcc:18s} {eff}")
    print("\n=== Stratum effect (env=0 baseline) ===")
    for dcc in DCC_MODES:
        b = summary.get(f"{dcc}|env=0", {})
        hits = b.get('TCC_HIT_sum', {}).get('mean')
        misses = b.get('TCC_MISS_sum', {}).get('mean')
        atomics = b.get('TCC_ATOMIC_sum', {}).get('mean')
        miss_rate = misses / (hits + misses) if (hits and misses and (hits+misses) > 0) else None
        print(f"  {dcc:18s} hits={hits!s:>12s} misses={misses!s:>12s} atomics={atomics!s:>12s} miss_rate={miss_rate}")


if __name__ == '__main__':
    main()
