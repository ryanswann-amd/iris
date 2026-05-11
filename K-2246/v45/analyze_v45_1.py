#!/usr/bin/env python3
"""
K-2246 retry — v45.1 analysis: combine the v45 anchor + event-timed paired sweep
on b21u01, build the corrected ladder, recompute deltas, write the v45.1 manifest
and 4 plots.
"""
from __future__ import annotations
import csv, json, hashlib, os, statistics
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/workspace/output'

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1<<16), b''): h.update(chunk)
    return h.hexdigest()

def load_csv(p):
    with open(p) as fh: return list(csv.DictReader(fh))

# -------- anchor: combine the two anchor runs (200 + 500 reps = 700 reps/prim) --------
anchor_rows = load_csv(f'{OUT}/v45_anchor.csv') + load_csv(f'{OUT}/v45_anchor_run2.csv')
by_prim = defaultdict(list)
for r in anchor_rows:
    by_prim[r['prim']].append(float(r['us']))
anchor = {p: {
    'n': len(v),
    'median_us': round(statistics.median(v), 4),
    'mean_us': round(statistics.mean(v), 4),
    'p25': round(sorted(v)[len(v)//4], 4),
    'p75': round(sorted(v)[3*len(v)//4], 4),
    'min_us': round(min(v), 4),
    'max_us': round(max(v), 4),
} for p, v in by_prim.items()}

# Order ladder by ordering pair
LADDER_ORDER = ['M3', 'N3', 'O3', 'F3', 'P3']
LADDER_NAMES = {
    'M3': 'M3 (relaxed-load + relaxed-CAS)',
    'N3': 'N3 (acquire-load + acquire-CAS)',
    'O3': 'O3 (relaxed-load + release-CAS)',
    'F3': 'F3 (acquire-load + acq_rel-CAS)',
    'P3': 'P3 (acq_rel-load + acq_rel-CAS)',
}

p3_med = anchor['P3']['median_us']
m3_med = anchor['M3']['median_us']
f3_med = anchor['F3']['median_us']
n3_med = anchor['N3']['median_us']
o3_med = anchor['O3']['median_us']

print('=== v45 in-corpus anchor (b21u01 / gfx942 / CUDA event timer) ===')
for k in LADDER_ORDER:
    a = anchor[k]
    print(f"  {LADDER_NAMES[k]:42s}  median={a['median_us']:6.3f} µs  n={a['n']:4d}  [p25={a['p25']:.2f} p75={a['p75']:.2f}]")

deltas = {
    'P3 - F3': round(p3_med - f3_med, 4),
    'P3 - M3': round(p3_med - m3_med, 4),
    'P3 - N3': round(p3_med - n3_med, 4),
    'P3 - O3': round(p3_med - o3_med, 4),
    'F3 - M3': round(f3_med - m3_med, 4),
    'N3 - M3': round(n3_med - m3_med, 4),
    'O3 - M3': round(o3_med - m3_med, 4),
}

# -------- paired (event-timed, focal-only): combine the two runs (200 + 500 = 700 reps each) --------
paired_rows = load_csv(f'{OUT}/v45_paired_canonical_event.csv') + load_csv(f'{OUT}/v45_paired_canonical_event_run2.csv')
by_inter = defaultdict(list)
for r in paired_rows:
    by_inter[r['interferer']].append(float(r['us']))
paired = {i: {
    'n': len(v),
    'median_us': round(statistics.median(v), 4),
    'p25': round(sorted(v)[len(v)//4], 4),
    'p75': round(sorted(v)[3*len(v)//4], 4),
} for i, v in by_inter.items()}

p3_focal_baseline = paired['__none__']['median_us']
print(f"\n=== v45 paired-event (focal-stream CUDA events, focal-only latency under contention) ===")
print(f"  P3 focal-only baseline:  median={p3_focal_baseline:.3f} µs  n={paired['__none__']['n']}")
print(f"  (vs v45 in-corpus anchor P3 = {p3_med:.3f} µs — agreement check)")

INTERFERER_ORDER = sorted(
    [k for k in paired if k != '__none__'],
    key=lambda k: paired[k]['median_us'],
)
print('\n  interferer  paired-µs   Δ-vs-baseline')
for inter in INTERFERER_ORDER:
    p = paired[inter]
    delta = round(p['median_us'] - p3_focal_baseline, 4)
    print(f"   {inter:5s}      {p['median_us']:6.2f}     +{delta:6.3f}")

# -------- compute prior wall-clock numbers for cross-comparison --------
# also load the original v45_baseline (wall-clock) for the "before" canonical median
old_baseline = load_csv(f'{OUT}/v45_baseline.csv')
old_canon = [float(r['us']) for r in old_baseline if r['K']=='4' and r['N_PROD']=='4' and r['N_OPS']=='32']
old_canon_med = round(statistics.median(old_canon), 4) if old_canon else None
print(f"\n  (prior wall-clock P3 canonical median:  {old_canon_med:.3f} µs;"
      f" delta = {p3_med - old_canon_med:+.3f} µs — measurement-method correction)")

# -------- build the v45.1 manifest --------
manifest = {
    'task': 'K-2246',
    'corpus_version': 'v45.1',
    'predecessor_corpus_version': 'v45',
    'fix_summary': [
        'Adds in-corpus anchor: M3,N3,O3,F3,P3 re-measured at canonical cell on b21u01 (same host as v45 baseline)',
        'Replaces wall-clock paired timing with focal-stream CUDA events (measures focal P3 latency under contention, not max(focal,interferer))',
        'Reconciles the +23.27 vs +24.11 inconsistency: those numbers came from cross-host wall-clock; v45.1 reports event-timed in-corpus deltas',
    ],
    'reviewer_addresses': {
        'Skeptic.paired_timing': 'Replaced perf_counter() wall-clock with focal-stream cudaEventRecord/elapsed_time pair; interferer launches BEFORE focal but is timed only via the focal stream events.  Sample script bench_p3_paired_events.py.',
        'Skeptic.cross_host_anchor': 'New v45_anchor.csv re-measures the FULL ladder (M3,N3,O3,F3,P3) on the same b21u01 host that produced v45_baseline.csv, so the P3-vs-F3 delta is now a single-host single-version comparison.',
        'UX.headline_consistency': 'Δ-vs-F3 and Δ-vs-M3 now reported in the same table with explicit labels; no more ambiguity between the two.',
    },
    'cluster': 'c42 (mi300x partition)',
    'node': 'b21u01',
    'gpu_arch': 'gfx942',
    'gpu_model': 'AMD Instinct MI300X',
    'container': 'rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0',
    'canonical_cell': {'K': 4, 'N_PROD': 4, 'N_OPS': 32},
    'anchor': {
        'csv': 'v45_anchor.csv (200 reps) + v45_anchor_run2.csv (500 reps) = 700 reps/prim',
        'sha256_v45_anchor': sha256(f'{OUT}/v45_anchor.csv'),
        'sha256_v45_anchor_run2': sha256(f'{OUT}/v45_anchor_run2.csv'),
        'timer': 'cuda_event (start.elapsed_time(end))',
        'medians_us': {p: anchor[p]['median_us'] for p in LADDER_ORDER},
        'n_per_prim': {p: anchor[p]['n'] for p in LADDER_ORDER},
        'p25_us': {p: anchor[p]['p25'] for p in LADDER_ORDER},
        'p75_us': {p: anchor[p]['p75'] for p in LADDER_ORDER},
    },
    'ladder_deltas_us': deltas,
    'paired_canonical_event': {
        'csv': 'v45_paired_canonical_event.csv (200 reps) + v45_paired_canonical_event_run2.csv (500 reps) = 700 reps/cell',
        'sha256_run1': sha256(f'{OUT}/v45_paired_canonical_event.csv'),
        'sha256_run2': sha256(f'{OUT}/v45_paired_canonical_event_run2.csv'),
        'timer': 'focal_stream_cuda_event (interferer launched on stream B but timed only via focal stream A start/end events)',
        'p3_focal_only_baseline_us': p3_focal_baseline,
        'per_interferer': {
            inter: {
                'median_us': paired[inter]['median_us'],
                'delta_vs_focal_only_us': round(paired[inter]['median_us'] - p3_focal_baseline, 4),
                'n': paired[inter]['n'],
            } for inter in INTERFERER_ORDER
        },
    },
    'prior_v45_wall_clock_canonical_us': old_canon_med,
    'measurement_method_correction_us': round(old_canon_med - p3_med, 4),
    'comm_data_branch': 'K-2246-v45-p3-atomic-cas-acqrel (additive v45.1 commit; ryanswann-amd/iris fork)',
}

with open(f'{OUT}/v45_1_manifest.json', 'w') as fh:
    json.dump(manifest, fh, indent=2)
print(f'\nWrote {OUT}/v45_1_manifest.json')

# ============================== PLOTS ==============================

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f'{OUT}/{name}', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {name}')

# Plot 1: corrected CAS ordering ladder (event-timed, in-corpus)
fig, ax = plt.subplots(figsize=(8.5, 4.6))
xs = list(range(len(LADDER_ORDER)))
medians = [anchor[p]['median_us'] for p in LADDER_ORDER]
yerr_lo = [anchor[p]['median_us'] - anchor[p]['p25'] for p in LADDER_ORDER]
yerr_hi = [anchor[p]['p75'] - anchor[p]['median_us'] for p in LADDER_ORDER]
colors = ['#bbbbbb', '#80b1d3', '#fdb462', '#8dd3c7', '#fb8072']
bars = ax.bar(xs, medians, yerr=[yerr_lo, yerr_hi], color=colors,
              error_kw={'capsize': 4, 'elinewidth': 1.0})
for x, m in zip(xs, medians):
    ax.text(x, m + 0.5, f'{m:.2f}', ha='center', fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels([LADDER_NAMES[p].split(' ', 1)[1] for p in LADDER_ORDER],
                   rotation=20, ha='right', fontsize=8)
ax.set_ylabel('focal latency (µs, CUDA event timer)')
ax.set_title('K-2246 v45.1 — CAS memory-ordering ladder (in-corpus, b21u01, n=700 per prim)\n'
             f'P3 − F3 = {deltas["P3 - F3"]:+.2f} µs    P3 − M3 = {deltas["P3 - M3"]:+.2f} µs    '
             f'P3 − O3 = {deltas["P3 - O3"]:+.2f} µs')
ax.grid(axis='y', alpha=0.3)
save(fig, 'v45_1_ladder.png')

# Plot 2: paired-event canonical bars sorted
fig, ax = plt.subplots(figsize=(10, 4.8))
labels = ['__none__'] + INTERFERER_ORDER
ys = [paired[k]['median_us'] for k in labels]
xs = list(range(len(labels)))
deltas_bar = [0] + [paired[k]['median_us'] - p3_focal_baseline for k in INTERFERER_ORDER]
clr = ['#444444'] + ['#377eb8' if v < 5 else '#fdae6b' if v < 10 else '#d73027' for v in deltas_bar[1:]]
ax.bar(xs, ys, color=clr)
for x, y, d in zip(xs, ys, deltas_bar):
    ax.text(x, y + 0.5, f'{y:.1f}\n{("+%.1f"%d) if d else "(P3 alone)"}',
            ha='center', va='bottom', fontsize=7)
ax.set_xticks(xs)
ax.set_xticklabels(['P3-only'] + INTERFERER_ORDER, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('focal P3 latency under contention (µs, event-timer)')
ax.set_title('K-2246 v45.1 — P3 focal latency vs 17 paired interferers @ canonical cell '
             '(K=4, N_PROD=4, N_OPS=32, n=700)')
ax.axhline(p3_focal_baseline, color='black', linestyle='--', linewidth=0.8, alpha=0.6,
           label=f'P3 focal-only baseline = {p3_focal_baseline:.2f} µs')
ax.legend(loc='upper left', fontsize=8)
ax.grid(axis='y', alpha=0.3)
save(fig, 'v45_1_paired_canonical_event.png')

# Plot 3: before-vs-after measurement method comparison
fig, ax = plt.subplots(figsize=(8, 4.4))
methods = ['v45 prior\n(wall-clock,\ncross-host F3)', 'v45.1 fix\n(CUDA event,\nin-corpus F3)']
p3_vals = [old_canon_med, p3_med]
f3_vals = [34.36, f3_med]  # 34.36 is the v44 cross-host value used in prior summary
delta_vals = [old_canon_med - 34.36, p3_med - f3_med]
xs = [0, 1]
w = 0.27
ax.bar([x-w for x in xs], p3_vals, w, label='P3 median', color='#fb8072')
ax.bar(xs, f3_vals, w, label='F3 median', color='#8dd3c7')
ax.bar([x+w for x in xs], delta_vals, w, label='Δ = P3 − F3', color='#80b1d3')
for i in range(2):
    ax.text(xs[i]-w, p3_vals[i]+0.5, f'{p3_vals[i]:.2f}', ha='center', fontsize=8)
    ax.text(xs[i],  f3_vals[i]+0.5, f'{f3_vals[i]:.2f}', ha='center', fontsize=8)
    ax.text(xs[i]+w, delta_vals[i]+0.3, f'{delta_vals[i]:+.2f}', ha='center', fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('canonical-cell µs')
ax.set_title('K-2246 v45.1 — Measurement-method correction\n'
             '(prior reported +23.27 µs ordering tax; corrected anchor shows +%.2f µs)' % deltas['P3 - F3'])
ax.legend(loc='upper right', fontsize=8)
ax.grid(axis='y', alpha=0.3)
save(fig, 'v45_1_method_correction.png')

# Plot 4: paired delta histogram (Δ vs focal-only baseline) per interferer family
families = {
    'CAS-on-CAS (M3,N3,O3)':  ['M3','N3','O3'],
    'XCHG (L3,K3,J3,E3)':     ['L3','K3','J3','E3'],
    'INT-atomic (Y,G,D3)':    ['Y','G','D3'],
    'FP-atomic (G3,H3,I3)':   ['G3','H3','I3'],
    'sync (F,H,R2)':          ['F','H','R2'],
    'comm (P)':               ['P'],
}
fig, ax = plt.subplots(figsize=(9, 4.4))
xs, ys, clrs, lbls = [], [], [], []
fam_colors = ['#d73027','#fc8d59','#fee090','#91bfdb','#4575b4','#984ea3']
i = 0
for (fam, mems), c in zip(families.items(), fam_colors):
    for m in mems:
        xs.append(i)
        ys.append(paired[m]['median_us'] - p3_focal_baseline)
        clrs.append(c); lbls.append(m); i += 1
ax.bar(xs, ys, color=clrs)
for x, y in zip(xs, ys):
    ax.text(x, y + 0.2, f'{y:+.2f}', ha='center', fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels(lbls, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Δ vs P3 focal-only (µs)')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=c, label=f) for f,c in zip(families.keys(), fam_colors)],
          loc='upper left', fontsize=7)
ax.set_title('K-2246 v45.1 — Per-family contention overhead on focal P3 (event-timed)')
ax.grid(axis='y', alpha=0.3)
save(fig, 'v45_1_paired_by_family.png')

print('\nDone.')
