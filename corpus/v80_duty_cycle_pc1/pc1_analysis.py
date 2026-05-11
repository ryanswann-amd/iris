"""K-2380 PC1 stratification analysis.

Per duty-cycle stratum: aggregate per-cell to median ordering-cost (per-atom µs),
form an (op_class × cell) matrix, run PCA on log-latency, report:
  - PC1 variance explained (compare to K-2317 baseline 95.8%)
  - PC1 loadings on op_class axis (cosine vs K-2317 reference)
  - p99/p50 ratio per cell (queue-drainage diagnostic)
"""
import sys, json, os
import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else 'output/k2380_corpus.csv'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'output'

df = pd.read_csv(CSV)
df['cell'] = df['wgp_count'].astype(str) + 'x' + df['block_size'].astype(str)
# ordering-cost per atom (µs)
df['ordcost_us'] = (df['latency_ms'] * 1e3) / df['expected_atoms']

OPS = ['XCHG_ACQREL', 'MAX_ACQREL', 'CAS_ACQREL', 'FADD_RELEASE']
CELLS = sorted(df['cell'].unique(), key=lambda s: (int(s.split('x')[0]), int(s.split('x')[1])))
DUTIES = sorted(df['duty_pct'].unique(), reverse=True)

# Per-(duty, op, cell) aggregate
agg = (df.groupby(['duty_pct', 'op_class', 'cell'])['ordcost_us']
         .agg(['median', lambda v: np.percentile(v, 99), 'std', 'count'])
         .reset_index().rename(columns={'<lambda_0>': 'p99'}))
agg['p99_p50'] = agg['p99'] / agg['median']
agg.to_csv(f'{OUT}/agg_per_cell.csv', index=False)
print('=== aggregate per (duty, op, cell) — first 15 rows ===')
print(agg.head(15).to_string(index=False))

# PCA per duty-cycle stratum
def pca_loadings(M):
    # M shape (n_ops, n_cells), values = log(median ordcost)
    Mc = M - M.mean(axis=0, keepdims=True)
    # SVD on columns-as-features → loadings = right singular vectors (cells)
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    var = (S**2) / (S**2).sum()
    pc1_loading_ops = U[:, 0]  # length n_ops — the carrier on the op axis
    pc1_loading_cells = Vt[0, :]
    return var, pc1_loading_ops, pc1_loading_cells

results = []
matrix_per_duty = {}
for duty in DUTIES:
    sub = agg[agg['duty_pct'] == duty]
    M = np.zeros((len(OPS), len(CELLS)))
    for i, op in enumerate(OPS):
        for j, c in enumerate(CELLS):
            row = sub[(sub['op_class'] == op) & (sub['cell'] == c)]
            if len(row) == 0:
                M[i, j] = np.nan
            else:
                M[i, j] = np.log(row['median'].iloc[0])
    matrix_per_duty[duty] = M
    if np.any(np.isnan(M)):
        print(f"duty={duty}% has NaN cells, skipping PCA")
        continue
    var, lo_op, lo_cell = pca_loadings(M)
    results.append({
        'duty_pct': duty,
        'pc1_var_explained': var[0],
        'pc2_var_explained': var[1] if len(var) > 1 else 0.0,
        **{f'load_op_{op}': float(lo_op[i]) for i, op in enumerate(OPS)},
    })

res = pd.DataFrame(results)
print('\n=== PCA per duty-cycle stratum ===')
print(res.to_string(index=False))

# Cosine drift vs sustained (100%) baseline
baseline = res[res['duty_pct'] == 100].iloc[0]
b_load = np.array([baseline[f'load_op_{op}'] for op in OPS])
drifts = []
for _, row in res.iterrows():
    v = np.array([row[f'load_op_{op}'] for op in OPS])
    cos = float(np.dot(b_load, v) / (np.linalg.norm(b_load) * np.linalg.norm(v) + 1e-12))
    drifts.append({'duty_pct': row['duty_pct'], 'pc1_cosine_vs_sustained': cos,
                   'pc1_var_explained': row['pc1_var_explained']})
drift_df = pd.DataFrame(drifts)
print('\n=== drift of PC1 op-loading vs sustained (100%) baseline ===')
print(drift_df.to_string(index=False))

# p99/p50 divergence vs sustained
print('\n=== p99/p50 by duty (mean across ops/cells) ===')
ratio = agg.groupby('duty_pct')['p99_p50'].agg(['mean', 'median', 'max']).reset_index()
print(ratio.to_string(index=False))

# Save
res.to_csv(f'{OUT}/pca_per_duty.csv', index=False)
drift_df.to_csv(f'{OUT}/pc1_drift.csv', index=False)
ratio.to_csv(f'{OUT}/p99_p50_by_duty.csv', index=False)

# Quick K-2317 baseline cross-check at 100% duty
K2317_PC1_BASELINE = 0.958
sustained_pc1 = res[res['duty_pct'] == 100]['pc1_var_explained'].iloc[0]
print(f'\n=== K-2317 comparison ===')
print(f'  K-2317 baseline PC1 var-explained: {K2317_PC1_BASELINE:.3f}')
print(f'  K-2380 sustained (100% duty)    : {sustained_pc1:.4f}')
print(f'  delta                           : {(sustained_pc1 - K2317_PC1_BASELINE)*100:+.1f} pp')

# Final verdict
min_pc1 = res['pc1_var_explained'].min()
max_drift_cos = drift_df['pc1_cosine_vs_sustained'].min()
verdict = 'HOLDS' if (min_pc1 > 0.90 and max_drift_cos > 0.95) else 'FALSIFIED'
print(f'\n=== VERDICT ===')
print(f'  min PC1 var-explained across strata : {min_pc1:.4f}  (threshold > 0.90)')
print(f'  min cosine vs sustained baseline    : {max_drift_cos:.4f}  (threshold > 0.95)')
print(f'  PC1 universality under duty-cycle   : {verdict}')

with open(f'{OUT}/verdict.json', 'w') as f:
    json.dump({
        'min_pc1_var': float(min_pc1),
        'min_cosine_vs_sustained': float(max_drift_cos),
        'verdict': verdict,
        'baseline_K2317': K2317_PC1_BASELINE,
        'sustained_pc1': float(sustained_pc1),
    }, f, indent=2)
