"""K-2446 RECONCILED analysis: PC1 ordering-cost universality test under
buffer-alignment×stride-spread strata (the as-encoded axis), with REAL rocprof
PMC subset showing HSA_ENABLE_DCC is no-op and the strata exercise L2 access
pattern (not DCC metadata-cache compression).

Reads the prior 12,800-row sweep + the new 24-row REAL rocprof PMC subset.
Produces:
 - reconciled_sweep.csv   : 12,800-row sweep with synthetic PMC columns flagged
 - rocprof_pmc.csv (already exists) : REAL 24-row PMC subset
 - procrustes_summary.{csv,json} : per-stratum PCA + Procrustes
 - 4 plots
"""
import argparse, csv, json, math, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORDERINGS = ['RELAXED', 'ACQUIRE', 'ACQ_REL', 'SEQ_CST']
STRATA = ['dcc_disabled', 'dcc_uncompressed', 'dcc_2to1', 'dcc_4to1']  # original labels (kept for traceability)
# Reinterpreted physical meaning (alignment, stride_mult)
STRATA_PHYSICAL = {
    'dcc_disabled': '(64B align, stride×1)',
    'dcc_uncompressed': '(256B align, stride×2)',
    'dcc_2to1': '(1024B align, stride×4)',
    'dcc_4to1': '(4096B align, stride×8)',
}
# K-2399 atlas reference (universality-preserving cluster)
K2399_ATLAS_PC1 = np.array([0.42, 0.49, 0.54, 0.55])
K2399_ATLAS_PC1 /= np.linalg.norm(K2399_ATLAS_PC1)
K2399_DISTORTING = [
    np.array([0.62, 0.30, 0.50, 0.52]),  # axis A
    np.array([0.20, 0.40, 0.60, 0.66]),  # axis B
]
K2399_DISTORTING = [v / np.linalg.norm(v) for v in K2399_DISTORTING]


def pc1_for_stratum(df):
    g = df.groupby(['block', 'wgp', 'ordering'])['mean_us'].mean().reset_index()
    pivot = g.pivot(index=['block', 'wgp'], columns='ordering', values='mean_us')[ORDERINGS]
    X = pivot.values
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    ev, evec = np.linalg.eigh(cov)
    o = np.argsort(ev)[::-1]
    ev = ev[o]; evec = evec[:, o]
    pc1 = evec[:, 0]
    if pc1.sum() < 0: pc1 = -pc1
    pc1_ve = float(ev[0] / ev.sum()) if ev.sum() > 0 else 0.0
    means = pivot.mean(axis=0).values
    return {
        'pc1_loadings': pc1.tolist(),
        'pc1_ve': pc1_ve,
        'eigvals': ev.tolist(),
        'mean_cost_by_order': means.tolist(),
        'rank_violation': bool(not all(means[i] <= means[i+1] + 0.5 for i in range(3))),
    }


def procrustes(v_ref, v_obs):
    a = v_ref / np.linalg.norm(v_ref); b = v_obs / np.linalg.norm(v_obs)
    cos = float(np.clip(a @ b, -1.0, 1.0))
    theta = math.acos(abs(cos))
    rho = float(np.corrcoef(v_ref, v_obs)[0, 1])
    return theta, rho


def k2427_predictor(df_strat, real_miss_rate=None):
    """K-2427 closed-form. Use REAL miss rate from rocprof if provided, else 0."""
    by_order = df_strat.groupby('ordering')['mean_us'].mean()
    c_order = float(by_order.std() / by_order.mean()) if by_order.mean() > 0 else 0.0
    miss_rate = float(real_miss_rate) if real_miss_rate is not None else 0.0
    # K-2427 form: PC1-VE_pred = 1/(1+sigma^2), sigma = c_order * (1+kappa)
    # NOTE: kappa is a tunable coupling — leave at 0.5*miss_rate per prior
    kappa = miss_rate * 0.5
    sigma_eff = c_order * (1 + kappa)
    pc1_ve_pred = 1.0 / (1.0 + sigma_eff ** 2) if sigma_eff > 0 else 1.0
    return pc1_ve_pred, c_order, kappa, miss_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep-csv', required=True)
    ap.add_argument('--pmc-csv', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.sweep_csv)
    df = df[df.status == 'ok'].copy()
    print(f"Loaded {len(df)} rows from {args.sweep_csv}")

    # Real PMC: aggregate per stratum (env=0 baseline only; env=0 vs 1 verified equal)
    pmc = pd.read_csv(args.pmc_csv)
    pmc = pmc[pmc.status == 'ok'].copy()
    pmc_e0 = pmc[pmc.hsa_enable_dcc == 0]
    real_miss_rate = {}
    for s in STRATA:
        sub = pmc_e0[pmc_e0.dcc_mode == s]
        if len(sub) == 0:
            real_miss_rate[s] = None; continue
        h = sub['TCC_HIT_sum'].astype(float).mean()
        m = sub['TCC_MISS_sum'].astype(float).mean()
        a = sub['TCC_ATOMIC_sum'].astype(float).mean()
        real_miss_rate[s] = {'tcc_hit': h, 'tcc_miss': m, 'tcc_atomic': a,
                             'miss_rate': m/(h+m) if (h+m) > 0 else None}

    # Per-stratum analysis
    summary = []
    for s in STRATA:
        sub = df[df.dcc_mode == s]
        pc = pc1_for_stratum(sub)
        theta_atlas, rho_atlas = procrustes(K2399_ATLAS_PC1, np.array(pc['pc1_loadings']))
        d_thetas = []; d_rhos = []
        for v in K2399_DISTORTING:
            t, r = procrustes(v, np.array(pc['pc1_loadings']))
            d_thetas.append(t); d_rhos.append(r)
        theta_dist_min = min(d_thetas)
        cluster = 'universality_preserving' if theta_atlas < theta_dist_min else 'pc1_distorting'
        rmr = real_miss_rate.get(s) or {}
        miss = rmr.get('miss_rate')
        pc1_pred, c_order, kappa, miss_used = k2427_predictor(sub, miss)
        summary.append({
            'stratum_label_orig': s,
            'stratum_physical': STRATA_PHYSICAL[s],
            'n_rows': len(sub),
            'pc1_ve': round(pc['pc1_ve'], 4),
            'pc1_ve_pred_k2427': round(pc1_pred, 4),
            'pc1_ve_residual': round(pc['pc1_ve'] - pc1_pred, 4),
            'pc1_loadings': [round(x, 4) for x in pc['pc1_loadings']],
            'mean_cost_us_by_order': [round(x, 3) for x in pc['mean_cost_by_order']],
            'rank_violation': pc['rank_violation'],
            'theta_vs_atlas_rad': round(theta_atlas, 4),
            'rho_vs_atlas': round(rho_atlas, 4),
            'theta_vs_dist_min_rad': round(theta_dist_min, 4),
            'cluster_assignment': cluster,
            'c_order': round(c_order, 4),
            'kappa_dcc': round(kappa, 4),
            'real_tcc_hit_mean': round(rmr.get('tcc_hit'), 1) if rmr.get('tcc_hit') is not None else None,
            'real_tcc_miss_mean': round(rmr.get('tcc_miss'), 1) if rmr.get('tcc_miss') is not None else None,
            'real_tcc_atomic_mean': round(rmr.get('tcc_atomic'), 1) if rmr.get('tcc_atomic') is not None else None,
            'real_l2_miss_rate': round(rmr.get('miss_rate'), 4) if rmr.get('miss_rate') is not None else None,
        })

    # Write CSV + JSON
    proc_csv = os.path.join(args.outdir, 'procrustes_summary.csv')
    rows = []
    for s in summary:
        r = dict(s)
        r['pc1_loadings'] = ';'.join(str(x) for x in s['pc1_loadings'])
        r['mean_cost_us_by_order'] = ';'.join(str(x) for x in s['mean_cost_us_by_order'])
        rows.append(r)
    with open(proc_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    proc_json = os.path.join(args.outdir, 'procrustes_summary.json')
    with open(proc_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {proc_csv}, {proc_json}")
    for s in summary:
        print(f"  {s['stratum_label_orig']:18s} {s['stratum_physical']:30s} PC1-VE={s['pc1_ve']:.3f} "
              f"theta_atlas={s['theta_vs_atlas_rad']:.3f} cluster={s['cluster_assignment']} "
              f"real_miss_rate={s['real_l2_miss_rate']}")

    # Reconciled CSV: original + flag synthetic PMC columns
    recon_path = os.path.join(args.outdir, 'reconciled_sweep.csv')
    df_out = df.copy()
    df_out.rename(columns={
        'tcc_dcc_hit': 'tcc_dcc_hit_SYNTHETIC',
        'tcc_dcc_miss': 'tcc_dcc_miss_SYNTHETIC',
        'tcp_tcc_atomic_req': 'tcp_tcc_atomic_req_SYNTHETIC',
        'sq_wait_proxy': 'sq_wait_proxy_SYNTHETIC',
    }, inplace=True)
    df_out['stratum_physical'] = df_out['dcc_mode'].map(STRATA_PHYSICAL)
    df_out.to_csv(recon_path, index=False)
    print(f"Wrote {recon_path} ({len(df_out)} rows)")

    # Plots
    # 1. PC1 loadings
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(ORDERINGS)); width = 0.18
    for i, s in enumerate(summary):
        ax.bar(x + i*width - 1.5*width, s['pc1_loadings'], width,
               label=f"{s['stratum_label_orig']} {s['stratum_physical']}")
    ax.plot(x, K2399_ATLAS_PC1, 'k--o', label='K-2399 atlas (universality-preserving)')
    for j, v in enumerate(K2399_DISTORTING):
        ax.plot(x, v, ':', alpha=0.6, label=f'K-2399 distorting axis {chr(65+j)}')
    ax.set_xticks(x); ax.set_xticklabels(ORDERINGS)
    ax.set_ylabel('PC1 loading')
    ax.set_title('K-2446: PC1 loadings under alignment×stride-spread strata\n(NOT actual DCC — strata reinterpreted; HSA_ENABLE_DCC verified no-op)')
    ax.legend(loc='upper left', fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, 'pc1_loadings.png'), dpi=120); plt.close()

    # 2. Mean cost by ordering
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in summary:
        ax.plot(ORDERINGS, s['mean_cost_us_by_order'], '-o',
                label=f"{s['stratum_label_orig']} {s['stratum_physical']}")
    ax.set_ylabel('mean kernel time (us)')
    ax.set_title('Mean atomic-RMW cost per ordering across strata')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, 'cost_by_ordering.png'), dpi=120); plt.close()

    # 3. Procrustes signature
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for s in summary:
        ax.scatter(s['theta_vs_atlas_rad'], s['theta_vs_dist_min_rad'], s=140, label=s['stratum_label_orig'])
        ax.annotate(s['stratum_label_orig'], (s['theta_vs_atlas_rad'], s['theta_vs_dist_min_rad']),
                    textcoords='offset points', xytext=(8, 6), fontsize=9)
    lim = max(0.7, max(s['theta_vs_atlas_rad'] for s in summary)*1.1)
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.4, label='atlas==distorting boundary')
    ax.set_xlabel('theta vs K-2399 atlas (rad)')
    ax.set_ylabel('theta vs nearest K-2399 distorting axis (rad)')
    ax.set_title('Procrustes signature: as-encoded strata vs K-2399 clusters')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, 'procrustes_signature.png'), dpi=120); plt.close()

    # 4. K-2427 predictor (now using REAL miss_rate)
    fig, ax = plt.subplots(figsize=(8, 5))
    obs = [s['pc1_ve'] for s in summary]
    pred = [s['pc1_ve_pred_k2427'] for s in summary]
    labels = [s['stratum_label_orig'] for s in summary]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, obs, 0.4, label='observed PC1-VE')
    ax.bar(x + 0.2, pred, 0.4, label='K-2427 predictor (with REAL L2 miss rate)')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel('PC1 variance explained')
    ax.set_title('K-2446: observed vs K-2427 predictor\n(uses REAL TCC_MISS/TCC_HIT rates from rocprofv2)')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, 'k2427_predictor.png'), dpi=120); plt.close()

    # 5. NEW plot: HSA_ENABLE_DCC no-op evidence
    fig, ax = plt.subplots(figsize=(9, 5))
    pmc_e0_grouped = pmc[pmc.hsa_enable_dcc == 0].groupby('dcc_mode')[['TCC_HIT_sum','TCC_MISS_sum']].mean()
    pmc_e1_grouped = pmc[pmc.hsa_enable_dcc == 1].groupby('dcc_mode')[['TCC_HIT_sum','TCC_MISS_sum']].mean()
    x = np.arange(len(STRATA))
    ax.bar(x - 0.4, [pmc_e0_grouped.loc[s, 'TCC_MISS_sum'] for s in STRATA], 0.18, label='TCC_MISS env=0')
    ax.bar(x - 0.22, [pmc_e1_grouped.loc[s, 'TCC_MISS_sum'] for s in STRATA], 0.18, label='TCC_MISS env=1')
    ax.bar(x + 0.0, [pmc_e0_grouped.loc[s, 'TCC_HIT_sum'] for s in STRATA], 0.18, label='TCC_HIT env=0')
    ax.bar(x + 0.18, [pmc_e1_grouped.loc[s, 'TCC_HIT_sum'] for s in STRATA], 0.18, label='TCC_HIT env=1')
    ax.set_xticks(x); ax.set_xticklabels(STRATA, rotation=15)
    ax.set_ylabel('PMC counter (rocprofv2 sum)')
    ax.set_title('HSA_ENABLE_DCC effect on TCC counters (env=0 vs env=1) — visually identical = no-op')
    ax.set_yscale('log'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, 'hsa_dcc_noop.png'), dpi=120); plt.close()

    print(f"Wrote 5 PNG plots to {args.outdir}")


if __name__ == '__main__':
    main()
