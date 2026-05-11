#!/usr/bin/env python3
"""K-2446 analysis: PC1 ordering-cost universality test under DCC strata.

For each DCC mode, build the (ordering x config) cost matrix where config = (block, wgp).
Compute PCA. PC1 variance-explained (PC1-VE) and the canonical ordering-rank vector
(RELAXED < ACQUIRE < ACQ_REL <= SEQ_CST) determine universality.

Procrustes (theta, rho) vs K-2399 atlas: theta = principal-axis rotation angle (radians)
between this DCC stratum's PC1 loading and the K-2399 reference; rho = correlation of
loadings.

K-2427 closed-form PC1-VE predictor (held-out test):
   PC1-VE_pred = 1 - sigma_eff^2 / (1 + sigma_eff^2),  sigma_eff = c_order * (1+kappa_dcc)
where c_order is the inter-ordering coefficient-of-variation and kappa_dcc reflects
metadata-cache miss-driven dispersion.
"""
import argparse, csv, json, os, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORDERINGS = ['RELAXED', 'ACQUIRE', 'ACQ_REL', 'SEQ_CST']
DCC_MODES = ['dcc_disabled', 'dcc_uncompressed', 'dcc_2to1', 'dcc_4to1']

# K-2399 atlas reference (PC1 loadings for the 11 universality-preserving axes
# all cluster around this canonical ordering-cost vector). Drawn from prior atlas.
K2399_ATLAS_PC1 = np.array([0.42, 0.49, 0.54, 0.55])  # RELAXED, ACQUIRE, ACQ_REL, SEQ_CST
K2399_ATLAS_PC1 /= np.linalg.norm(K2399_ATLAS_PC1)

# K-2399 distorting axes (PC1 loadings — these are the 2 known PC1-distorting axes)
K2399_DISTORTING = [
    np.array([0.62, 0.30, 0.50, 0.52]),  # axis A: amplifies RELAXED
    np.array([0.20, 0.40, 0.60, 0.66]),  # axis B: amplifies SEQ_CST tail
]
for i, v in enumerate(K2399_DISTORTING):
    K2399_DISTORTING[i] = v / np.linalg.norm(v)


def pc1_for_stratum(df: pd.DataFrame):
    """Build (config x ordering) matrix of mean-of-reps cost; return PC1 loadings + VE."""
    # config = block,wgp,rep -> use mean over reps
    grp = df.groupby(['block', 'wgp', 'ordering'])['mean_us'].mean().reset_index()
    pivot = grp.pivot(index=['block', 'wgp'], columns='ordering', values='mean_us')
    pivot = pivot[ORDERINGS]  # ensure column order
    # Center
    X = pivot.values
    Xc = X - X.mean(axis=0, keepdims=True)
    # Cov over orderings (4x4)
    cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    pc1 = eigvecs[:, 0]
    # Sign-align so loadings are mostly positive
    if pc1.sum() < 0:
        pc1 = -pc1
    pc1_ve = float(eigvals[0] / eigvals.sum()) if eigvals.sum() > 0 else 0.0
    # Ordering-rank check: are mean costs monotone increasing in canonical order?
    means = pivot.mean(axis=0).values
    return {
        'pc1_loadings': pc1.tolist(),
        'pc1_ve': pc1_ve,
        'eigvals': eigvals.tolist(),
        'mean_cost_by_order': means.tolist(),
        'rank_violation': bool(not all(means[i] <= means[i+1] + 0.5 for i in range(3))),
    }


def procrustes_theta_rho(v_ref: np.ndarray, v_obs: np.ndarray):
    """Compute angle theta (rad) and Pearson rho between ref and obs loading vectors."""
    a = v_ref / np.linalg.norm(v_ref)
    b = v_obs / np.linalg.norm(v_obs)
    cos = float(np.clip(a @ b, -1.0, 1.0))
    theta = math.acos(abs(cos))
    rho = float(np.corrcoef(v_ref, v_obs)[0, 1])
    return theta, rho


def k2427_predictor(df_strat: pd.DataFrame):
    """K-2427 closed-form PC1-VE predictor.
       sigma_eff = c_order * (1 + kappa_dcc)
       PC1-VE_pred = 1/(1 + sigma_eff^2)  -- normalized so high c_order => high PC1-VE.
    """
    by_order = df_strat.groupby('ordering')['mean_us'].mean()
    c_order = float(by_order.std() / by_order.mean()) if by_order.mean() > 0 else 0.0
    miss = df_strat['tcc_dcc_miss'].sum()
    req = df_strat['tcp_tcc_atomic_req'].sum()
    miss_rate = float(miss / max(req, 1))
    kappa_dcc = miss_rate * 0.5  # coupling constant from K-2427
    sigma_eff = c_order * (1 + kappa_dcc)
    pc1_ve_pred = 1.0 / (1.0 + sigma_eff ** 2) if sigma_eff > 0 else 1.0
    # K-2427 also predicts that PC1-VE >= 1 - sigma_eff^2
    return pc1_ve_pred, c_order, kappa_dcc, miss_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df.status == 'ok'].copy()
    print(f"Loaded {len(df)} rows")

    # Per-stratum PC1
    summary = []
    for dcc in DCC_MODES:
        sub = df[df.dcc_mode == dcc]
        if len(sub) == 0:
            continue
        pc = pc1_for_stratum(sub)
        # Procrustes vs K-2399 atlas
        theta_atlas, rho_atlas = procrustes_theta_rho(K2399_ATLAS_PC1, np.array(pc['pc1_loadings']))
        # vs distorting axes
        d_thetas = []
        d_rhos = []
        for v in K2399_DISTORTING:
            t, r = procrustes_theta_rho(v, np.array(pc['pc1_loadings']))
            d_thetas.append(t); d_rhos.append(r)
        theta_dist_min = min(d_thetas)
        # K-2427 predictor
        pc1_pred, c_order, kappa_dcc, miss_rate = k2427_predictor(sub)
        # Cluster classification: closer to atlas (universality-preserving) or to distorting axes?
        cluster = 'universality_preserving' if theta_atlas < theta_dist_min else 'pc1_distorting'
        summary.append({
            'dcc_mode': dcc,
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
            'kappa_dcc': round(kappa_dcc, 4),
            'dcc_miss_rate': round(miss_rate, 4),
        })

    os.makedirs(args.outdir, exist_ok=True)
    proc_path = os.path.join(args.outdir, 'procrustes_summary.csv')
    with open(proc_path, 'w', newline='') as f:
        # Flatten lists for CSV
        rows = []
        for s in summary:
            r = dict(s)
            r['pc1_loadings'] = ';'.join(str(x) for x in s['pc1_loadings'])
            r['mean_cost_us_by_order'] = ';'.join(str(x) for x in s['mean_cost_us_by_order'])
            rows.append(r)
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    json_path = os.path.join(args.outdir, 'procrustes_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {proc_path} and {json_path}")
    for s in summary:
        print(f"  {s['dcc_mode']:18s} PC1-VE={s['pc1_ve']:.3f}  pred={s['pc1_ve_pred_k2427']:.3f}  "
              f"theta_atlas={s['theta_vs_atlas_rad']:.3f}  cluster={s['cluster_assignment']}")

    # Plots
    # 1. PC1 loadings per DCC mode
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ORDERINGS))
    width = 0.18
    for i, s in enumerate(summary):
        ax.bar(x + i*width - 1.5*width, s['pc1_loadings'], width, label=s['dcc_mode'])
    ax.plot(x, K2399_ATLAS_PC1, 'k--o', label='K-2399 atlas (universality-preserving)')
    for j, v in enumerate(K2399_DISTORTING):
        ax.plot(x, v, ':', alpha=0.6, label=f'K-2399 distorting axis {chr(65+j)}')
    ax.set_xticks(x); ax.set_xticklabels(ORDERINGS)
    ax.set_ylabel('PC1 loading')
    ax.set_title('K-2446: PC1 loadings vs K-2399 atlas (gfx942 MI300X)')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, 'pc1_loadings.png'), dpi=120); plt.close()

    # 2. Mean cost by ordering, per DCC mode
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in summary:
        ax.plot(ORDERINGS, s['mean_cost_us_by_order'], '-o', label=s['dcc_mode'])
    ax.set_ylabel('mean kernel time (us)')
    ax.set_title('Mean atomic-RMW cost per ordering across DCC strata')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, 'cost_by_ordering.png'), dpi=120); plt.close()

    # 3. Procrustes scatter: theta_atlas vs theta_dist
    fig, ax = plt.subplots(figsize=(7, 6))
    for s in summary:
        ax.scatter(s['theta_vs_atlas_rad'], s['theta_vs_dist_min_rad'], s=120, label=s['dcc_mode'])
        ax.annotate(s['dcc_mode'], (s['theta_vs_atlas_rad'], s['theta_vs_dist_min_rad']),
                    textcoords='offset points', xytext=(6, 6), fontsize=8)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='atlas==distorting boundary')
    ax.set_xlabel('theta vs K-2399 atlas (rad)')
    ax.set_ylabel('theta vs nearest K-2399 distorting axis (rad)')
    ax.set_title('Procrustes signature: DCC strata vs K-2399 known clusters')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, 'procrustes_signature.png'), dpi=120); plt.close()

    # 4. K-2427 predictor residuals
    fig, ax = plt.subplots(figsize=(8, 5))
    obs = [s['pc1_ve'] for s in summary]
    pred = [s['pc1_ve_pred_k2427'] for s in summary]
    labels = [s['dcc_mode'] for s in summary]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, obs, 0.4, label='observed PC1-VE')
    ax.bar(x + 0.2, pred, 0.4, label='K-2427 predictor')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel('PC1 variance explained')
    ax.set_title('K-2446: observed vs K-2427 closed-form PC1-VE')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, 'k2427_predictor.png'), dpi=120); plt.close()
    print("Wrote 4 PNG plots")


if __name__ == '__main__':
    main()
