#!/usr/bin/env python3
# K-2388 PC1 fit: per-stratum (each (prod_waves, cons_waves) imbalance) PCA across
# 4 ordering classes × (wgp,block,buffer) cells. Assess universality (PC1 VE ≥ 90%)
# and decide whether producer-side or consumer-side occupancy dominates the residual
# (compare PC1 loadings shift across imbalances and OLS log-time vs prod_w / cons_w).
import argparse, os, sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["us_per_call"])
    df = df[df["us_per_call"] > 0].copy()
    df["log_us"] = np.log(df["us_per_call"])
    return df


def pca_explained(M: np.ndarray, n_comp=4):
    if M.shape[0] < 2 or M.shape[1] < 2:
        return np.zeros(n_comp), np.zeros((max(1, M.shape[1]), n_comp))
    Mc = M - M.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    var = (S ** 2) / max(Mc.shape[0] - 1, 1)
    total = var.sum()
    if total <= 0:
        return np.zeros(n_comp), np.zeros((Mc.shape[1], n_comp))
    ratios = var / total
    out_r = np.zeros(n_comp)
    out_r[: min(n_comp, len(ratios))] = ratios[: min(n_comp, len(ratios))]
    out_l = np.zeros((Mc.shape[1], n_comp))
    out_l[:, : min(n_comp, Vt.shape[0])] = Vt[: min(n_comp, Vt.shape[0]), :].T
    return out_r, out_l


def per_stratum_pc1(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """For each (prod_num_warps, cons_num_warps) stratum, build a
    [cells × prims] matrix of per-cell median log-us, fit PCA."""
    rows = []
    loadings = {}
    df = df.copy()
    g = df.groupby(["prod_num_warps", "cons_num_warps", "ordering_class",
                    "n_workgroups", "block_size", "buffer_bytes"])["log_us"].median().reset_index()
    g = g.rename(columns={"log_us": "med_log_us"})
    for (p, c), gpc in g.groupby(["prod_num_warps", "cons_num_warps"]):
        wide = gpc.pivot_table(
            index=["n_workgroups", "block_size", "buffer_bytes"],
            columns="ordering_class",
            values="med_log_us",
            aggfunc="first",
        ).dropna(axis=0, how="any")
        n_cells, n_prims = wide.shape
        ratios, V = pca_explained(wide.values, n_comp=min(4, n_prims))
        rows.append(dict(
            prod_num_warps=int(p), cons_num_warps=int(c),
            imbalance=f"({p},{c})",
            n_cells=int(n_cells), n_prims=int(n_prims),
            pc1_ve=float(ratios[0]),
            pc2_ve=float(ratios[1] if n_prims > 1 else 0.0),
            pc3_ve=float(ratios[2] if n_prims > 2 else 0.0),
            pc1_pc2=float(ratios[0] + (ratios[1] if n_prims > 1 else 0.0)),
        ))
        loadings[f"{p}_{c}"] = dict(
            cols=list(wide.columns),
            loadings=V.tolist(),
        )
    out = pd.DataFrame(rows).sort_values(["prod_num_warps", "cons_num_warps"]).reset_index(drop=True)
    return out, loadings


def global_pc1(df: pd.DataFrame) -> dict:
    g = df.groupby(["ordering_class", "n_workgroups", "block_size",
                    "buffer_bytes", "prod_num_warps", "cons_num_warps"])["log_us"].median().reset_index()
    wide = g.pivot_table(
        index=["n_workgroups", "block_size", "buffer_bytes",
               "prod_num_warps", "cons_num_warps"],
        columns="ordering_class",
        values="log_us",
    ).dropna(axis=0, how="any")
    if wide.shape[0] < 2:
        return {}
    ratios, V = pca_explained(wide.values, n_comp=min(4, wide.shape[1]))
    return dict(
        n_cells=int(wide.shape[0]), n_prims=int(wide.shape[1]),
        pc1_ve=float(ratios[0]), pc2_ve=float(ratios[1]) if wide.shape[1] > 1 else 0.0,
        pc1_loadings={c: float(v) for c, v in zip(wide.columns, V[:, 0])},
        prims=list(wide.columns),
    )


def residual_decomp(df: pd.DataFrame) -> pd.DataFrame:
    """For each ordering class, regress log_us on log(prod_waves) and log(cons_waves)
    plus baseline cell fixed effects. Reports beta_prod, beta_cons, R²."""
    df = df.copy()
    df["log_prod_w"] = np.log(df["prod_num_warps"])
    df["log_cons_w"] = np.log(df["cons_num_warps"])
    rows = []
    for cls, g in df.groupby("ordering_class"):
        # collapse to per-cell medians per (prod, cons, geometry)
        med = g.groupby(["prod_num_warps", "cons_num_warps", "n_workgroups",
                         "block_size", "buffer_bytes"])["log_us"].median().reset_index()
        med["log_prod_w"] = np.log(med["prod_num_warps"])
        med["log_cons_w"] = np.log(med["cons_num_warps"])
        # cell FE = (n_workgroups, block_size, buffer_bytes)
        med["cell_id"] = med.groupby(["n_workgroups", "block_size", "buffer_bytes"]).ngroup()
        # Demean log_us by cell to remove cell-level fixed effects
        med["log_us_demean"] = med["log_us"] - med.groupby("cell_id")["log_us"].transform("mean")
        med["log_prod_demean"] = med["log_prod_w"] - med.groupby("cell_id")["log_prod_w"].transform("mean")
        med["log_cons_demean"] = med["log_cons_w"] - med.groupby("cell_id")["log_cons_w"].transform("mean")
        X = med[["log_prod_demean", "log_cons_demean"]].values
        y = med["log_us_demean"].values
        # OLS
        try:
            XtX = X.T @ X
            beta = np.linalg.solve(XtX, X.T @ y)
            yhat = X @ beta
            ss_res = float(((y - yhat) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum()) if y.size else 1.0
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        except np.linalg.LinAlgError:
            beta = [float("nan"), float("nan")]
            r2 = float("nan")
        rows.append(dict(
            ordering_class=cls,
            n=len(med),
            beta_log_prod_w=float(beta[0]),
            beta_log_cons_w=float(beta[1]),
            r2_within=float(r2),
            dominance=("PRODUCER" if abs(beta[0]) > abs(beta[1]) else "CONSUMER"),
        ))
    return pd.DataFrame(rows)


def make_plots(per_strat, global_pc1_d, residual, df, out_dir):
    # 1. PC1 VE per stratum (heatmap-ish bar)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = per_strat["imbalance"].tolist()
    ax.bar(labels, per_strat["pc1_ve"] * 100, color="steelblue", label="PC1")
    ax.bar(labels, per_strat["pc2_ve"] * 100, bottom=per_strat["pc1_ve"] * 100,
           color="lightcoral", alpha=0.6, label="PC2")
    ax.axhline(90, color="red", ls=":", lw=1, label="90% threshold")
    if global_pc1_d:
        ax.axhline(global_pc1_d["pc1_ve"] * 100, color="grey", ls="--", lw=1,
                   label=f"global PC1={global_pc1_d['pc1_ve']*100:.1f}%")
    ax.set_xlabel("(prod_num_warps, cons_num_warps) imbalance")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("K-2388: PC1 VE per producer-consumer wave imbalance")
    ax.legend(loc="lower right")
    ax.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pc1_per_imbalance.png"), dpi=120)
    plt.close()

    # 2. Median latency per ordering class × imbalance
    g = df.groupby(["ordering_class", "prod_num_warps", "cons_num_warps"])["us_per_call"].median().reset_index()
    g["imb"] = g.apply(lambda r: f"({int(r['prod_num_warps'])},{int(r['cons_num_warps'])})", axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = g.pivot_table(index="imb", columns="ordering_class", values="us_per_call", aggfunc="median")
    pivot = pivot.reindex(["(1,8)", "(2,8)", "(4,8)", "(8,4)", "(8,2)", "(8,1)"])
    pivot.plot(kind="bar", ax=ax, logy=True)
    ax.set_xlabel("(prod_num_warps, cons_num_warps)")
    ax.set_ylabel("Median per-call latency (us, log)")
    ax.set_title("K-2388: median latency per ordering × imbalance")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "median_latency_per_imb.png"), dpi=120)
    plt.close()

    # 3. Residual analysis: producer vs consumer beta per ordering class
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(residual))
    w = 0.35
    ax.bar(x - w/2, residual["beta_log_prod_w"], w, color="C0", label="β log(prod_warps)")
    ax.bar(x + w/2, residual["beta_log_cons_w"], w, color="C1", label="β log(cons_warps)")
    ax.set_xticks(x)
    ax.set_xticklabels(residual["ordering_class"], rotation=15)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Within-cell OLS coefficient on log_us")
    ax.set_title("K-2388: producer vs consumer occupancy elasticity\n(within-cell FE residuals)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residual_pc_dominance.png"), dpi=120)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load(args.input_csv)
    print(f"[fit] loaded {len(df):,} cells (NaN dropped)")

    per_strat, loadings = per_stratum_pc1(df)
    per_strat.to_csv(os.path.join(args.output_dir, "pc1_per_stratum.csv"), index=False)
    print("[fit] per-stratum PC1:")
    print(per_strat.to_string(index=False))

    with open(os.path.join(args.output_dir, "pc1_per_stratum_loadings.json"), "w") as f:
        json.dump(loadings, f, indent=2)

    global_d = global_pc1(df)
    with open(os.path.join(args.output_dir, "global_pc1.json"), "w") as f:
        json.dump(global_d, f, indent=2)
    print(f"[fit] global PC1 VE = {global_d.get('pc1_ve', float('nan'))*100:.2f}%  "
          f"loadings={global_d.get('pc1_loadings')}")

    residual = residual_decomp(df)
    residual.to_csv(os.path.join(args.output_dir, "residual_decomposition.csv"), index=False)
    print("[fit] residual decomposition (within-cell elasticity):")
    print(residual.to_string(index=False))

    make_plots(per_strat, global_d, residual, df, args.output_dir)
    print(f"[fit] outputs → {args.output_dir}")


if __name__ == "__main__":
    main()
