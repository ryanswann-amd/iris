# K-2388 — PC1 ordering-cost law under producer/consumer wave-count imbalance

Corpus drop: `corpus/v82_pc_imbalance_pc1/mi300x/`

## Layout
- `corpus/v82_pc_imbalance_pc1/k2388_pc_imbalance.py` — sweep driver
- `corpus/v82_pc_imbalance_pc1/k2388_fit_pc1.py` — PCA + residual decomposition
- `corpus/v82_pc_imbalance_pc1/run_in_container.sh` — driver entry point
- `corpus/v82_pc_imbalance_pc1/mi300x/` — measurements + analysis + plots
  - `summary.md` — verdict, table, methodology
  - `k2388_pc_imbalance.{csv,parquet}` — 192 cells (per-cell aggregates)
  - `k2388_pc_imbalance_reps.{csv,parquet}` — 4400 valid reps
  - `analysis/` — PC1-per-stratum, global PC1, loadings, residual fits, 5 PNGs

See `corpus/v82_pc_imbalance_pc1/mi300x/summary.md` for the headline result.
