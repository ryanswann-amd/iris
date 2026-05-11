# K-2388 — PC1 ordering-cost universality under producer/consumer wave-count IMBALANCE

**Result: PC1 universality HOLDS at every imbalance ratio. PC1 VE = 99.60–99.89% per stratum (global PC1 = 99.15%, well above the 90% gate). Producer-side wave occupancy dominates the residual for CAS_ACQREL (β_log_prod ≈ 0.48, β_log_cons ≈ 0.04 within-cell); other ordering classes show weak/balanced dominance. This validates that the K-2317 ordering-cost law is invariant to producer/consumer occupancy asymmetry — Origami can use it for COMM/GEMM CU resource allocation predictions.**

## Per-stratum PC1 (PCA across the 4 ordering classes per imbalance)

| (prod_nw, cons_nw) | n_cells | n_prims | PC1 VE | PC2 VE | PC1+PC2 | cosine→global PC1 |
|--------------------|--------:|--------:|-------:|-------:|--------:|------------------:|
| (1, 8)             | 8       | 3*      | 99.87% | 0.11%  | 99.98%  | 0.856             |
| (2, 8)             | 8       | 3*      | 99.89% | 0.10%  | 99.99%  | 0.856             |
| (4, 8)             | 8       | 4       | 99.60% | 0.39%  | 99.99%  | 0.9996            |
| (8, 4)             | 8       | 4       | 99.71% | 0.27%  | 99.98%  | 1.0000            |
| (8, 2)             | 8       | 4       | 99.87% | 0.10%  | 99.97%  | 0.9999            |
| (8, 1)             | 8       | 4       | 99.88% | 0.09%  | 99.97%  | 0.9999            |

\* CAS_ACQREL fails to compile at prod_num_warps ∈ {1,2} (Triton-AMD MLIR `llvm.extractvalue` type-mismatch bug, K-2349-known). Strata where prod≥4 carry the full 4-prim manifold; prod=1,2 strata fit a 3-prim manifold.

## Global pooled PC1 (32 cells × 4 prims, all imbalances)

- **PC1 VE = 99.15%, PC2 VE = 0.61%**
- Loadings: XCHG_ACQREL=0.449, MAX_ACQUIRE=0.537, CAS_ACQREL=0.517, FADD_RELEASE=0.492 — near-uniform → universal manifold

## Producer vs consumer occupancy elasticity (within-cell FE OLS, β on log_us)

| ordering_class | n  | β log(prod_w) | β log(cons_w) | within-R² | dominance |
|----------------|---:|--------------:|--------------:|----------:|-----------|
| **CAS_ACQREL** | 32 | **+0.480**    | +0.035        | 0.945     | **PRODUCER** (extreme) |
| FADD_RELEASE   | 48 | +0.004        | +0.049        | 0.790     | consumer (weak) |
| MAX_ACQUIRE    | 48 | +0.042        | +0.037        | 0.701     | producer (balanced) |
| XCHG_ACQREL    | 48 | +0.070        | +0.049        | 0.588     | producer (balanced) |

Sign of β: positive ⇒ more waves per CU → higher per-call latency (contention). CAS is uniquely producer-bottlenecked because its inner-tile RMW serializes per CU.

## Key findings

- **Universality survives.** Every (prod, cons) imbalance keeps PC1 VE ≥ 99.6%; the 90% gate is cleared by ~10pp margin.
- **Loading cosine ≥ 0.856** to global PC1 in all strata; ≥ 0.9996 once CAS is present (prod≥4). The low-prod strata still align well even without CAS.
- **CAS_ACQREL is producer-occupancy-bound**: a 13.6× elasticity ratio (0.48/0.035) — the cost of each CAS RMW scales primarily with how many producer waves contend per CU.
- **Other primitives (XCHG, MAX, FADD) are nearly occupancy-balanced**, so the residual is small and PC1 absorbs ~all variance.
- **PC1 ordering-cost law is invariant to producer/consumer occupancy asymmetry** (prerequisite for using it in the Origami COMM/GEMM CU-allocation model — confirmed).

## Methodology

- Hardware: c42 / MI300X (gfx942), single 8-GPU node `c09u13`, ROCm 7.2 / PyTorch 2.10 / Triton 3.6, iris SHA `9459a5e9`.
- Paired kernel: producer issues the cost-defining RMW (XCHG/MAX/CAS/FADD with sem in {acq_rel, acquire, release}); consumer does an `atomic_max` with matching sem to force the visibility chain. Cross-GPU: rank 0 → rank 1.
- Sweep: 4 ordering × 6 imbalance × 4 (n_workgroups, block_size) ∈ {(304,256),(304,1024),(1216,256),(1216,1024)} × 2 buffer ∈ {2 MiB L2-resident, 32 MiB HBM} × 25 reps + 4 warmup = **192 cells, 4400 valid reps** (16 CAS-low-prod cells dropped per known MLIR bug).
- Per-call latency timed via `iris.do_bench`; per-cell median used for PC1; within-cell fixed-effects OLS for occupancy elasticity. Per-cell rep CV median = 0.97% (p90 = 3.09%).

## Files

- `k2388_pc_imbalance.{csv,parquet}` — 192 per-cell aggregated rows
- `k2388_pc_imbalance_reps.{csv,parquet}` — 4400 per-rep rows
- `analysis/pc1_per_stratum.csv`, `pc1_per_stratum_loadings.json`, `global_pc1.json`
- `analysis/residual_decomposition.csv`, `loading_cosine.csv`
- `analysis/pc1_per_imbalance.png`, `pc1_loadings_drift.png`, `residual_pc_dominance.png`, `median_latency_per_imb.png`, `latency_per_prim_imb.png`
- `scripts/k2388_pc_imbalance.py`, `k2388_fit_pc1.py`, `run_in_container.sh`

## Push location

Corpus pushed to **`ryanswann-amd/iris`** branch **`k-2388-pc-imbalance`** at commit **`60e01a7adfc588e2755490faa7f3aa69918ed186`**, payload under `corpus/v82_pc_imbalance_pc1/mi300x/`. Verified with `git ls-remote`. (Note: `ryanswann-amd/comm_data` is not accessible to the deploy key available in this session; same fork-repo pattern used by K-2380 in `corpus/v80_duty_cycle_pc1/`.)

## Data quality

`scripts/quality.py` PASS — 192/192 cells (16 NaN = CAS at prod_nw∈{1,2}, MLIR-known); per-cell rep CV median 0.97% / p90 3.09%; us range [58.7, 1015.4]; full prim×imbalance×buffer coverage.
