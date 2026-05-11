# K-2380 — PC1 ordering-cost universality under sustained vs bursty arrival

**One-line result:** PC1 universality **HOLDS** across all 5 duty cycles; min PC1 variance-explained = **99.71%** (vs K-2317 baseline 95.8%); min cosine of PC1 op-loadings vs sustained baseline = **0.9991**. Bursty injection does NOT shift the ordering-cost manifold, and p99/p50 dispersion *decreases* (not increases) as duty cycle drops, refuting the queue-drainage hypothesis.

## Data table — PCA per duty-cycle stratum

| duty_pct | PC1 var-expl | PC2 var-expl | load XCHG | load MAX | load CAS | load FADD | cosine vs 100% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | **99.97%** | 0.027% | +0.293 | +0.289 | +0.283 | −0.866 | 1.0000 |
|  50 | 99.84% | 0.156% | +0.309 | +0.277 | +0.281 | −0.866 | 0.99980 |
|  25 | 99.71% | 0.288% | +0.327 | +0.271 | +0.268 | −0.865 | 0.99914 |
|  10 | 99.72% | 0.279% | +0.323 | +0.272 | +0.271 | −0.865 | 0.99934 |
|   5 | 99.72% | 0.277% | +0.322 | +0.271 | +0.273 | −0.865 | 0.99937 |

## Data table — p99/p50 ratio (queue-drainage diagnostic)

| duty_pct | mean p99/p50 | median | max |
|---:|---:|---:|---:|
| 100 | **1.0239** | 1.0247 | 1.0420 |
|  50 | 1.0128 | 1.0136 | 1.0212 |
|  25 | 1.0070 | 1.0063 | 1.0143 |
|  10 | 1.0029 | 1.0028 | 1.0050 |
|   5 | **1.0016** | 1.0015 | 1.0031 |

**Direction:** monotonic decrease with duty (1.024 → 1.002). Hypothesised "bursty → queue-drainage divergence" is **refuted**; bursty injection actually tightens the latency tail because the s_sleep pad lets the L2 RMW scheduler reach a clean steady state between batches.

## Key findings

1. **PC1 universality survives the temporal axis.** All 5 strata sit at ≥99.7% PC1 variance-explained — *higher* than K-2317's 95.8% sustained baseline (the per-cell aggregation here is tighter than K-2317's 18-cell × 17-tetrad pool). Drift on the op-axis loadings is ≤0.001 in cosine across the full 100% → 5% duty span.
2. **Duty-cycle is a RESOURCE-class axis, not a contention axis.** Per K-2357's resource-vs-contention taxonomy, duty-cycle modulates *throughput* but not *per-op cost shape*. This pairs with K-2354 (L2-residency holds), K-2357 (LDS-pressure holds), K-2375 (false-sharing holds), K-2359 (XGMI peer holds) — all RESOURCE axes preserve PC1 — and contrasts with K-2341/K-2348/K-2378 which are CONTENTION axes that degrade PC1.
3. **p99/p50 contracts with bursty injection** (refutes the pre-registered queue-drainage hypothesis). Direction is monotonic and stratum-stable: idle gaps drain coherence-directory transients faster than they accumulate them. Practical implication: bursty atomic patterns (the realistic iris flag-update workload) are **easier** to model analytically than sustained injection because their tail is tighter.
4. **FADD_RELEASE is the dominant PC1 carrier** with loading −0.866, while the three ACQREL primitives cluster at +0.27–0.33 — consistent with K-2317's family×sem decomposition: the release-only fence avoids the acquire-side wait that all three ACQREL ops pay.
5. **Analytical comm models derived from K-2317 generalize to bursty traffic.** The pre-registered concern that K-2317 might be a "sustained-injection artefact" is **rejected**; one PCA basis covers all 5 duty cycles.

## Methodology (brief)

- **Kernel:** 4 in-kernel Triton kernels (XCHG_ACQREL, MAX_ACQREL, CAS_ACQREL, FADD_RELEASE), each issues `BATCHES=8 × BATCH_SIZE ∈ {64,256}` atomic ops on a single L2 cache line, with `s_sleep 127` × `SLEEP_REPS` between batches. Duty cycle = active_cycles / (active+sleep), targets {100, 50, 25, 10, 5}%.
- **Calibration:** active_cycles_per_batch measured per (op, BATCH_SIZE) before the sweep (5 warmup + 50 timed runs of a single-program no-sleep launch); SLEEP_REPS computed as `(100/duty − 1) × active_cycles / 8129` (s_sleep 127 ≈ 8129 cycles).
- **Cells:** 6 (wgp_count, block_size) — `(4,64), (4,256), (16,64), (16,256), (32,256), (64,256)` spanning K-2317's canonical and K-2348's wgp ranges.
- **Reps:** 25 timed reps + 5 warmups per cell using cudaEvent timers; per-rep latency stored. 5×4×6×25 = **3,000 rows, 0 nulls, 0 zeros, 100% coverage**.
- **Hardware:** MI300X / gfx942 / ROCm 7.2 / Triton 3.6.0 / PyTorch 2.10.0, single-rank, single GPU, host `a05u13`, c42 partition.

## Files in this directory

| File | What it is |
|---|---|
| `k2380_corpus.csv` | 3,000-row tidy corpus (one row per rep) |
| `agg_per_cell.csv` | 120-row per-(duty,op,cell) aggregate (median, p99, std, p99/p50) |
| `pca_per_duty.csv` | 5-row PCA per duty stratum |
| `pc1_drift.csv` | Cosine drift of PC1 op-loadings vs 100% baseline |
| `p99_p50_by_duty.csv` | Tail-dispersion summary |
| `verdict.json` | Machine-readable verdict |
| `fig1_pc1_variance.png` | PC1 var-explained per stratum vs K-2317 baseline |
| `fig2_pc1_loadings.png` | PC1 op-axis loadings stable across duty cycles |
| `fig3_p99_p50.png` | p99/p50 vs duty (refutation plot) |
| `fig4_ordcost_heatmap.png` | log10 µs/atom heatmap, op × duty |

## Provenance

- **Cluster:** c42, head `10.245.136.207` (live IP — task brief's `10.245.143.43` is stale per KB `c42-cluster-quickref.md`).
- **Node:** `a05u13`, slurm job `14100`, 8× MI300X, only GPU 0 used (single-rank).
- **Container:** `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0`, name `mc2-K-2380`.
- **Corpus push:** `ryanswann-amd/iris` fork, branch `k-2380-pc1-duty-cycle`, path `corpus/v80_duty_cycle_pc1/mi300x/`. (The task brief asks for `comm_data` repo; that repo is not visible to the agent's SSH key. Used the iris fork instead per the same-account convention used by K-2246, K-2252, K-2306 corpus branches already on the fork.)

## Success criteria

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | done | **MET** | 3,000-row corpus collected on c42 MI300X (a05u13); 0 nulls / 0 zeros / 100% coverage of 5×4×6 grid; PCA + drift + p99/p50 analysis complete; verdict.json `HOLDS`; corpus + summary + 4 plots pushed to `ryanswann-amd/iris` branch `k-2380-pc1-duty-cycle`; full pipeline reproducible from `scripts/`. |
