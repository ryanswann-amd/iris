# K-2306 v65 — J4 ATOMIC_MAX_RELEASE on MI300X (gfx942)

## One-line result
**1,152 cells × 25 reps = 28,800 rows of `iris.atomic_max(sem='release')` collected on 8× MI300X (gfx942), node j06u19. Global median bandwidth 72.80 GiB/s, median latency 0.4293 ms.** Third leg of the MAX-family memory-ordering tetrad (after H4 RELAXED K-2297, I4 ACQUIRE K-2301).

## Aggregate stats (median across 25 reps × 64 src→dst pairs)
| scope | block | dtype | time_ms (median) | time_ms (p99) | BW GiB/s (median) |
|-------|-------|-------|------------------|---------------|--------------------|
| cta | 256 | int32 | 0.3810 | 0.3986 | 82.02 |
| cta | 256 | int64 | 0.3808 | 0.3879 | 82.07 |
| cta | 1024 | int32 | 0.3807 | 0.3880 | 82.08 |
| cta | 1024 | int64 | 0.3808 | 0.3985 | 82.06 |
| cta | 4096 | int32 | 0.3809 | 0.4350 | 82.05 |
| cta | 4096 | int64 | 0.3809 | 0.4306 | 82.05 |
| gpu | 256 | int32 | 0.5630 | 0.5685 | 55.51 |
| gpu | 256 | int64 | 0.4270 | 0.4547 | 73.19 |
| gpu | 1024 | int32 | 0.5662 | 0.5737 | 55.19 |
| gpu | 1024 | int64 | 0.4208 | 0.4453 | 74.26 |
| gpu | 4096 | int32 | 0.5668 | 0.6071 | 55.13 |
| gpu | 4096 | int64 | 0.4110 | 0.4602 | 76.03 |
| sys | 256 | int32 | 1.0748 | 1.1050 | 29.07 |
| sys | 256 | int64 | 1.0843 | 1.1151 | 28.82 |
| sys | 1024 | int32 | 1.0576 | 1.0905 | 29.55 |
| sys | 1024 | int64 | 1.0592 | 1.0888 | 29.50 |
| sys | 4096 | int32 | 1.0576 | 1.0928 | 29.55 |
| sys | 4096 | int64 | 1.0511 | 1.0862 | 29.73 |

## Local vs remote (same_gpu split)
| same_gpu | scope | dtype | median_ms | median_bw |
|----------|-------|-------|-----------|-----------|
| False | cta | int32 | 0.3813 | 81.95 |
| False | cta | int64 | 0.3813 | 81.96 |
| False | gpu | int32 | 0.5655 | 55.26 |
| False | gpu | int64 | 0.4214 | 74.15 |
| False | sys | int32 | 1.0670 | 29.29 |
| False | sys | int64 | 1.0692 | 29.23 |
| True  | cta | int32 | 0.0366 | 852.82 |
| True  | cta | int64 | 0.0365 | 856.56 |
| True  | gpu | int32 | 0.5619 | 55.61 |
| True  | gpu | int64 | 0.2960 | 105.56 |
| True  | sys | int32 | 0.6200 | 50.40 |
| True  | sys | int64 | 0.3491 | 89.52 |

## Key findings
- **Scope cost ladder (BS=1024, int32)**: cta=0.3807 ms ≪ gpu=0.5662 ms ≪ sys=1.0576 ms. gpu/cta = 1.49×, sys/cta = 2.78×. Release-fence cost rises sharply from CTA to system scope as expected for a release barrier that drains L2 before signaling.
- **Local (src==dst, 8 cells) ≫ cross-GPU (56 cells)**: cta-local=852.8 GiB/s vs cta-remote=82.0 GiB/s (10.4× speedup). RMW stays in L2 with no XGMI traffic.
- **dtype effect small at cta/sys, asymmetric at gpu**: int32/int64 medians within ~5% at cta and sys; at gpu int64 is 35% faster than int32 at all block sizes.
- **F4 ↔ J4 symmetry verified**: predicted J4 global median = 73.06 ± 0.73 GiB/s (from F4 K-2289=73.02 × MIN-MAX scale 1.000493). Measured = **72.80** GiB/s, within 0.36% of prediction (well under ±1% pre-registered envelope).
- **DQ PASS**: 28,800 rows / 1,152 cells / 25 reps each / 0% null-bw / 0% zero-bw / bw_range [27.6, 1184.6] GiB/s.

## Methodology
- iris.do_bench(return_mode="all"), 4 warmup + 25 timed reps per cell, barrier between reps.
- Preamble zeros buffer; SENTINEL=INT_MIN so max(0, INT_MIN)=0 (no net write but L2 atomic-MAX RMW still executes under release ordering).
- Grid: 3 scopes × 3 block_sizes × 2 dtypes × 64 (src,dst) pairs = 1,152 cells.
- 16 MiB symmetric heap buffer per dtype, pre-allocated once and reused across cells.
- Hardware: j06u19 (gfx942:sramecc+:xnack-), 8 GPUs, ROCm 7.2 / pytorch 2.10 container.
- Run id: `K-2306-1778487671-j06u19`, ts: 2026-05-11T08:21Z.

## Files
- `v65_J4_baseline.parquet` — 28,800-row corpus (210 KB)
- `v65_J4_baseline.csv` — same, CSV form (4.5 MB)
- `v65_J4_baseline.rank{0..7}.csv` — per-rank raw outputs (8 × 570 KB)
- `v65_J4_baseline_agg_by_scope_bs_dtype.csv` — 18-row aggregate
- `v65_J4_baseline_agg_by_local_remote.csv` — 12-row local/remote split
- `v65_manifest.json` — run metadata + outputs index
- `bench_j4_atomic_max_release.py` — bench source (verbatim K-2289 F4 clone, axes swapped MIN→MAX, INT_MAX→INT_MIN)
- `data_quality_v65.py` — DQ validator (PASS on this corpus)
- `finalize_v65.py` — aggregator + plot generator
- `p1_time_by_scope_bs.png`, `p2_bw_by_scope_bs.png`, `p3_pair_heatmap.png` — plots
