# K-2246 — v45 → v45.1 (rev v45.2) P3 ATOMIC_CAS_ACQREL on MI300X (CDNA3 / gfx942)

## Headline (v45.2 — Skeptic-corrected)

**P3 (acq_rel-load + acq_rel-CAS) canonical median = 39.85 µs at K=4 N_PROD=4 N_OPS=32 on MI300X b21u01.**

> **Conclusion (softened per Skeptic):** P3 is **statistically indistinguishable from F3/N3/O3 within measurement noise**. The ordering tax on CDNA3 CAS is at most a few µs and is **NOT monotonic** in fence strength (O3 release-CAS at 36.94 µs is *faster* than M3 relaxed at 37.61 µs). The previously claimed +23 µs P3-vs-F3 ordering tax was a measurement artifact (wall-clock timer + cross-host F3 anchor); the corrected single-host event-timed delta is **+1.28 µs**, which is within the per-primitive p25→p75 IQR (~2-3 µs).

| comparison (in-corpus, b21u01, n=700/prim, CUDA-event timer) | µs    | within IQR? |
|---|---:|---|
| Δ P3 vs F3 *(load: acquire→acq_rel; CAS unchanged)*           | +1.28 | YES (~2.5 µs IQR) |
| Δ P3 vs M3 *(both sides: relaxed→acq_rel — full ordering tax)*| +2.25 | YES (~3.0 µs IQR) |
| Δ P3 vs N3 *(load+CAS: acquire→acq_rel)*                      | +1.32 | YES (~2.3 µs IQR) |
| Δ P3 vs O3 *(load: relaxed→acq_rel; CAS: release→acq_rel)*    | +2.91 | borderline (~2.0 µs IQR) |
| O3 vs M3 *(release-CAS faster than relaxed — non-monotonic!)* | −0.66 | YES |

## Reviewer fixes applied

| reviewer | issue | fix |
|---|---|---|
| **Skeptic (retry-7)** | "P3 > all four neighbors" claim was unsupported — O3 is *lower* than F3 and N3, ladder is non-monotonic, deltas are within IQR | Conclusion softened to "P3 ∼ F3/N3/O3 within noise; ordering tax bounded at a few µs and not monotonic". `monotonicity_check: FAILS` and `noise_band_check` recorded in `v45_1_manifest.json`. IQR column added to ladder table. |
| **Skeptic (retry-7)** | Wall-clock CSVs still in corpus with no programmatic deprecation marker | `v45_baseline.csv` + `v45_paired.csv` flagged in `v45_manifest.json` via top-level `superseded_by:"v45.1"`, `corrected_p3_canonical_us:39.85`, and `superseded_artifacts.<file>.{superseded_by, reason, do_not_use_for}`; also duplicated as `*.SUPERSEDED.csv` and indexed in `SUPERSEDED.md`. |
| **UX (retry-7)** | v45_manifest.json showed superseded `p3_canonical_us:57.6251` as a top-level field with no inline correction | Top-level `DEPRECATED_NOTICE`, `superseded_by:"v45.1"`, `corrected_p3_canonical_us:39.85` added; every stale numeric field renamed with `DEPRECATED_` prefix. The manifest is now self-describing. |
| Skeptic (prior retry) | Paired timing measured `max(focal,interferer)` | `bench_p3_paired_events.py` — focal-stream cudaEventRecord pair; interferer launches first on stream B but timed only via focal stream events. n=700 reps/cell. |
| Skeptic (prior retry) | F3 anchor was cross-host (v44 different node) | `bench_v45_anchor.py` re-measures M3/N3/O3/F3/P3 on b21u01 in v45 environment, n=700 reps each. |
| UX (prior retry) | +23.27 vs +24.11 ambiguity in headline | Both deltas now in same table with explicit Δ-vs-F3 / Δ-vs-M3 labels. |

## v45.1 in-corpus CAS-ordering ladder (CUDA-event timer, n=700 reps each, b21u01)

Sorted by median (ascending) — note: **NOT monotonic in fence strength**.

| primitive | load sem | CAS sem  | median µs | p25 → p75       | IQR  | n   |
|---|---|---|---:|---|---:|---:|
| O3        | relaxed  | release  | 36.94     | 36.12 → 38.09   | 1.97 | 700 |
| M3        | relaxed  | relaxed  | 37.61     | 36.28 → 39.33   | 3.05 | 700 |
| N3        | acquire  | acquire  | 38.53     | 37.57 → 39.89   | 2.33 | 700 |
| F3        | acquire  | acq_rel  | 38.57     | 37.57 → 40.05   | 2.49 | 700 |
| **P3 NEW**| **acq_rel** | **acq_rel** | **39.85** | **39.53 → 40.25** | **0.72** | **700** |

**Whole-ladder spread is only ~3 µs**, comparable to a single primitive's IQR. P3 has the narrowest IQR (0.72 µs) but its median sits within the M3/N3/F3 IQR envelope. **The ladder is NOT monotonic** because O3 (release-CAS, in principle stronger ordering on the store side) measures *faster* than M3 (fully relaxed). This rules out the simple "more fences = more cost" model on CDNA3 single-address CAS.

## Ordering-tax decomposition (single-host, in-corpus) — interpret with caution

| step           | promotion                                       | Δ µs  | comment |
|---|---|---:|---|
| M3 → N3        | relaxed-load → acquire-load (CAS also acq)      | +0.92 | within noise |
| M3 → F3        | relaxed-load → acquire-load + relaxed → acq_rel-CAS | +0.96 | within noise |
| F3 → P3        | load: acquire → acq_rel (CAS unchanged)         | +1.28 | within noise |
| M3 → P3        | both sides: relaxed → acq_rel (full tax)        | +2.25 | within noise |
| O3 → P3        | load: relaxed → acq_rel; CAS: release → acq_rel | +2.91 | borderline |
| **M3 → O3**    | **CAS: relaxed → release (no other change)**    | **−0.66** | **NON-MONOTONIC — release is faster than relaxed** |

The fence-fusion claim from the prior summary (+23.27 µs "load-side acq_rel-fence cost") **collapses** under proper measurement. CAS_ACQREL on CDNA3 is **at most a few microseconds** more expensive than CAS_ACQUIRE/CAS_RELAXED — and the difference is at the edge of statistical resolution at n=700.

## v45.1 paired-event canonical sweep (focal P3 vs 17 interferers, n=700)

Focal-only baseline: **39.65 µs** (matches the in-corpus anchor 39.85 µs, ±0.5%).

| interferer | family         | focal P3 (µs) | Δ vs focal-only | regime        |
|---|---|---:|---:|---|
| F  (FENCE)        | sync         | 42.54   | +2.89  | LIGHTEST    |
| E3 (xchg-acqrel)  | XCHG         | 48.79   | +9.14  | LIGHT       |
| P  (PUT)          | comm         | 48.83   | +9.18  | LIGHT       |
| I3 (fp-fmin)      | FP-atomic    | 48.91   | +9.26  | LIGHT       |
| H3 (fp-fmax)      | FP-atomic    | 48.95   | +9.30  | LIGHT       |
| D3 (atomic-dec)   | INT-atomic   | 49.07   | +9.42  | LIGHT       |
| L3 (xchg-relaxed) | XCHG         | 49.07   | +9.42  | LIGHT       |
| G  (atomic-or)    | INT-atomic   | 49.07   | +9.42  | LIGHT       |
| J3 (xchg-release) | XCHG         | 49.11   | +9.46  | LIGHT       |
| K3 (xchg-acquire) | XCHG         | 49.11   | +9.46  | LIGHT       |
| Y  (atomic-add)   | INT-atomic   | 49.31   | +9.66  | LIGHT       |
| M3 (cas-relaxed)  | CAS          | 50.35   | +10.71 | LIGHT-MED   |
| O3 (cas-release)  | CAS          | 50.76   | +11.11 | LIGHT-MED   |
| H  (barrier-atomic)| sync        | 50.98   | +11.33 | LIGHT-MED   |
| N3 (cas-acquire)  | CAS          | 51.12   | +11.47 | LIGHT-MED   |
| G3 (fp-fadd)      | FP-atomic    | 53.56   | +13.91 | MED         |
| R2 (barrier-all)  | sync         | 54.44   | +14.79 | MED-HEAVY   |

**Three regimes (corrected):**
1. **F (fence) is uniquely cheap** at +2.89 µs — `tl.atomic_add(addr, 0, sem='acq_rel')` on stream B does not contend for the L2 atomic dispatcher; it just issues an L2 invalidate+flush.
2. **Most interferers cluster at +9.2-9.7 µs** — concurrent atomic traffic to a different address class adds a single dispatcher-arbitration overhead.
3. **CAS-on-CAS = +10.7-11.5 µs** — pairing P3 against M3/N3/O3 adds ~+1 µs over the +9.2 µs baseline (NOT 2× as the prior summary claimed). The +47 µs CAS-on-CAS doubling reported in the prior summary was an artifact of the wall-clock timer capturing the longer-running interferer.

## Measurement-method correction (before vs after)

| metric                        | v45 prior (wall-clock, cross-host F3) | v45.1 fix (CUDA-event, in-corpus) |
|---|---:|---:|
| P3 canonical median µs        | 57.63   | **39.85**   |
| F3 canonical median µs        | 34.36 (v44, different host) | **38.57** (v45 anchor, b21u01) |
| Δ P3 − F3                     | +23.27  | **+1.28**   |
| Δ P3 − M3                     | +24.11  | **+2.25**   |

The headline +23.27 µs claim was **94.5 % measurement noise** (cross-host + wall-clock-captures-interferer); the corrected single-host event-timed delta is +1.28 µs — small, and within the per-primitive p25→p75 IQR.

## Key findings (v45.2, softened)

- **P3 = 39.85 µs canonical** (CUDA-event timer, b21u01, n=700). Prior 57.63 µs reading was inflated by host overhead in `time.perf_counter()`.
- **The +23.27 µs P3-vs-F3 ordering-tax claim is FALSIFIED**. True single-host event-timed delta is +1.28 µs — within the per-primitive IQR.
- **The CAS ordering ladder on CDNA3 is NOT monotonic in fence strength**: O3 (release-CAS) is faster than M3 (relaxed). All five primitives in M3/N3/O3/F3/P3 lie within a ~3-µs window, comparable to a single primitive's p25→p75 IQR.
- **CAS-on-CAS does NOT double P3 latency**. M3/N3/O3 paired against P3 raise the focal latency by only ~+10-11 µs, of which the *additional* CAS-vs-non-CAS cost is ~+1 µs.
- **F (FENCE) is the cheapest interferer** for P3 at +2.89 µs.
- **PROPOSED R-2246.1 (revised, weakened):** acq_rel-load may add up to ~+1.3 µs on top of acquire-load on a CDNA3 single-address CAS; effect is at the edge of statistical resolution at n=700, and the wider M3/N3/O3/F3 ladder is non-monotonic, so this should be treated as an upper bound, not a clean ordering-tax law.
- **PROPOSED R-2246.2 (revised, weakened):** CCL design rule — for CAS handoff requiring acq_rel ordering, the cost vs F3 is ≤1.3 µs/iter (within noise). Choice between F3 and P3 should be driven by correctness/ordering needs, not perf, on this microbenchmark cell.

## Methodology (event-timed)

- **Anchor**: `bench_v45_anchor.py` — for each of M3/N3/O3/F3/P3, time `reps` launches at canonical cell (K=4, N_PROD=4, N_OPS=32) using `cuda.Event(enable_timing=True)` start/end pair around each kernel. WARMUP=5. Combined runs n=200 + n=500 = 700 reps/prim.
- **Paired**: `bench_p3_paired_events.py` — interferer launched on stream B FIRST (so it's in flight when focal starts); start-event recorded on focal stream A immediately before focal P3 launch; end-event recorded on focal stream A immediately after; `e_end.synchronize()` blocks host only for the focal stream end. The interferer may still be running, providing the contention. **The recorded latency is exactly the focal P3 GPU time under the contended L2 dispatcher.** WARMUP=5; combined n=700/cell.
- **Cell**: K=4, N_PROD=4, N_OPS=32 (canonical, matches K-2240/K-2243 lineage). Single-rank multi-CTA emulation.

## Files in this output dir

| file                                       | purpose                                                  | status |
|---|---|---|
| `summary.md`                               | this file                                                | LIVE   |
| `STATUS.md`                                | retry status + scoreboard                                | LIVE   |
| `SUPERSEDED.md`                            | index of stale wall-clock artifacts                      | LIVE   |
| `v45_anchor.csv`                           | anchor sweep run-1 (200 reps × 5 prims)                  | LIVE   |
| `v45_anchor_run2.csv`                      | anchor sweep run-2 (500 reps × 5 prims)                  | LIVE   |
| `v45_paired_canonical_event.csv`           | paired-event sweep run-1 (200 reps × 18 cells)           | LIVE   |
| `v45_paired_canonical_event_run2.csv`      | paired-event sweep run-2 (500 reps × 18 cells)           | LIVE   |
| `v45_1_manifest.json`                      | corrected v45.1 manifest (rev v45.2)                     | LIVE   |
| `v45_1_*.png`                              | corrected plots                                          | LIVE   |
| `v45_manifest.json`                        | original v45 manifest, now with `DEPRECATED_` prefixes + `superseded_by` markers | DEPRECATED in-place |
| `v45_baseline.csv` + `.SUPERSEDED.csv`     | original wall-clock baseline                              | SUPERSEDED |
| `v45_paired.csv` + `.SUPERSEDED.csv`       | original wall-clock paired                                | SUPERSEDED |
| `v45_*.png` originals                      | wall-clock plots                                          | SUPERSEDED |
