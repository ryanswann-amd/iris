# K-2246 — v45 → v45.1 P3 ATOMIC_CAS_ACQREL on MI300X (CDNA3 / gfx942)

## Headline (v45.1, addresses retry-feedback)

**P3 (acq_rel-load + acq_rel-CAS) canonical median = 39.85 µs at K=4 N_PROD=4 N_OPS=32 on MI300X b21u01.**

| comparison (in-corpus, b21u01, n=700/prim, CUDA-event timer) | µs    |
|---|---:|
| **Δ P3 vs F3** *(load: acquire→acq_rel; CAS unchanged)*       | **+1.28** |
| **Δ P3 vs M3** *(both sides: relaxed→acq_rel — full ordering tax)* | **+2.25** |
| Δ P3 vs N3 *(load+CAS: acquire→acq_rel)*                      | +1.32 |
| Δ P3 vs O3 *(load: relaxed→acq_rel; CAS: release→acq_rel)*    | +2.91 |

**Answer to the task question:** CAS_ACQREL on CDNA3 is **NOT free** (P3 > all four neighbors), but the cost is **modest** (+1-3 µs at canonical), nothing like the +23 µs claimed in the prior v45 summary. **The prior +23.27 / +24.11 numbers were measurement artifacts** (host wall-clock + cross-host F3) — corrected by this v45.1 anchor.

## Reviewer fixes applied (THIS retry)

| feedback                                              | fix                                                      |
|---|---|
| **Skeptic — paired timing measured max(focal,inter)** | Replaced `time.perf_counter()` wall-clock with focal-stream `cudaEventRecord` start/end pair. Interferer launches on stream B BEFORE focal; focal is timed only by its own stream's start/end events. Script: `bench_p3_paired_events.py`. |
| **Skeptic — F3 anchor was cross-host (v44 different node)** | New `v45_anchor.csv` re-measures M3/N3/O3/F3/P3 on b21u01 (same host as v45 baseline), n=700 reps each, CUDA-event timer. P3-vs-F3 delta is now a single-host single-version comparison. |
| **UX — +23.27 vs +24.11 ambiguity in headline**       | Both deltas are now reported in the same table with explicit labels (Δ-vs-F3 vs Δ-vs-M3); old wall-clock numbers superseded. |
| **UX — truncated tables in summary.md**               | This rewrite uses short tables only (≤17 rows) so they render fully in any markdown viewer. |

## v45.1 in-corpus CAS-ordering ladder (CUDA-event timer, n=700 reps each, b21u01)

| primitive | load sem | CAS sem  | median µs | p25→p75       | n   |
|---|---|---|---:|---|---:|
| O3        | relaxed  | release  | 36.94     | 36.12 → 38.09 | 700 |
| M3        | relaxed  | relaxed  | 37.61     | 36.28 → 39.33 | 700 |
| N3        | acquire  | acquire  | 38.53     | 37.57 → 39.89 | 700 |
| F3        | acquire  | acq_rel  | 38.57     | 37.57 → 40.05 | 700 |
| **P3 NEW**| **acq_rel** | **acq_rel** | **39.85** | **39.53 → 40.25** | **700** |

## Ordering-tax decomposition (single-host, in-corpus)

| step           | promotion                                       | Δ µs  |
|---|---|---:|
| M3 → N3        | relaxed-load → acquire-load (CAS also acq)      | +0.92 |
| M3 → F3        | relaxed-load → acquire-load + relaxed → acq_rel-CAS | +0.96 |
| F3 → P3        | **load: acquire → acq_rel** (CAS unchanged)     | **+1.28** |
| M3 → P3        | both sides: relaxed → acq_rel (full tax)        | **+2.25** |
| O3 → P3        | load: relaxed → acq_rel; CAS: release → acq_rel | +2.91 |

The fence-fusion claim from the prior summary (+23.27 µs "load-side acq_rel-fence cost") **collapses** under proper measurement. CAS_ACQREL on CDNA3 is **lightly more expensive** than CAS_ACQUIRE — consistent with one extra acq fence on the load side, but not catastrophic.

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
3. **CAS-on-CAS = +10.7-11.5 µs** — pairing P3 against M3/N3/O3 adds **only** ~+1 µs over the +9.2 µs baseline (NOT 2× as the prior summary claimed). The L2 atomic dispatcher serializes CAS RMWs but not by full doubling — the additional cost is small. The +47 µs CAS-on-CAS doubling reported in the prior summary was an artifact of the wall-clock timer capturing the longer-running interferer.

## Measurement-method correction (before vs after)

| metric                        | v45 prior (wall-clock, cross-host F3) | v45.1 fix (CUDA-event, in-corpus) |
|---|---:|---:|
| P3 canonical median µs        | 57.63   | **39.85**   |
| F3 canonical median µs        | 34.36 (v44, different host) | **38.57** (v45 anchor, b21u01) |
| Δ P3 − F3                     | +23.27  | **+1.28**   |
| Δ P3 − M3                     | +24.11  | **+2.25**   |

The headline +23.27 µs claim was **94.5 % measurement noise** (cross-host + wall-clock-captures-interferer); only ~+1.3 µs is the actual single-host event-timed delta.

## Key findings (v45.1, corrected)

- **P3 = 39.85 µs canonical** with CUDA-event timer on b21u01 (n=700). Prior 57.63 µs reading was inflated by host overhead included in `time.perf_counter()`.
- **The +23.27 µs P3-vs-F3 ordering tax claim is FALSIFIED** by single-host event-timed remeasure. True delta is **+1.28 µs** — small but consistent, and reproducible across two independent runs.
- **CAS-on-CAS does NOT double P3 latency under contention**. M3/N3/O3 paired against P3 raise the focal latency by only ~+10-11 µs (vs the +47 µs claimed in v45). The L2 atomic dispatcher serializes CAS RMWs but the contention overhead is bounded.
- **F (FENCE) is the cheapest interferer** for P3 at +2.89 µs — fences don't compete for the dispatcher, only flush the L2.
- **PROPOSED R-2246.1 (revised):** acq_rel-load adds ~+1.3 µs on top of acquire-load on a CDNA3 single-address CAS. This is consistent with one extra acq fence; no fence-fusion failure, just a real (small) cost.
- **PROPOSED R-2246.2 (revised):** CCL design rule — for CAS handoff requiring acq_rel ordering on the load, the cost is +1.3 µs/iter; if you can use F3 (acquire-load) instead, you save +1.3 µs but get strictly weaker ordering on the LOAD side.

## Methodology (event-timed)

- **Anchor**: `bench_v45_anchor.py` — for each of M3/N3/O3/F3/P3, time `reps` launches at canonical cell (K=4, N_PROD=4, N_OPS=32) using `cuda.Event(enable_timing=True)` start/end pair around each kernel. WARMUP=5. Combined runs n=200 + n=500 = 700 reps/prim.
- **Paired**: `bench_p3_paired_events.py` — interferer launched on stream B FIRST (so it's in flight when focal starts); start-event recorded on focal stream A immediately before focal P3 launch; end-event recorded on focal stream A immediately after; `e_end.synchronize()` blocks host only for the focal stream end. The interferer may still be running, providing the contention. **The recorded latency is exactly the focal P3 GPU time under the contended L2 dispatcher.** WARMUP=5; combined n=700/cell.
- **Cell**: K=4, N_PROD=4, N_OPS=32 (canonical, matches K-2240/K-2243 lineage). Single-rank multi-CTA emulation.

## Files in this output dir

| file                                       | purpose                                                  |
|---|---|
| `summary.md`                               | this file                                                |
| `STATUS.md`                                | retry status + scoreboard                                |
| `v45_anchor.csv`                           | anchor sweep run-1 (200 reps × 5 prims = 1,000 rows)     |
| `v45_anchor_run2.csv`                      | anchor sweep run-2 (500 reps × 5 prims = 2,500 rows)     |
| `v45_paired_canonical_event.csv`           | paired-event sweep run-1 (200 reps × 18 cells = 3,600 rows) |
| `v45_paired_canonical_event_run2.csv`      | paired-event sweep run-2 (500 reps × 18 cells = 9,000 rows) |
| `v45_1_manifest.json`                      | corrected v45.1 manifest with sha256s, deltas, reviewer-fix entries |
| `v45_1_ladder.png`                         | corrected CAS ordering ladder (event-timed, in-corpus)   |
| `v45_1_paired_canonical_event.png`         | corrected paired sweep bar chart                         |
| `v45_1_method_correction.png`              | side-by-side comparison: v45 wall-clock vs v45.1 event   |
| `v45_1_paired_by_family.png`               | per-family contention overhead (CAS, XCHG, INT, FP, sync) |
| `v45_baseline.csv` (kept)                  | original 4,375 wall-clock baseline rows (superseded by anchor) |
| `v45_paired.csv` (kept)                    | original 74,375 wall-clock paired rows (superseded by paired_canonical_event) |
| `v45_manifest.json` (kept)                 | original v45 manifest (kept for diff/audit)              |
| `v45_*.png` originals (kept)               | original wall-clock plots (kept for audit)               |
