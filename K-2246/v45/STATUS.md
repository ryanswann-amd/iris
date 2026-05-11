# K-2246 — Status (retry #6, v45.1 with reviewer fixes)

**Date**: 2026-05-11 (attempt #6)
**Workspace (logical)**: `/home/ryaswann/mc2-workspaces/K-2246/`
**Workspace (sandbox mount)**: `/workspace/`
**Output dir**: `/workspace/output/`
**Cluster**: c42 / mi300x partition / **b21u01** / ROCm 7.2 / Triton 3.6.0+rocm7.2.0
**Container**: `mc2-K-2246` based on `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0`

## Headline (corrected from prior retry)

**P3 ATOMIC_CAS_ACQREL canonical median = 39.85 µs** at K=4 N_PROD=4 N_OPS=32 on MI300X b21u01.
**Δ vs F3 (in-corpus, single-host, event-timed) = +1.28 µs** — far smaller than the +23.27 µs claimed in v45.
**Δ vs M3 (full ordering tax) = +2.25 µs**.

The prior +23.27 µs / +24.11 µs deltas were **measurement artifacts** (host wall-clock + cross-host F3 anchor). When measured properly (CUDA event timer + same-host F3 anchor), the ordering tax is small and consistent.

## Reviewer feedback addressed

| reviewer  | concern                                                               | fix in v45.1                                              |
|---|---|---|
| Skeptic   | Paired timing used `time.perf_counter()` around two streams + one sync — captured `max(focal,interferer)` not focal latency | New `bench_p3_paired_events.py`: focal-stream `cudaEventRecord` start/end pair; interferer launches first on stream B, focal timed only via its own stream events. Verified in `v45_paired_canonical_event{,_run2}.csv` (700 reps/cell). |
| Skeptic   | F3 anchor in v45 ladder came from v44 on a different host → cross-version + cross-host comparison | New `bench_v45_anchor.py` re-measures M3/N3/O3/F3/P3 on b21u01 in v45 environment, n=700 reps each. P3-vs-F3 delta now single-host single-version. |
| UX        | +23.27 vs +24.11 inconsistency in headline                            | Both deltas now in the same table with explicit Δ-vs-F3 / Δ-vs-M3 labels. Old wall-clock numbers superseded. |
| UX        | Truncated tables in summary.md / STATUS.md                            | Rewritten with short tables (≤17 rows) so they render in any viewer. Verified by re-reading the generated file.|

## Data files added (v45.1, b21u01, gfx942, CUDA-event timer)

| file                                  | rows  | purpose                                       | sha256 head        |
|---|---:|---|---|
| `v45_anchor.csv`                      | 1,000 | anchor run-1 (5 prims × 200 reps)             | `35caea3c8626b806…` |
| `v45_anchor_run2.csv`                 | 2,500 | anchor run-2 (5 prims × 500 reps)             | `b260d79e8186b17b…` |
| `v45_paired_canonical_event.csv`      | 3,600 | paired-event run-1 (18 cells × 200 reps)      | `a0f0e39f2aae9e95…` |
| `v45_paired_canonical_event_run2.csv` | 9,000 | paired-event run-2 (18 cells × 500 reps)      | `699e6a0c1f07f0e2…` |
| `v45_1_manifest.json`                 |   —   | combined manifest with deltas, sha256s, fixes | (regenerated)      |

(Combined: **n=700 reps per cell** for both anchor and paired sweeps.)

Original v45 wall-clock files retained (`v45_baseline.csv`, `v45_paired.csv`, `v45_manifest.json`, original PNGs) for diff/audit, but the v45.1 anchor + event files are the corrected science.

## In-corpus CAS ladder (b21u01, n=700/prim, event-timed)

| primitive | load sem | CAS sem  | median µs | Δ vs M3 |
|---|---|---|---:|---:|
| O3        | relaxed  | release  | 36.94     | −0.66   |
| M3        | relaxed  | relaxed  | 37.61     |  0.00   |
| N3        | acquire  | acquire  | 38.53     | +0.92   |
| F3        | acquire  | acq_rel  | 38.57     | +0.96   |
| **P3**    | **acq_rel** | **acq_rel** | **39.85** | **+2.25** |

Decomposition (no truncation):

| step           | promotion                                | Δ µs   | label                  |
|---|---|---:|---|
| M3 → N3        | relaxed-load → acquire-load              | +0.92  | acq-load tax           |
| M3 → F3        | + relaxed → acq_rel-CAS                  | +0.96  | acq-load + acq_rel-CAS |
| **F3 → P3**    | **acquire-load → acq_rel-load**          | **+1.28**  | **load-side acq_rel tax (Δ-vs-F3)** |
| **M3 → P3**    | both sides relaxed → acq_rel             | **+2.25**  | **full ordering tax (Δ-vs-M3)** |
| O3 → P3        | release-CAS → acq_rel-CAS + relaxed → acq_rel-load | +2.91  | release→acq_rel both-side |

## QC re-validation (v45.1 anchor + paired-event)

| split              | rows   | nulls | zeros | n/cell | nodes  | arch    | timer       | PASS |
|---|---:|---:|---:|---:|---|---|---|---|
| anchor             | 3,500  | 0     | 0     | 700    | b21u01 | gfx942  | cuda_event  | YES  |
| paired-event       | 12,600 | 0     | 0     | 700    | b21u01 | gfx942  | cuda_event  | YES  |

## Success criterion `done` (manual)

| sub-check                                                          | status |
|---|---|
| Workspace exists at `/workspace/` (linked to `/home/ryaswann/mc2-workspaces/K-2246/`) | DONE |
| All science artifacts present (anchor + paired CSVs + manifest + 4 PNGs + summary.md) | DONE |
| Reviewer feedback addressed (Skeptic.paired_timing, Skeptic.cross_host_anchor, UX.headline_consistency, UX.truncated_tables) | DONE |
| In-corpus single-host event-timed deltas reported with explicit labels | DONE |
| Push to ryanswann-amd corpus host (v45.1 commit on K-2246-v45-p3-atomic-cas-acqrel) | (see below) |

## Files in `/workspace/output/`

(see summary.md for the full table; all 14 prior v45 files retained + 9 new v45.1 files)
