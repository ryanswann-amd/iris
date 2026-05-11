# K-2246 — Status (retry #7, v45.1 manifest rev v45.2 with Skeptic-corrected conclusion)

**Date**: 2026-05-11 (attempt #7)
**Workspace (logical)**: `/home/ryaswann/mc2-workspaces/K-2246/`
**Workspace (sandbox mount)**: `/workspace/`
**Output dir**: `/workspace/output/`
**Cluster**: c42 / mi300x partition / **b21u01** / ROCm 7.2 / Triton 3.6.0+rocm7.2.0
**Container**: `mc2-K-2246` based on `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0`

## Headline (Skeptic-corrected)

**P3 ATOMIC_CAS_ACQREL canonical median = 39.85 µs** at K=4 N_PROD=4 N_OPS=32 on MI300X b21u01.

**P3 is statistically indistinguishable from F3/N3/O3 within measurement noise.** Five-primitive ladder (M3,N3,O3,F3,P3) spans only ~3 µs, comparable to a single primitive's p25→p75 IQR. **Ladder is non-monotonic** (O3 release-CAS at 36.94 µs is *faster* than M3 relaxed at 37.61 µs). Ordering tax on CDNA3 CAS is at most a few µs and is NOT a clean monotonic function of fence strength.

The prior +23.27 µs / +24.11 µs deltas were **measurement artifacts** (host wall-clock + cross-host F3 anchor). When measured properly (CUDA event timer + same-host F3 anchor), the ordering tax is at the edge of statistical resolution at n=700.

## Reviewer feedback addressed (retry #7)

| reviewer | concern | fix in v45.2 |
|---|---|---|
| **Skeptic** | Headline "P3 > all four neighbors → monotonic ordering tax" is unsupported; O3 (36.94) is *lower* than F3 (38.57) and N3 (38.53); deltas within IQR | Conclusion softened; `monotonicity_check: FAILS` and `noise_band_check` recorded in manifest; IQR column added to ladder table; "non-monotonic" called out explicitly in summary, STATUS, and `softened_conclusion` field of the manifest |
| **Skeptic** | Wall-clock CSVs (`v45_baseline.csv`, `v45_paired.csv`) still in corpus alongside corrected event-timed values with no programmatic deprecation marker | (a) `v45_manifest.json` now exposes top-level `DEPRECATED_NOTICE`, `superseded_by:"v45.1"`, `corrected_p3_canonical_us:39.85`, and `superseded_artifacts.<file>.{superseded_by, reason, do_not_use_for}`; (b) every stale numeric field renamed with `DEPRECATED_` prefix; (c) wall-clock CSVs duplicated as `*.SUPERSEDED.csv` so the filename itself flags status; (d) `SUPERSEDED.md` index added to dir |
| **UX** | v45_manifest.json presented `p3_canonical_us:57.6251` as a top-level field with no inline correction (a user reading only that JSON would see the wrong number) | Top-level `superseded_by:"v45.1"`, `corrected_p3_canonical_us:39.85`, and `corrected_conclusion` fields added; manifest is now self-describing without requiring v45_1_manifest.json |

## Reviewer feedback addressed (prior retries — preserved)

| reviewer | concern | fix |
|---|---|---|
| Skeptic | Paired timing used `time.perf_counter()` around two streams + one sync — captured `max(focal,interferer)` not focal latency | `bench_p3_paired_events.py`: focal-stream `cudaEventRecord` start/end pair; interferer launches first on stream B, focal timed only via its own stream events. n=700 reps/cell. |
| Skeptic | F3 anchor in v45 ladder came from v44 on a different host → cross-version + cross-host comparison | `bench_v45_anchor.py` re-measures M3/N3/O3/F3/P3 on b21u01 in v45 environment, n=700 reps each |
| UX | +23.27 vs +24.11 inconsistency in headline | Both deltas now in same table with explicit Δ-vs-F3 / Δ-vs-M3 labels |
| UX | Truncated tables in summary.md / STATUS.md | Rewritten with short tables (≤17 rows) so they render in any viewer |

## Data files (live science = anchor + paired-event; wall-clock = SUPERSEDED)

| file                                  | rows  | purpose                                       | status |
|---|---:|---|---|
| `v45_anchor.csv`                      | 1,000 | anchor run-1 (5 prims × 200 reps), CUDA-event | LIVE |
| `v45_anchor_run2.csv`                 | 2,500 | anchor run-2 (5 prims × 500 reps), CUDA-event | LIVE |
| `v45_paired_canonical_event.csv`      | 3,600 | paired-event run-1 (18 cells × 200 reps)      | LIVE |
| `v45_paired_canonical_event_run2.csv` | 9,000 | paired-event run-2 (18 cells × 500 reps)      | LIVE |
| `v45_1_manifest.json`                 |   —   | combined manifest (rev v45.2: monotonicity, IQR, softened conclusion) | LIVE |
| `v45_baseline.csv`                    | 4,375 | original wall-clock baseline                  | SUPERSEDED (also as `v45_baseline.SUPERSEDED.csv`) |
| `v45_paired.csv`                      | 74,375| original wall-clock paired                    | SUPERSEDED (also as `v45_paired.SUPERSEDED.csv`) |
| `v45_manifest.json`                   |   —   | original v45 manifest                         | DEPRECATED in-place (top-level `superseded_by`, DEPRECATED_ prefixes) |

## In-corpus CAS ladder (b21u01, n=700/prim, event-timed) — sorted, with IQR

| primitive | load sem | CAS sem  | median µs | IQR µs | Δ vs M3 |
|---|---|---|---:|---:|---:|
| O3        | relaxed  | release  | 36.94     | 1.97   | **−0.66 (NON-MONOTONIC)** |
| M3        | relaxed  | relaxed  | 37.61     | 3.05   |  0.00   |
| N3        | acquire  | acquire  | 38.53     | 2.33   | +0.92   |
| F3        | acquire  | acq_rel  | 38.57     | 2.49   | +0.96   |
| **P3**    | **acq_rel** | **acq_rel** | **39.85** | **0.72** | **+2.25** |

**Ladder span 2.91 µs is comparable to per-primitive IQR (2-3 µs).** All five primitives statistically overlap.

## QC re-validation (v45.1 anchor + paired-event)

| split              | rows   | nulls | zeros | n/cell | nodes  | arch    | timer       | PASS |
|---|---:|---:|---:|---:|---|---|---|---|
| anchor             | 3,500  | 0     | 0     | 700    | b21u01 | gfx942  | cuda_event  | YES  |
| paired-event       | 12,600 | 0     | 0     | 700    | b21u01 | gfx942  | cuda_event  | YES  |

## Success criterion `done` (manual)

| sub-check | status |
|---|---|
| Workspace exists at `/workspace/` (linked to `/home/ryaswann/mc2-workspaces/K-2246/`) | DONE |
| All science artifacts present (anchor + paired CSVs + manifest + 4 PNGs + summary.md + SUPERSEDED.md) | DONE |
| All retry-7 reviewer feedback addressed (Skeptic.headline_softened, Skeptic.wall_clock_deprecation_markers, UX.manifest_self_describing) | DONE |
| All prior-retry reviewer feedback preserved (Skeptic.paired_timing, Skeptic.cross_host_anchor, UX.headline_consistency, UX.truncated_tables) | DONE |
| In-corpus single-host event-timed deltas reported with explicit labels and IQR context | DONE |
| Wall-clock CSVs flagged in JSON manifest (machine-readable) AND duplicated as `.SUPERSEDED.csv` (human-readable filename) | DONE |
| Push to ryanswann-amd corpus host (v45.2 retry-7 commit on K-2246-v45-p3-atomic-cas-acqrel) | see commit SHA below |

## Push verification (v45.2 retry-7)

| field           | value |
|---|---|
| repo            | `git@github.com:ryanswann-amd/iris.git` |
| branch          | `K-2246-v45-p3-atomic-cas-acqrel` |
| dir-in-repo     | `K-2246/v45/` (text-only updates: 2 manifests, summary.md, STATUS.md, SUPERSEDED.md, two `.SUPERSEDED.csv` copies) |
| browse          | https://github.com/ryanswann-amd/iris/tree/K-2246-v45-p3-atomic-cas-acqrel/K-2246/v45 |

(See git log in the corpus repo for the retry-7 SHA — appended below after push.)

## Files in `/workspace/output/`

(see summary.md for the full annotated table; all prior v45 + v45.1 files retained, plus retry-7 corrections)
