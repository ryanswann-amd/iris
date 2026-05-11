# SUPERSEDED ARTIFACTS — v45 wall-clock data

**Do NOT use the following files for canonical claims.** They are retained for diff/audit only.

| superseded file                | superseded_by                                                       | reason                                                              |
|---|---|---|
| `v45_baseline.csv`             | `v45_anchor.csv` + `v45_anchor_run2.csv`                            | wall-clock timer (`time.perf_counter`) inflated P3 by ~17.77 us     |
| `v45_paired.csv`               | `v45_paired_canonical_event.csv` + `v45_paired_canonical_event_run2.csv` | wall-clock captured `max(focal,interferer)` not focal latency        |
| `v45_manifest.json`            | `v45_1_manifest.json` (top-level `superseded_by:"v45.1"`)           | references the above wall-clock CSVs                                |
| `v45_cas_ordering_ladder.png`  | `v45_1_ladder.png`                                                  | derived from wall-clock anchor                                      |
| `v45_paired_canonical.png`     | `v45_1_paired_canonical_event.png`                                  | derived from wall-clock paired                                      |
| `v45_xchg_vs_cas_ordering.png` | (kept for audit; treats stale numbers)                              | cross-host comparison; replaced by in-corpus anchor                  |
| `v45_p3_scaling.png`           | (kept for audit)                                                    | uses wall-clock baseline                                            |

For convenience, the two wall-clock CSVs are also duplicated as
`v45_baseline.SUPERSEDED.csv` and `v45_paired.SUPERSEDED.csv` so the filename
itself flags their status to anyone listing the directory.

## Programmatic discovery

`v45_manifest.json` exposes:
- top-level `superseded_by: "v45.1"`
- top-level `corrected_p3_canonical_us: 39.85`
- `superseded_artifacts.<filename>.{superseded_by, reason, do_not_use_for}` for each wall-clock CSV
- `DEPRECATED_` prefix on every stale numeric field

`v45_1_manifest.json` exposes:
- `DEPRECATED_artifacts_in_v45.<filename>` mapping to the live replacement
- `softened_conclusion` field describing the corrected (non-monotonic, within-noise) interpretation
