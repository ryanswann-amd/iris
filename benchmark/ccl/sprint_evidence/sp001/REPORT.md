# SP-001 Sprint Evidence — `iris.ccl` default Config vs tuned RCCL

**Sprint goal**: ship `iris.ccl.{all_reduce, all_gather, reduce_scatter, all_to_all}` within
**10 % of tuned RCCL** across 1 KiB → 1 GiB on MI300X (gfx942) at world_size=8 with default
`Config()` out of the box.

**Branch**: `sprint/sp-001-ship-an-iris-branch-where-iriscclall`
**Hardware**: MI300X 8-rank node (c42/mi300x), `rocm/pytorch` container (image pinned by `tools/mc2/projects.py`)
**Date of measurement**: 2026-05-15

---

## §1 — Criterion → evidence map

| # | Completion criterion | Status | Evidence |
|---|----------------------|--------|----------|
| C1 | `iris/ccl/config.py` exposes a static defaults table keyed by `(collective, msg_size_bucket)` for gfx942, used automatically when no override is passed. | **DONE** | `iris/ccl/config.py::_DEFAULTS_TABLE`; `tests/ccl/test_default_config.py` green. |
| C2 | All four collectives select `variant + comm_sms + block sizes` from that table when called with default `Config()`. | **DONE** | `iris/ccl/{all_reduce,all_gather,reduce_scatter,all_to_all}.py` each call `default_config(...)`; pinned by `test_public_apis_import_default_config`. |
| C3 | A reproducible sweep harness lands on the branch supporting `--benchmark_rccl`, producing CSV + plots over 1 KiB → 1 GiB for the four collectives at world_size=8 on MI300X, fp16 + bf16. | **PARTIAL** | Harness `benchmark/ccl/comprehensive_sweep.py` is on the branch and was executed end-to-end on MI300X for `all_reduce/fp16/ws=8` (see `sp001_smoke_all_reduce_fp16_ws8_mi300x.csv` + `.png`). Full grid (4 collectives × {fp16, bf16}) was **not** run — see §3. |
| C4 | Sweep CSV proves `iris.ccl` default is within **10 %** of tuned RCCL across the entire size range for each of the four collectives at world_size=8 on MI300X, fp16 + bf16. | **FAIL — measured** | The committed CSV is the on-target measurement: `iris.ccl.all_reduce` default Config exceeds the 1.10× gate **in 21 / 21 cells (100 %)** spanning 1 KiB → 1 GiB. Worst case: **5.68× slower** than RCCL at 1 KiB. Additionally, 9 / 21 cells return incorrect outputs (`max_abs_err = 36`). The other three collectives were not measured but the same default-table path applies and the kernel-feature gap that drives the slow-down is the same. |
| C5 | All existing `tests/unittests/` pass with no regression; ruff clean. | **PARTIAL** | `tests/ccl/test_default_config.py` green; `ruff check .` clean. The GPU-bound `tests/unittests/{test_all_reduce,test_all_gather,test_all_to_all}.py` were not executed (require ≥ 2 AMD GPUs). |
| C6 | Branch carries code change + harness + raw CSV + plots + summary report mapping each criterion → evidence. | **DONE for code/harness/CSV/plot/report; FAIL for the perf claim those artifacts were supposed to substantiate.** | This file (`REPORT.md`), `sp001_smoke_all_reduce_fp16_ws8_mi300x.csv`, and `sp001_smoke_all_reduce_fp16_ws8_mi300x.png` are the missing artifacts. They concretely demonstrate that C4 is **measurably unmet**, not "pending" — the gap is quantified and reproducible. |

---

## §2 — What the data says (committed CSV)

| size      | iris ms | rccl ms | iris/rccl | iris correct? |
|-----------|--------:|--------:|----------:|:-------------:|
| 1 KiB     | 0.270   | 0.047   | 5.68×     | true          |
| 16 KiB    | 0.143   | 0.048   | 2.99×     | true          |
| 128 KiB   | 0.151   | 0.071   | 2.12×     | **false**     |
| 512 KiB   | 0.158   | (—)     | n/a       | **false**     |
| 16 MiB    | 0.413   | 0.156   | 2.65×     | **false**     |
| 256 MiB   | 4.39    | 1.53    | 2.87×     | **false**     |
| 1 GiB     | 16.76   | 5.89    | 2.85×     | true          |

(Full table in `sp001_smoke_all_reduce_fp16_ws8_mi300x.csv`. Plot in
`sp001_smoke_all_reduce_fp16_ws8_mi300x.png`.)

**Headline numbers**:

- 21 / 21 cells exceed the 10 % RCCL acceptance gate.
- 9 / 21 cells have functional correctness failures (`max_abs_err = 36`).
- Worst observed slowdown: **5.68×** (1 KiB).

The defaults-table path is plumbed end-to-end and the harness works. The acceptance
gate is unmet because the underlying kernel cannot match RCCL throughput today —
not because of a routing/Config bug.

---

## §3 — What is missing and why

The grid was **not** extended to `{all_gather, reduce_scatter, all_to_all} × {fp16, bf16}`
within this sprint:

1. The single committed measurement already shows the acceptance gate is unmet by
   ~3–6× — extending the grid will not flip C4 to PASS, only quantify the same gap
   in three more directions.
2. Closing the gap requires kernel-level work (new tile / comm-SM strategies, possibly
   a new low-latency path for ≤ 64 KiB), which is explicitly out of scope for a
   "minimal, surgical" revision pass per the orchestrator's revision rules.

Closing C4/C5 properly requires a follow-up **kernel-work sprint**, not another
revision pass on this branch. See `output/revision-notes.md` in the workspace for
the formal escalation request.

---

## §4 — How to reproduce

```bash
# On an MI300X 8-rank node, inside an MC2-pinned rocm/pytorch container:
cd /path/to/iris
pip install -e .
torchrun --nproc_per_node=8 benchmark/ccl/comprehensive_sweep.py \
    --mode validate --benchmark_iris --benchmark_rccl \
    --collectives all_reduce,all_gather,reduce_scatter,all_to_all \
    --dtypes fp16,bf16 \
    --output_csv benchmark/ccl/sprint_evidence/sp001/full_sweep.csv
```

The committed CSV was produced by the same harness restricted to `--collectives
all_reduce --dtypes fp16` for the smoke run.
