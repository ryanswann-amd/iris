# K-2246 — v44→v45 P3 ATOMIC_CAS_ACQREL on MI300X (CDNA3 / gfx942)

## One-line result

**P3 (acq_rel-load + acq_rel-CAS) canonical median = 57.63 µs at K=4 N_PROD=4 N_OPS=32 on MI300X gfx942 → P3 − F3 = +23.27 µs isolates the cost of upgrading the load-side from `acquire` to `acq_rel`. Fence-fusion (K-2243 R-2243.2) BREAKS when both sides are acq_rel: P3 stacks the load-side and store-side L2 invalidate/flush sequences instead of collapsing them.**

Cluster: c42 / mi300x partition / b21u01 / ROCm 7.2 / Triton 3.6.0+rocm7.2.0. Container `mc2-K-2246` based on `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0`. Cluster IP-drift fix from K-2248 applied (10.245.136.207, not 143.43).

## Corpus shape — v45 / CDNA3

| split    | rows   | shape                                | quality                              |
|---|---:|---|---|
| baseline | 4,375  | 1 prim (P3) × 175 cells × 25 reps    | 0 nulls / 0 zeros / 0 errors / 100% reps_per_cell=25 |
| paired   | 74,375 | 17 interferers × 175 cells × 25 reps | 0 nulls / 0 zeros / single-host gfx942 |
| **total** | **78,750** | both PASS quality gates (1 single-host WARN, expected single-node replay) |

Cell grid (matches K-2248 published K-2240 lineage convention): K∈{1,2,4,8,16}, N_PROD∈{2..8}, N_OPS∈{8,16,32,64,128} = 175 cells. REPS=25, WARMUP=3.

## CAS memory-ordering ladder at canonical cell K=4, N_PROD=4, N_OPS=32

| primitive | load sem | CAS sem  | µs       | Δ vs M3   | Source                  |
|---|---|---|---:|---:|---|
| M3        | relaxed  | relaxed  | 33.52    | 0.00      | K-2243 v44 within       |
| N3        | acquire  | acquire  | 33.36    | −0.16     | K-2243 v44 within       |
| O3        | relaxed  | release  | 43.68    | +10.16    | K-2243 v44 within       |
| F3        | acquire  | acq_rel  | 34.36    | +0.84     | K-2243 v44 within       |
| **P3**    | **acq_rel** | **acq_rel** | **57.63** | **+24.11** | **K-2246 v45 NEW**      |

**Step-by-step ladder:**

| step           | promotion                                | Δ µs   |
|---|---|---:|
| M3 → N3        | relaxed-load → acquire-load              | −0.16  |
| M3 → F3        | both → acq_rel-CAS (acq-load)            | +0.84  |
| F3 → **P3**    | **acq-load → acq_rel-load (CAS already acq_rel)** | **+23.27** |
| M3 → **P3**    | **full ordering tax (relaxed → acq_rel both sides)** | **+24.11** |
| O3 → P3        | relaxed-load → acq_rel-load (release → acq_rel-CAS) | +13.95 |

**The fence-fusion observed in K-2243 (acquire-load + acq_rel-CAS share a single L2 invalidate/flush) BREAKS when the load is acq_rel.** The acq_rel-load issues its OWN L2 invalidate/flush BEFORE the CAS, and the acq_rel-CAS then issues its own pair AFTER. The two sequences cannot fuse because they are ordered (load completes before CAS dispatches). Net effect: P3 pays ≈2× the per-iteration fence cost of F3.

## P3 baseline scaling at K=4, N_PROD=4

| N_OPS | median µs | µs/op |
|---:|---:|---:|
| 8   | 29.38  | (baseline anchor) |
| 16  | 38.79  | +1.18 |
| 32  | 57.63  | +1.18 |
| 64  | 96.25  | +1.21 |
| 128 | 171.13 | +1.17 |

**Linear fit: latency ≈ 1.18 µs/op + 19.6 µs launch baseline (R² ≈ 1.00).** Matches the expected serialization model for a fully-fenced atomic. Comparison to K-2248 Q3 (seq_cst CAS): Q3 is 1.5 µs/op + 12 µs baseline, so per-op P3 < Q3 by ~21% but P3 baseline launch is +7.6 µs higher (the load-side acq_rel fence on the WARMUP+first-iter path).

## P3 vs K (contention multiplier) at N_PROD=4, N_OPS=32

| K | median µs |
|---:|---:|
| 1  | 57.68 |
| 2  | 57.33 |
| 4  | 57.63 |
| 8  | 69.99 |
| 16 | 98.35 |

**P3 saturates from K=1 to K=4** (single SE busy with sys-scope fences); CDNA3 contention regime kicks in at K≥8.

## P3 paired-interference at canonical (sorted by impact)

| interferer | family    | paired µs | Δ vs P3-baseline | regime |
|---|---|---:|---:|---|
| P  (PUT)   | comm      | 72.83     | +15.21          | LIGHT — P3 fences dominate |
| L3 (xchg-relaxed) | XCHG | 73.48 | +15.86 | LIGHT |
| Y  (atomic-add)   | INT-atomic | 73.68 | +16.06 | LIGHT |
| D3 (atomic-dec)   | INT-atomic | 74.06 | +16.44 | LIGHT |
| K3 (xchg-acquire) | XCHG    | 79.96 | +22.34 | LIGHT-MED |
| J3 (xchg-release) | XCHG    | 81.01 | +23.39 | LIGHT-MED |
| E3 (xchg-acqrel)  | XCHG    | 82.80 | +25.18 | LIGHT-MED |
| I3 (fp-fmin)      | FP-atomic | 87.98 | +30.36 | MED |
| H3 (fp-fmax)      | FP-atomic | 90.64 | +33.01 | MED |
| G  (atomic-or)    | INT-atomic | 96.20 | +38.57 | MED |
| F  (fence)        | sync    | 97.45 | +39.83 | MED |
| H  (barrier-atomic)| sync   | 97.79 | +40.16 | MED |
| **N3 (cas-acquire)** | **CAS** | **99.15** | **+41.53** | **CAS-on-CAS** |
| **M3 (cas-relaxed)** | **CAS** | **105.06** | **+47.44** | **CAS-on-CAS** |
| **O3 (cas-release)** | **CAS** | **105.85** | **+48.22** | **CAS-on-CAS** |
| G3 (fp-fadd)      | FP-atomic | 135.84 | +78.22 | HEAVY |
| R2 (barrier-all)  | sync    | 148.45 | +90.82 | HEAVY |

**Key observations:**

1. **Floor of +15 µs** — light interferers can't push P3 below ~+15 µs because P3's own sys-scope fences serialize the address. This is consistent with K-2248 Q3 saturation (≤+2 µs floor) but with much more headroom because P3 is per-address acq_rel, not seq_cst.
2. **CAS-on-CAS ≈ +47 µs** — pairing P3 against any of M3/N3/O3 roughly DOUBLES P3 latency. The L2 atomic dispatcher on gfx942 serializes CAS RMWs to the same address class even across streams. Confirms K-2243's "CAS-on-CAS doubles release tax" finding (carried via R-2220.1) extends to acq_rel both-sides.
3. **R2 (BARRIER_ALL) and G3 (FP-atomic FADD) are heaviest.** R2's emulated atomic_xor + acq_rel atomic_add hits the same L2 dispatcher; G3's FP-atomic engine on gfx942 contends for the same L2 buffer.

## Question answered

> Does CAS_ACQREL on CDNA3 show the same ordering-cost stacking observed in XCHG_ACQREL vs XCHG_RELAXED, or is CAS already serialized enough that ACQREL is free?

**Neither.** CAS_ACQREL with `acq_rel` on the LOAD side (P3) shows **strong ordering-cost stacking** (+24 µs over relaxed, +23 µs over F3) — even bigger than the XCHG ladder (E3-L3 ≈ +14 µs from K-2243 priors). The K-2243 fence-fusion that made F3 nearly free **does not generalize**: it requires the load to be `acquire` (not `acq_rel`). P3 is the first measured CAS configuration where the per-iteration cost is dominated by load-side fencing rather than store-side. **Engine-class fence-additivity is a function of (load_sem, store_sem) jointly, not of either side alone.**

## Key findings

- **P3 = 57.63 µs canonical.** First measured CAS configuration with `acq_rel` on the load side; mirrors the K-2243 O3-vs-M3 release-fence isolation but on the load side instead.
- **+23.27 µs P3-F3 isolates the load-side acq_rel-fence cost on a CDNA3 CAS RMW** — far larger than the load-side acquire-fence cost (N3-M3 = −0.16 µs), confirming acq_rel ≠ acquire in fence-fusion behavior.
- **Fence-fusion non-generalization (FALSIFIES naive extension of K-2243 R-2243.2):** acquire-load fuses with acq_rel-CAS into a single L2 invalidate/flush; acq_rel-load does NOT fuse with acq_rel-CAS. Promote to PROPOSED **R-2246.1**.
- **CAS-on-CAS doubles P3** — M3/N3/O3 paired against P3 each add +41-48 µs, ~2× the baseline. Same L2 atomic dispatcher serialization as the K-2243 CAS-on-CAS finding.
- **CCL design rule (PROPOSED R-2246.2):** for any CAS handoff requiring full acq_rel semantics, prefer `acquire-load + acq_rel-CAS` (F3 pattern) over `acq_rel-load + acq_rel-CAS` (P3 pattern). The F3 pattern is **23 µs cheaper per iteration** with identical effective ordering on a single address.

## Methodology

Single-rank multi-CTA emulation (matches K-2240/K-2243 lineage). Focal P3 launched on stream A, interferer on stream B; per-cell median across 25 reps. Triton 3.6.0+rocm7.2.0 lowering. WARMUP=3 launches per cell to prime JIT and warmup the L2 atomic dispatcher. `torch.cuda.synchronize()` before/after each rep gate. Wall-clock via `time.perf_counter()`. P3 kernel: `tl.atomic_add(addr, 0, sem='acq_rel', scope='sys')` as no-op acq_rel-fenced load + `tl.atomic_cas(addr, cur, cur+1, sem='acq_rel', scope='sys')`. Identical scaffolding to K-2240 N3 except for the `sem=` argument.

## Files in this output dir

- `summary.md` — this file
- `v45_baseline.csv` — 4,375 P3 baseline rep-rows (sha256 in manifest)
- `v45_paired.csv`   — 74,375 P3 × 17 interferer paired rep-rows
- `v45_manifest.json` — full manifest with QC, sha256, schema, and per-interferer canonical medians
- `v45_cas_ordering_ladder.png` — CAS family ordering ladder including new P3
- `v45_xchg_vs_cas_ordering.png` — XCHG vs CAS cross-comparison with P3 starred
- `v45_paired_canonical.png` — paired-interference ranking
- `v45_p3_scaling.png` — P3 latency vs N_OPS (linear scaling)

## Push status — HALT-IS-A-DELIVERABLE (K-1996 protocol)

The science deliverable is complete; only the upload to `ryanswann-amd/comm_data` is blocked.
Reproducible evidence is in `push_evidence/push_attempt.log`. Verified facts (re-run on retry):

| check | result |
|---|---|
| `GH_TOKEN` length | **0** (empty in this sandbox) |
| `GITHUB_TOKEN` length | **0** (empty in this sandbox) |
| `GET api.github.com/user` (both tokens) | **401 Bad credentials** |
| `~/.ssh/` keys | **none** (only `known_hosts`) |
| `GET github.com/ryanswann-amd/comm_data` | **404** (repo does not exist) |
| `ryanswann-amd` public repo list | 15 repos, **comm_data not present** |
| `git push https://github.com/ryanswann-amd/comm_data.git v45` | exit 128, no terminal/credential helper |

**Hand-off package** (ready to publish in 2 commands once the repo+credentials exist):

- `v45_corpus_repo/` — git repo on branch `v45` with one signed-style commit containing all artifacts and a top-level `corpus_manifest.json` (lineage v42 → v44 → v45, 46th-primitive entry, file sha256s).
- `v45_corpus_v45.bundle` — single-file `git bundle` of that branch (sha256 `cbf2ca712c…`).
- `v45_corpus_v45.tgz` — tarball of the working-tree files only (sha256 `883de881be…`).
- `push_evidence/push_attempt_retry2.log` — re-verified 2026-05-11T06:19Z: identical blocker, same 401/404 fingerprint.

**Two-command finish (any authorized agent):**
```bash
cd /workspace/output/v45_corpus_repo
git remote add origin https://<TOKEN>@github.com/ryanswann-amd/comm_data.git
git push -u origin v45
```

If the repo does not yet exist, create it first:
```bash
curl -sS -H "Authorization: token <TOKEN>" -d '{"name":"comm_data"}' https://api.github.com/orgs/ryanswann-amd/repos
```

## Success criterion status

- **done**: SCIENCE deliverable COMPLETE (4,375 baseline + 74,375 paired rows, 0 nulls/zeros, manifest + 4 plots + summary, all PASS QC). PUBLICATION step requires credentials/repo not present in the sandbox (evidence above). Per K-1996 HALT-IS-A-DELIVERABLE: partial completion shipped with a turnkey push package and reproducible blocker log.
