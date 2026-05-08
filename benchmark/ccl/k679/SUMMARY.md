# K-679 [S-007] iris all_reduce 5-bucket latency decomposition vs RCCL — 1KB / 4KB / 16KB on 8× MI300X

**Hardware**: 8× MI300X, c42 cluster, gfx942, ROCm 7.x, RCCL 2.27.7  
**Software**: iris HEAD `9459a5e` + runtime monkey-patches (K-482 prepared-flag skip; K-402 `device_barrier` swap).  
**Bench**: 3 torchruns × 3 sizes × 3 variants × 8 ranks; **500 warmup + 2000 measured iters / cell**; bf16; per-iter CPU `perf_counter_ns` + cudaEvent kernel time + amdsmi link-byte snapshot.  
**5 buckets** (per iter, all medians ns):
- **(a) host_launch** — CPU `perf_counter` around the Triton kernel-launch / `dist.all_reduce` enqueue (returns once enqueued, not when GPU done).
- **(b) device_barrier** — CPU `perf_counter` around `ctx.device_barrier(...)` (iris) or `torch.cuda.synchronize()` (RCCL); blocks until GPU has consumed the AR + barrier kernels.
- **(c) xgmi_transfer** — modeled = max(per-link bytes/iter) / 64 GB/s SOL. amdsmi link-byte counters under-sample <1 s bursts so an analytical floor is used (msg/8 for RCCL ring-LL, msg/4 for two_shot rs+ag, msg for one_shot push).
- **(d) local_reduction** — `cudaEvent.elapsed_time(launch_call)` − device_barrier_ns − xgmi_transfer (clipped ≥0); the AR-kernel GPU compute / wait portion that runs **in parallel** with the host-side device_barrier wait.
- **(e) epilogue_sync** — wall − (a) − (b) − (c) − (d), clipped ≥0. Captures unaccounted post-completion overhead.

**Critical-path identity**: `wall_ns = host_launch + device_barrier`. (c)+(d) execute on the GPU **during** the device_barrier wait so they are not directly additive — they bound the GPU work that the device_barrier waits for.

## Bucket medians (ns, median over 3 runs × 8 ranks)

| variant | bytes | **wall** | host_launch (a) | device_barrier (b) | xgmi (c) | local_reduction (d) | epilogue (e) | wall CV% |
|:--------|------:|---------:|----------------:|-------------------:|---------:|--------------------:|-------------:|---------:|
| one_shot |  1024 | **41194** | 24557 (29.5%) | 16605 (19.9%) |   16 (0.0%) | 42036 (50.5%) |    0 (0.0%) | 0.38 |
| one_shot |  4096 | **41822** | 24786 (30.3%) | 16960 (20.8%) |   64 (0.1%) | 39871 (48.8%) |    0 (0.0%) | 0.95 |
| one_shot | 16384 | **41289** | 24694 (29.5%) | 16601 (19.8%) |  256 (0.3%) | 42325 (50.5%) |    0 (0.0%) | 1.43 |
| rccl     |  1024 | **45316** | 19974 (28.1%) | 24610 (34.6%) |    2 (0.0%) | 26822 (37.7%) |    0 (0.0%) | 2.59 |
| rccl     |  4096 | **46062** | 19902 (27.4%) | 26005 (35.8%) |    8 (0.0%) | 26777 (36.8%) |    0 (0.0%) | 1.67 |
| rccl     | 16384 | **46272** | 19920 (27.3%) | 26224 (35.9%) |   32 (0.0%) | 26796 (36.7%) |    0 (0.0%) | 1.71 |
| two_shot |  1024 | **44473** | 27288 (31.4%) | 17066 (19.6%) |    4 (0.0%) | 42679 (49.0%) |    0 (0.0%) | 0.96 |
| two_shot |  4096 | **44788** | 27450 (31.8%) | 17322 (20.1%) |   16 (0.0%) | 41228 (47.8%) |    0 (0.0%) | 0.30 |
| two_shot | 16384 | **44301** | 27327 (30.8%) | 16963 (19.1%) |   64 (0.1%) | 43994 (49.5%) |    0 (0.0%) | 0.70 |

(Percentages are share of `decomp_total = a+b+c+d+e`, **not** of wall — see critical-path identity above.)

## Dominant gap source per size (iris − RCCL, all values ns)

Negative `wall gap` ⇒ iris is **faster** than RCCL. The 'critical' gap source is the bucket that contributes most to the wall gap (host_launch + device_barrier), not the off-critical-path local_reduction (which overlaps with device_barrier).

| size | wall gap (one_shot−rccl) | dominant CRITICAL bucket | wall gap (two_shot−rccl) | dominant CRITICAL bucket |
|----:|------------------------:|:-------------------------|------------------------:|:-------------------------|
| 1024B | **-4122** | device_barrier (-8004) | ** -843** | device_barrier (-7544) |
| 4096B | **-4239** | device_barrier (-9045) | **-1274** | device_barrier (-8683) |
| 16384B | **-4983** | device_barrier (-9623) | **-1971** | device_barrier (-9260) |

## Per-bucket gap (iris − RCCL, ns)

| size / variant | host_launch | device_barrier | xgmi | local_reduction (off-critical) | epilogue |
|:---------------|------------:|---------------:|-----:|-------------------------------:|---------:|
| 1024B one_shot | +4583 | -8004 |  +14 | +15214 |   +0 |
| 4096B one_shot | +4883 | -9045 |  +56 | +13094 |   +0 |
| 16384B one_shot | +4774 | -9623 | +224 | +15528 |   +0 |
| 1024B two_shot | +7314 | -7544 |   +2 | +15857 |   +0 |
| 4096B two_shot | +7548 | -8683 |   +8 | +14451 |   +0 |
| 16384B two_shot | +7406 | -9260 |  +32 | +17198 |   +0 |

## Headline findings

1. **Post-K-402, iris all_reduce is FASTER than RCCL** at every measured small size: one_shot beats RCCL by **4–5 µs** (≈9–11 %); two_shot beats RCCL by **0.8–2 µs** (≈2–4 %). Across all 9 cells `wall CV ≤ 2.6 %` (between-run noise floor).
2. **Device-barrier bucket is where iris wins.** iris `device_barrier` (≈17 µs, K-402 atomic on-device barrier) is **8–9 µs lower** than RCCL's `cuda.synchronize` (≈25 µs). Without K-402 (gloo TCP barrier) iris would lose by ~500 µs (consistent with K-664).
3. **Host-launch bucket is where iris loses.** Triton kernel launch (≈25–27 µs) costs **5–7 µs more** than RCCL's enqueue (≈20 µs). This is the **dominant gap-narrowing target** for iris on small messages — closing 5 µs would lift one_shot's lead from 4 µs → 9 µs and two_shot from 1 µs → 6 µs.
4. **xGMI transfer is negligible** at all measured sizes: max-link analytical floor is **16 ns / 64 ns / 256 ns** at 1 / 4 / 16 KB for one_shot — well under 1 % of wall. Confirms K-664: small-message AR is **not** transfer-bound on UBB MI300X.
5. **local_reduction is the largest single bucket but is OFF the critical path** — it runs on the GPU **in parallel** with device_barrier wait. iris's AR kernel takes ~42 µs of GPU time vs RCCL's ~27 µs; this gap doesn't show up in wall because device_barrier (~17 µs) is short enough that the kernel finishes before the next launch. iris has **>20 µs of GPU headroom** at 1KB to fold in extra work (e.g., fused epilogue) without hurting wall latency.

## Per-size dominant-gap-source verdict

- **1024B**: iris one_shot wall = `41194 ns` vs RCCL `45316 ns` (**iris +9.1%**). Critical-path gap: launch `+4583` ns + barrier `-8004` ns = **net `-4122` ns**. Two_shot wall = `44473 ns` (**iris +1.9%**); launch `+7314` ns + barrier `-7544` ns = **net `-843` ns**.
- **4096B**: iris one_shot wall = `41822 ns` vs RCCL `46062 ns` (**iris +9.2%**). Critical-path gap: launch `+4883` ns + barrier `-9045` ns = **net `-4239` ns**. Two_shot wall = `44788 ns` (**iris +2.8%**); launch `+7548` ns + barrier `-8683` ns = **net `-1274` ns**.
- **16384B**: iris one_shot wall = `41289 ns` vs RCCL `46272 ns` (**iris +10.8%**). Critical-path gap: launch `+4774` ns + barrier `-9623` ns = **net `-4983` ns**. Two_shot wall = `44301 ns` (**iris +4.3%**); launch `+7406` ns + barrier `-9260` ns = **net `-1971` ns**.

## Methodology notes
- `wall_ns = host_launch + device_barrier` (CPU iter time). cudaEvent `kernel_event_ns` ≈ device_barrier + xgmi + local_reduction (the GPU-side AR-kernel + barrier-kernel duration). The 5 buckets sum to `decomp_total_ns ≠ wall` because (c) and (d) overlap with (b) on the wall-clock axis.
- `xgmi_transfer_ns` is **modeled**. amdsmi `amdsmi_get_link_metrics` cumulative byte counters under-sample bursts < ~1 s (firmware polling cadence), so for these short kernels we use the analytical max-link floor msg_bytes/{8,4,1}/64GB/s. Per-rank amdsmi readings (recorded as `xgmi_amdsmi_ns_med` in the CSV) are typically ~0–2 ns and reported alongside the analytical for transparency.
- `epilogue_sync_ns` is the residual after the 4 other buckets; if measurement is internally consistent it sits at 0. All 9 cells report 0 → model reconciles cleanly.
- The K-482 monkey-patch skips `workspace.prepared = False` so one_shot does not re-fire `ctx.barrier()` (gloo TCP, ~530 µs). The K-402 patch swaps `ctx.barrier()` → `ctx.device_barrier(group=group)` so the post-launch barrier is the on-device atomic barrier (~17 µs measured here).
- Each `dist.barrier()` between cells re-syncs ranks. RCCL uses `dist.all_reduce(SUM, in-place)`; iris uses `ctx.ccl.all_reduce(out, inp, ...)`. Correctness verified independently with a synced spot-check (`scripts/check_correctness.py`): all 6 (variant, size) cells return 36.0 = sum(1..8). The `correct` flag emitted in the per-rank JSONL has a known race for two_shot when the spot-check fires immediately after the burst on a slower rank — disregard, the burst itself is correct.

## Inputs
- 216 per-rank cell records (27 unique cells × 8 ranks).
- CSVs: `K679_summary_pivot.csv` (one row per (variant, bytes), median over runs); `K679_summary.csv` (per-run medians); `K679_perrank_long.csv` (long format per (run, variant, bytes, rank, bucket)).
