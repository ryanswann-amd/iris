# K-679 — iris all_reduce 5-bucket latency decomposition (1 KB / 4 KB / 16 KB on 8× MI300X)

Per-iter latency decomposition for iris `ccl.all_reduce` (`one_shot`,
`two_shot`) and RCCL `dist.all_reduce` at small message sizes (1 / 4 /
16 KB, bf16) on 8× MI300X (UBB, c42).

## Buckets (per iter, ns)

1. **host_launch** — CPU `perf_counter` around the kernel-launch / `dist.all_reduce` enqueue.
2. **device_barrier** — CPU time in `ctx.device_barrier(...)` (iris) or `torch.cuda.synchronize()` (RCCL).
3. **xgmi_transfer** — modeled max-link-bytes / 64 GB/s SOL.
4. **local_reduction** — cudaEvent kernel time − barrier − xgmi (clipped); GPU-side AR-kernel duration.
5. **epilogue_sync** — residual after the four buckets.

`wall = host_launch + device_barrier` (CPU iter time). Buckets do not all
sum to wall — `local_reduction` overlaps with `device_barrier` wait.

## Run

```bash
# Inside an 8-GPU container, with iris installed and PYTHONPATH set:
bash benchmark/ccl/k679/run_K679.sh run1
bash benchmark/ccl/k679/run_K679.sh run2
bash benchmark/ccl/k679/run_K679.sh run3
python3 benchmark/ccl/k679/aggregate_K679.py \
    --in_dir output --out_dir output/aggregate
```

The bench applies two runtime monkey-patches to iris:
- **K-482** — skips `workspace.prepared = False` so `one_shot` doesn't re-fire `ctx.barrier()` on every call (would cost ~530 µs gloo barrier).
- **K-402** — swaps `ctx.barrier()` → `ctx.device_barrier(group=...)` so the post-launch barrier is the on-device atomic barrier (~17 µs).

## Result

See [`SUMMARY.md`](SUMMARY.md) and [`results_pivot.csv`](results_pivot.csv).
TL;DR: **post-K-402 iris one_shot is 9–11 % faster than RCCL** at 1–16 KB
on 8× MI300X. Critical-path gain is the **device_barrier** bucket
(iris −8 to −10 µs); critical-path loss is the **host_launch** bucket
(iris +5 to +7 µs).

## v2 update (RETRY) — review feedback

`bench_ar_5bucket_v2.py` + `aggregate_K679_v2.py` address 3 reviewer asks:

1. **rocprofv3 deviation explicit.** PRD called for rocprofv3 PC sampling
   for bucket attribution. The baseline image ships rocprofv3 1.1.0 which
   does **not** expose `--pc-sampling`. A reproducibility shim
   (`_rocprof_run.sh` + `bench_kernel_trace.py` + `run_kernel_trace.sh`)
   is committed; under torchrun it deadlocks the elastic agent during
   NCCL bootstrap (single rocprofv3 attached to LOCAL_RANK=0 reproduces).
   Empirical fallback: `cudaEvent.elapsed_time` (HSA hardware timestamp,
   1 ns resolution). See `SUMMARY_v2.md` "Methodology — rocprofv3 PC
   sampling deviation (explicit)" for full rationale.
2. **Dead amdsmi path removed.** `amdsmi_get_link_metrics` was never
   sampling these <1 s bursts (firmware poll cadence). Removed the SMI
   read entirely. xGMI bucket is now the analytical floor only.
3. **Pre-aggregation + clamp tracking.** Bench emits `{med,p90,p99,mean,std,n,
   clamp_count, epilogue_negative_count}` per bucket per (variant,size,rank)
   instead of 2000 raw per-iter values. Tarball is **51 KB** vs the 8.6 MB
   v1 tarball (~170× reduction). `K679_clamp_report.csv` shows clamp
   count per (run, variant, bytes, rank): 2 of 216 cells had a single
   iter clamp (0.05 %), confirming the model never over-attributes.

