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
