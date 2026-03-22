#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Parameter derivation for the HBM-buffered all_gather_matmul kernel.

Given a problem size (M, N, K), world size, and per-link XGMI bandwidth,
derives kernel parameters that balance communication and computation in
the device-level pipeline.

The kernel fuses all-gather with GEMM using two workgroup roles:
  - Fetcher WGs: gather remote A tiles into an HBM staging buffer,
    setting per-tile ready flags as data arrives.
  - GEMM WGs: poll flags, then compute C += A_staged @ B tile-by-tile.

The M dimension is split into `num_fetch_stages` pipeline stages.  Each
stage's fetchers and GEMM WGs are interleaved in the launch grid so that
stage N+1's fetch overlaps with stage N's compute.

Pipeline timeline (S stages):
  |-- fetch stage 0 --|-- max(fetch, compute) * (S-1) --|-- compute last --|

Usage:
    python derive_params.py -m 131072 -n 2048 -k 16384
    python derive_params.py -m 196608 -n 2304 -k 16384 --link_bw 50
    python derive_params.py -m 196608 -n 2304 -k 16384 -v -b --trace

When --link_bw is omitted the script automatically profiles the XGMI
link bandwidth by timing GPU-to-GPU copies across all peer pairs visible
from GPU 0.
"""

import argparse
import math
import time

# ── MI300X hardware defaults ──────────────────────────────────────────────
DEFAULT_NUM_CUS = 304
DEFAULT_PEAK_TFLOPS_FP16 = 1300.0
DEFAULT_HBM_BW_GBPS = 5300.0
DEFAULT_L2_SIZE_BYTES = 256 * 1024 * 1024
DEFAULT_NUM_XCDS = 8
DEFAULT_WORLD_SIZE = 8

# Calibrated from MI300X trace data: the ratio of measured wall time to
# the CU-work-queue lower bound.  Captures WG dispatch overhead,
# cross-XCD coherence latency, and pipeline bubble effects.
DEFAULT_SCHEDULING_FACTOR = 4.5


def profile_link_bandwidth(world_size=DEFAULT_WORLD_SIZE):
    """Measure per-link unidirectional XGMI bandwidth.

    Copies a 256 MB fp16 tensor from GPU 0 to every other visible GPU,
    times the transfers with host-side timing after explicit device syncs,
    and returns the conservative (min) per-link bandwidth.
    """
    import torch

    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        raise RuntimeError(
            f"Need >= 2 visible GPUs for bandwidth profiling, found {n_gpus}. Pass --link_bw explicitly instead."
        )

    n_peers = min(world_size, n_gpus) - 1
    size_bytes = 256 * 1024 * 1024
    numel = size_bytes // 2
    warmup_iters = 10
    timed_iters = 40

    print(f"\n── Link Bandwidth Profiling {'─' * 43}")
    print(f"  GPUs visible:   {n_gpus}")
    print(f"  Testing:        GPU 0 → GPUs 1..{n_peers}")
    print(f"  Transfer size:  {size_bytes // (1024**2)} MB × {timed_iters} iterations\n")

    src = torch.empty(numel, dtype=torch.float16, device="cuda:0").normal_()
    bandwidths = []

    for peer in range(1, n_peers + 1):
        dst = torch.empty(numel, dtype=torch.float16, device=f"cuda:{peer}")

        for _ in range(warmup_iters):
            dst.copy_(src)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(peer)

        t_start = time.perf_counter()
        for _ in range(timed_iters):
            dst.copy_(src)
        torch.cuda.synchronize(peer)
        elapsed_s = time.perf_counter() - t_start

        bw = size_bytes * timed_iters / elapsed_s / 1e9
        bandwidths.append(bw)
        print(f"    GPU 0 → GPU {peer}:  {bw:6.1f} GB/s")

        del dst

    del src
    torch.cuda.empty_cache()

    bw_min = min(bandwidths)
    bw_max = max(bandwidths)
    bw_avg = sum(bandwidths) / len(bandwidths)
    print(f"\n  min = {bw_min:.1f}   avg = {bw_avg:.1f}   max = {bw_max:.1f} GB/s")
    print(f"  Using conservative (min): {bw_min:.1f} GB/s per link")

    return bw_min


# ── Tile / block size heuristics ──────────────────────────────────────────


def _choose_block_sizes(M, N, K, K_local):
    """Heuristic tile-size selection for MI300X MFMA."""
    bk = 64

    bm = 256 if M >= 8192 else 128
    while M % bm != 0 and bm > 64:
        bm //= 2

    if N >= 512:
        bn = 256
    elif N >= 256:
        bn = 256 if N % 256 == 0 else 128
    else:
        bn = 128
    while N % bn != 0 and bn > 32:
        bn //= 2

    while K % bk != 0 and bk > 16:
        bk //= 2
    while K_local % bk != 0 and bk > 16:
        bk //= 2

    nw = 8 if bm * bn >= 256 * 256 else 4
    return bm, bn, bk, nw


def _choose_k_per_flag(num_k_blocks, num_k_blocks_local, target_groups=8):
    """Pick k_per_flag so that flag groups align to rank boundaries when
    possible, falling back to the largest divisor near the target."""
    if num_k_blocks % num_k_blocks_local == 0:
        candidate = num_k_blocks_local
        groups = num_k_blocks // candidate
        if groups >= 4:
            return candidate

    kpf = max(1, num_k_blocks // target_groups)
    while num_k_blocks % kpf != 0 and kpf > 1:
        kpf -= 1
    return kpf


# ── Per-tile roofline model ──────────────────────────────────────────────


def _tile_roofline(bm, bn, bk, M, K, N, dtype_bytes, peak_tflops, hbm_bw_gbps, l2_size):
    """Compute achievable per-CU TFLOPS from tile arithmetic intensity.

    staged_a is always >> L2, so A tiles come from HBM.  B may fit in L2
    only when staged_a is small enough that reads don't evict B.
    Returns (roofline_tflops, tile_intensity, ridge_point, b_in_l2).
    """
    tile_flops = 2 * bm * bn * bk
    a_bytes = bm * bk * dtype_bytes
    b_bytes = bk * bn * dtype_bytes

    b_total = K * N * dtype_bytes
    staged_a_total = M * K * dtype_bytes
    # When staged_a exceeds L2, streaming GEMM reads evict B regardless
    # of B's absolute size.
    b_in_l2 = (staged_a_total <= l2_size) and (b_total <= l2_size)

    hbm_bytes = a_bytes + (0 if b_in_l2 else b_bytes)
    intensity = tile_flops / max(hbm_bytes, 1)

    ridge = peak_tflops * 1e3 / hbm_bw_gbps
    if intensity >= ridge:
        roofline = peak_tflops
    else:
        roofline = hbm_bw_gbps * intensity / 1e3

    return roofline, intensity, ridge, b_in_l2


# ── Per-WG execution time models ────────────────────────────────────────


def _gemm_wg_time_us(bm, bn, bk, K, num_flag_groups, roofline_tflops, num_cus):
    """Estimate per-WG GEMM execution time in microseconds.

    Uses the per-tile roofline to get the per-CU throughput, then applies
    a calibrated overhead for memory-latency hiding and instruction
    scheduling at single-WG occupancy (large tiles).
    """
    total_flops = 2 * bm * bn * K
    per_cu_tflops = roofline_tflops / num_cus

    # Roofline-ideal per-WG time
    ideal_us = total_flops / (per_cu_tflops * 1e6)

    # Single-occupancy overhead: imperfect latency hiding, instruction
    # scheduling gaps, cross-XCD coherence on staged_a reads.
    # Calibrated from MI300X traces: actual/ideal ≈ 1.2-1.3.
    occupancy_factor = 1.25 if bm * bn >= 256 * 256 else 1.10

    # Flag polling: acquire-semantics atomic per flag group
    flag_us = num_flag_groups * 2.5

    return ideal_us * occupancy_factor + flag_us


def _fetch_wg_time_us(bm, bk, kpf, world_size, link_bw, dtype_bytes, num_fgs_per_wg):
    """Estimate per-fetcher-WG execution time in microseconds.

    Each flag group fetches kpf K-blocks (each BM × BK) from one rank.
    Remote data traverses XGMI; local data uses HBM.
    """
    bytes_per_fg = bm * kpf * bk * dtype_bytes
    remote_frac = (world_size - 1) / world_size

    # XGMI gather: raw transfer + iris.x.gather software overhead
    remote_bytes = bytes_per_fg * remote_frac
    gather_overhead = 1.5
    xgmi_us = remote_bytes / (link_bw * 1e3) * gather_overhead

    # HBM write to staged_a (.cg → L2/HBM, per-WG share of bandwidth)
    write_bw = 15.0  # GB/s effective per fetcher WG (calibrated from traces)
    write_us = bytes_per_fg / (write_bw * 1e3)

    # Read and write overlap within each tile; dominant cost + flag-store
    per_fg_us = max(xgmi_us, write_us) + 5.0

    return num_fgs_per_wg * per_fg_us


# ── Kernel time estimation ───────────────────────────────────────────────


def _estimate_kernel_time(total_gemm_wgs, gemm_wg_us, total_fetch_wgs, fetch_wg_us, num_cus, scheduling_factor):
    """Estimate kernel wall-clock time from the CU work queue model.

    total_CU_work / num_CUs gives the ideal (work-conserving) lower
    bound.  The scheduling_factor captures GPU dispatch overhead,
    cross-XCD coherence, and pipeline bubble effects measured on MI300X.
    """
    total_cu_work_us = total_gemm_wgs * gemm_wg_us + total_fetch_wgs * fetch_wg_us

    ideal_ms = total_cu_work_us / num_cus / 1e3
    estimated_ms = ideal_ms * scheduling_factor
    return estimated_ms, ideal_ms


# ── Pipeline stage selection ─────────────────────────────────────────────


def _choose_fetch_stages(num_m_tiles, num_tiles_n, group_size_m, comm_time_ms, compute_time_ms, num_cus):
    """Choose num_fetch_stages for good pipeline efficiency while keeping
    m_per_stage divisible by group_size_m."""
    ratio = comm_time_ms / compute_time_ms if compute_time_ms > 0 else 999

    if ratio > 1.5:
        ideal_stages = 32
    elif ratio > 0.8:
        ideal_stages = 16
    elif ratio > 0.3:
        ideal_stages = 8
    else:
        ideal_stages = 4

    min_gemm_tiles = max(num_cus // 4, 32)
    min_m_per_stage = max(group_size_m, math.ceil(min_gemm_tiles / max(num_tiles_n, 1)))
    max_stages = max(1, num_m_tiles // min_m_per_stage)
    num_stages = min(ideal_stages, max_stages)
    num_stages = max(1, num_stages)

    m_per_stage = math.ceil(num_m_tiles / num_stages)
    if m_per_stage % group_size_m != 0:
        m_per_stage = ((m_per_stage + group_size_m - 1) // group_size_m) * group_size_m
        num_stages = max(1, math.ceil(num_m_tiles / m_per_stage))

    m_per_stage = math.ceil(num_m_tiles / num_stages)
    return num_stages, m_per_stage


# ── num_fetch_sms optimisation ───────────────────────────────────────────


def _choose_num_fetch_sms(
    m_per_stage,
    group_size_m,
    num_flag_groups_k,
    link_bw,
    world_size,
    num_cus,
    bm,
    bk,
    kpf,
    dtype_bytes,
    gemm_wg_us,
    gemm_tiles_per_stage,
):
    """Choose num_fetch_sms for good pipeline overlap.

    Balances three constraints:
      1. Flag delivery parallelism: ≥ m_per_stage so every M-tile gets
         a fetcher early (good for reducing GEMM flag-poll stalls).
      2. Link saturation: enough concurrent fetchers to use the XGMI
         aggregate bandwidth.
      3. CU budget: leave enough CUs for GEMM in the first dispatch wave.

    Returns (num_fetch_sms, per-WG timing info dict).
    """
    total_fg_per_stage = num_flag_groups_k * m_per_stage

    # Constraint 1: one fetcher per M-group for broad flag delivery
    parallel_min = max(1, m_per_stage // group_size_m)

    # Constraint 2: enough fetchers to keep XGMI links busy
    per_fg_bytes = bm * kpf * bk * dtype_bytes
    per_fg_remote = per_fg_bytes * (world_size - 1) / world_size
    per_fg_xgmi_us = per_fg_remote / (link_bw * 1e3) * 1.5
    per_fg_write_us = per_fg_bytes / (15.0 * 1e3)
    per_fg_us = max(per_fg_xgmi_us, per_fg_write_us) + 5.0

    # Total flag groups per stage should finish within the stage GEMM time
    gemm_waves = math.ceil(gemm_tiles_per_stage / num_cus)
    stage_gemm_us = gemm_waves * gemm_wg_us
    if per_fg_us > 0:
        balance_min = max(1, math.ceil(total_fg_per_stage * per_fg_us / stage_gemm_us))
    else:
        balance_min = 1

    nf = max(parallel_min, balance_min, 64)
    nf = min(nf, num_cus // 2)
    nf = max(1, nf)

    return nf


# ── Main derivation ──────────────────────────────────────────────────────


def derive(M, N, K, world_size, link_bw, num_cus, peak_tflops, hbm_bw_gbps, l2_size, scheduling_factor, dtype_bytes):
    K_local = K // world_size

    # 1. Tile sizes
    bm, bn, bk, nw = _choose_block_sizes(M, N, K, K_local)
    gm = 4
    num_m_tiles = M // bm
    num_tiles_n = math.ceil(N / bn)
    num_k_blocks = K // bk
    num_k_blocks_local = K_local // bk

    # 2. Per-tile roofline
    roofline_tflops, intensity, ridge, b_in_l2 = _tile_roofline(
        bm, bn, bk, M, K, N, dtype_bytes, peak_tflops, hbm_bw_gbps, l2_size
    )

    # 3. Communication model (link-limited)
    total_remote_bytes = M * K_local * (world_size - 1) * dtype_bytes
    total_link_bw = link_bw * (world_size - 1)
    comm_time_ms = total_remote_bytes / (total_link_bw * 1e9) * 1e3

    # 4. Compute model (roofline-limited)
    total_flops = 2 * M * N * K
    compute_time_ms = total_flops / (roofline_tflops * 1e12) * 1e3

    ratio = comm_time_ms / compute_time_ms if compute_time_ms > 0 else 999

    # 5. k_per_flag
    kpf = _choose_k_per_flag(num_k_blocks, num_k_blocks_local)
    num_flag_groups_k = num_k_blocks // kpf

    # 6. Pipeline stages
    num_stages, m_per_stage = _choose_fetch_stages(num_m_tiles, num_tiles_n, gm, comm_time_ms, compute_time_ms, num_cus)
    gemm_tiles_per_stage = m_per_stage * num_tiles_n

    # 7. first_stage_fetch_sms: use all CUs to fill the pipeline ASAP
    fsf = num_cus

    # 8. Per-WG timing
    gemm_wg_us_val = _gemm_wg_time_us(bm, bn, bk, K, num_flag_groups_k, roofline_tflops, num_cus)

    # 9. Choose num_fetch_sms
    nf = _choose_num_fetch_sms(
        m_per_stage,
        gm,
        num_flag_groups_k,
        link_bw,
        world_size,
        num_cus,
        bm,
        bk,
        kpf,
        dtype_bytes,
        gemm_wg_us_val,
        gemm_tiles_per_stage,
    )

    # 10. Compute per-WG fetch times
    total_fg_per_stage = num_flag_groups_k * m_per_stage
    fgs_per_wg_stg0 = max(1, math.ceil(total_fg_per_stage / fsf))
    fgs_per_wg_rest = max(1, math.ceil(total_fg_per_stage / nf))
    fetch_us_stg0 = _fetch_wg_time_us(bm, bk, kpf, world_size, link_bw, dtype_bytes, fgs_per_wg_stg0)
    fetch_us_rest = _fetch_wg_time_us(bm, bk, kpf, world_size, link_bw, dtype_bytes, fgs_per_wg_rest)

    # 11. Grid geometry
    first_stage_size = fsf + gemm_tiles_per_stage
    rest_stage_size = nf + gemm_tiles_per_stage
    grid_size = first_stage_size + rest_stage_size * max(0, num_stages - 1)
    total_fetch_wgs = fsf + nf * max(0, num_stages - 1)
    total_gemm_wgs = gemm_tiles_per_stage * num_stages

    # 12. Kernel time estimate (CU-work model)
    avg_fetch_us = fsf * fetch_us_stg0 + nf * max(0, num_stages - 1) * fetch_us_rest
    avg_fetch_us /= max(total_fetch_wgs, 1)
    est_kernel_ms, est_ideal_ms = _estimate_kernel_time(
        total_gemm_wgs, gemm_wg_us_val, total_fetch_wgs, avg_fetch_us, num_cus, scheduling_factor
    )

    # 13. Link-limited pipeline estimate (simple model for comparison)
    stage_m = m_per_stage * bm
    stage_comm_ms = stage_m * K_local * (world_size - 1) * dtype_bytes / (total_link_bw * 1e9) * 1e3
    stage_compute_ms = 2 * stage_m * N * K / (roofline_tflops * 1e12) * 1e3
    startup_ms = stage_comm_ms
    steady_ms = max(stage_comm_ms, stage_compute_ms) * max(0, num_stages - 1)
    drain_ms = stage_compute_ms
    pipeline_ms = startup_ms + steady_ms + drain_ms
    sequential_ms = comm_time_ms + compute_time_ms

    # 14. Standalone GEMM estimate (rocBLAS-class efficiency for comparison)
    standalone_gemm_eff = 0.30
    standalone_tflops = roofline_tflops * standalone_gemm_eff
    standalone_gemm_ms = total_flops / (standalone_tflops * 1e12) * 1e3
    pytorch_est_ms = comm_time_ms + standalone_gemm_ms

    staged_a_gb = M * K * dtype_bytes / (1024**3)

    return dict(
        block_size_m=bm,
        block_size_n=bn,
        block_size_k=bk,
        group_size_m=gm,
        num_warps=nw,
        num_fetch_sms=nf,
        k_per_flag=kpf,
        num_fetch_stages=num_stages,
        first_stage_fetch_sms=fsf,
        # derived
        K_local=K_local,
        num_m_tiles=num_m_tiles,
        num_tiles_n=num_tiles_n,
        num_k_blocks=num_k_blocks,
        num_flag_groups_k=num_flag_groups_k,
        m_per_stage=m_per_stage,
        gemm_tiles_per_stage=gemm_tiles_per_stage,
        grid_size=grid_size,
        total_fetch_wgs=total_fetch_wgs,
        total_gemm_wgs=total_gemm_wgs,
        # roofline
        roofline_tflops=roofline_tflops,
        tile_intensity=intensity,
        ridge_point=ridge,
        b_in_l2=b_in_l2,
        # per-WG timing
        gemm_wg_us=gemm_wg_us_val,
        fetch_wg_us_stg0=fetch_us_stg0,
        fetch_wg_us_rest=fetch_us_rest,
        # estimates
        total_remote_bytes=total_remote_bytes,
        total_link_bw=total_link_bw,
        comm_time_ms=comm_time_ms,
        total_flops=total_flops,
        compute_time_ms=compute_time_ms,
        ratio=ratio,
        stage_comm_ms=stage_comm_ms,
        stage_compute_ms=stage_compute_ms,
        pipeline_ms=pipeline_ms,
        sequential_ms=sequential_ms,
        est_kernel_ms=est_kernel_ms,
        est_ideal_ms=est_ideal_ms,
        standalone_gemm_ms=standalone_gemm_ms,
        pytorch_est_ms=pytorch_est_ms,
        staged_a_gb=staged_a_gb,
        scheduling_factor=scheduling_factor,
    )


# ── Formatting helpers ───────────────────────────────────────────────────


def _fmt_bytes(n):
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024:.1f} KB"


def _fmt_flops(n):
    if n >= 1e15:
        return f"{n / 1e15:.2f} PFLOPs"
    return f"{n / 1e12:.2f} TFLOPs"


def _fmt_tflops(t):
    return f"{t:.0f} TFLOPS"


# ── Analysis output ──────────────────────────────────────────────────────


def print_analysis(M, N, K, world_size, link_bw, p, passthrough_args, bw_profiled=False):
    K_local = p["K_local"]
    dtype_bytes = 2
    bound = "COMM-BOUND" if p["ratio"] > 1.0 else "COMPUTE-BOUND"

    print("=" * 72)
    print("  All-Gather Matmul HBM-Buffer — Parameter Derivation")
    print("=" * 72)

    # ── Problem ───────────────────────────────────────────────────────
    print(f"\n{'Problem':>14}:  C({M}, {N}) = all_gather(A_shard({M}, {K_local})) @ B({K}, {N})")
    print(f"{'World size':>14}:  {world_size} GPUs")
    print(f"{'Dtype':>14}:  fp16 ({dtype_bytes}B)")

    # ── Data sizes ────────────────────────────────────────────────────
    a_shard = M * K_local * dtype_bytes
    b_size = K * N * dtype_bytes
    c_size = M * N * dtype_bytes
    staged = M * K * dtype_bytes
    print(f"\n{'A_shard':>14}:  ({M}, {K_local})  {_fmt_bytes(a_shard)}")
    print(f"{'B':>14}:  ({K}, {N})  {_fmt_bytes(b_size)}")
    print(f"{'C':>14}:  ({M}, {N})  {_fmt_bytes(c_size)}")
    print(f"{'staged_a':>14}:  ({M}, {K})  {_fmt_bytes(staged)}")
    if staged > 4 * 1024**3:
        print(f"{'':>14}   *** > 4 GB: requires int64 pointer arithmetic ***")

    # ── Per-tile roofline ─────────────────────────────────────────────
    print(f"\n── Roofline {'─' * 59}")
    print(f"{'Tile':>14}:  ({p['block_size_m']}, {p['block_size_n']}, {p['block_size_k']})")
    print(f"{'Intensity':>14}:  {p['tile_intensity']:.0f} FLOPs/byte {'(B in L2)' if p['b_in_l2'] else '(B from HBM)'}")
    print(f"{'Ridge point':>14}:  {p['ridge_point']:.0f} FLOPs/byte")
    region = "COMPUTE" if p["tile_intensity"] >= p["ridge_point"] else "MEMORY"
    print(f"{'Roofline':>14}:  {_fmt_tflops(p['roofline_tflops'])}  ({region}-bound tiles)")

    # ── Communication ─────────────────────────────────────────────────
    print(f"\n── Communication {'─' * 54}")
    print(f"{'Remote bytes':>14}:  {_fmt_bytes(p['total_remote_bytes'])}  (from {world_size - 1} peers)")
    bw_src = "profiled" if bw_profiled else "user"
    print(
        f"{'Link BW':>14}:  {link_bw:.1f} GB/s/link × {world_size - 1} links "
        f"= {p['total_link_bw']:.0f} GB/s aggregate  ({bw_src})"
    )
    print(f"{'Comm time':>14}:  {p['comm_time_ms']:.3f} ms  (link-limited)")

    # ── Compute ───────────────────────────────────────────────────────
    print(f"\n── Compute {'─' * 60}")
    print(f"{'Total FLOPs':>14}:  {_fmt_flops(p['total_flops'])}")
    print(f"{'Roofline time':>14}:  {p['compute_time_ms']:.3f} ms  (at {_fmt_tflops(p['roofline_tflops'])})")
    print(f"{'Comm/Compute':>14}:  {p['ratio']:.2f}x  →  {bound}")

    # ── Per-WG timing ─────────────────────────────────────────────────
    print(f"\n── Per-WG Model {'─' * 55}")
    print(f"{'GEMM WG':>14}:  {p['gemm_wg_us']:.0f} us  ({p['total_flops'] / p['total_gemm_wgs'] / 1e9:.2f} GFLOPs/WG)")
    print(f"{'Fetch WG stg0':>14}:  {p['fetch_wg_us_stg0']:.0f} us")
    if p["num_fetch_stages"] > 1:
        print(f"{'Fetch WG rest':>14}:  {p['fetch_wg_us_rest']:.0f} us")

    # ── Pipeline ──────────────────────────────────────────────────────
    S = p["num_fetch_stages"]
    print(f"\n── Pipeline {'─' * 59}")
    print(f"{'Stages (S)':>14}:  {S}")
    print(f"{'M tiles/stage':>14}:  {p['m_per_stage']}  ({p['m_per_stage'] * p['block_size_m']} rows)")
    print(
        f"{'GEMM WGs/stg':>14}:  {p['gemm_tiles_per_stage']}  ({p['m_per_stage']} m-tiles × {p['num_tiles_n']} n-tiles)"
    )
    print(f"{'K flag groups':>14}:  {p['num_flag_groups_k']}  (k_per_flag={p['k_per_flag']})")
    print(f"{'Stage comm':>14}:  {p['stage_comm_ms']:.3f} ms")
    print(f"{'Stage compute':>14}:  {p['stage_compute_ms']:.3f} ms")

    # ── Grid ──────────────────────────────────────────────────────────
    print(f"\n── Grid Layout {'─' * 56}")
    print(
        f"{'Stage 0':>14}:  {p['first_stage_fetch_sms']} fetchers + "
        f"{p['gemm_tiles_per_stage']} GEMM  = "
        f"{p['first_stage_fetch_sms'] + p['gemm_tiles_per_stage']} WGs"
    )
    if S > 1:
        print(
            f"{'Stages 1..{}'.format(S - 1):>14}:  {p['num_fetch_sms']} fetchers + "
            f"{p['gemm_tiles_per_stage']} GEMM  = "
            f"{p['num_fetch_sms'] + p['gemm_tiles_per_stage']} WGs  (×{S - 1})"
        )
    print(f"{'Total grid':>14}:  {p['grid_size']} WGs  ({p['total_fetch_wgs']} fetch + {p['total_gemm_wgs']} GEMM)")

    # ── Time estimates ────────────────────────────────────────────────
    print(f"\n── Time Estimates {'─' * 53}")
    print(f"{'CU-work lower':>14}:  {p['est_ideal_ms']:.1f} ms  (total WG time / {DEFAULT_NUM_CUS} CUs)")
    print(f"{'Fused kernel':>14}:  {p['est_kernel_ms']:.1f} ms  (×{p['scheduling_factor']:.1f} scheduling overhead)")
    est_tflops = p["total_flops"] / (p["est_kernel_ms"] * 1e-3) / 1e12
    print(
        f"{'Est. TFLOPS':>14}:  {est_tflops:.0f} TFLOPS  ({est_tflops / p['roofline_tflops'] * 100:.0f}% of roofline)"
    )
    print(f"{'':>14}")
    print(
        f"{'PyTorch est.':>14}:  {p['pytorch_est_ms']:.1f} ms  "
        f"(all_gather {p['comm_time_ms']:.1f} + matmul {p['standalone_gemm_ms']:.1f})"
    )
    if p["est_kernel_ms"] < p["pytorch_est_ms"]:
        speedup = p["pytorch_est_ms"] / p["est_kernel_ms"]
        print(f"{'Fused speedup':>14}:  {speedup:.2f}x over sequential PyTorch")
    else:
        slowdown = p["est_kernel_ms"] / p["pytorch_est_ms"]
        print(f"{'Fused speedup':>14}:  {1 / slowdown:.2f}x (slower than sequential by {slowdown:.2f}x)")

    # ── Recommended parameters ────────────────────────────────────────
    print(f"\n── Recommended Kernel Parameters {'─' * 38}")
    params = [
        ("block_size_m", p["block_size_m"]),
        ("block_size_n", p["block_size_n"]),
        ("block_size_k", p["block_size_k"]),
        ("group_size_m", p["group_size_m"]),
        ("num_fetch_sms", p["num_fetch_sms"]),
        ("k_per_flag", p["k_per_flag"]),
        ("num_warps", p["num_warps"]),
        ("num_fetch_stages", p["num_fetch_stages"]),
        ("first_stage_fetch_sms", p["first_stage_fetch_sms"]),
    ]
    for name, val in params:
        print(f"  --{name:30s} {val}")

    # ── Command line ──────────────────────────────────────────────────
    extra = " ".join(passthrough_args)
    if extra:
        extra = " " + extra
    cmd = (
        f"HSA_NO_SCRATCH_RECLAIM=1 torchrun --nproc_per_node {world_size} "
        f"benchmark/ops/all_gather_matmul/benchmark_hbm_buffer.py "
        f"-m {M} -n {N} -k {K} "
        f"--block_size_m {p['block_size_m']} "
        f"--block_size_n {p['block_size_n']} "
        f"--block_size_k {p['block_size_k']} "
        f"--group_size_m {p['group_size_m']} "
        f"--num_fetch_sms {p['num_fetch_sms']} "
        f"--k_per_flag {p['k_per_flag']} "
        f"--num_warps {p['num_warps']} "
        f"--num_fetch_stages {p['num_fetch_stages']} "
        f"--first_stage_fetch_sms {p['first_stage_fetch_sms']}"
        f"{extra}"
    )
    print(f"\n── Command {'─' * 60}")
    print(f"  {cmd}")
    print()


def derive_col_parallel(M, N, K, world_size, link_bw, num_cus, peak_tflops, hbm_bw_gbps, l2_size, scheduling_factor, dtype_bytes):
    """Derive parameters for column-parallel (M-sharded) all_gather_matmul.

    Col-parallel layout:
      - A_local[M/ws, K] per GPU -> gather along M -> staged_a[M, K]
      - B_local[K, N/ws] per GPU (no gather)
      - C_local[M, N/ws] output

    Key difference from row-parallel: each M-block from a remote rank has ALL
    K columns, so GEMM can process it immediately once staged. The k_per_flag
    should ideally equal num_k_blocks (1 flag per M-tile).
    """
    M_local = M // world_size
    N_local = N // world_size

    # 1. Tile sizes (use N_local for block size selection)
    bm = 256 if M >= 8192 else 128
    while M % bm != 0 and bm > 64:
        bm //= 2
    while M_local % bm != 0 and bm > 64:
        bm //= 2

    # Col-parallel: N_local is typically small (1024 for 8-GPU with N=8192).
    # bn=128 gives more N-tiles for better GEMM occupancy than bn=256.
    if N_local >= 512:
        bn = 128
    elif N_local >= 256:
        bn = 128 if N_local % 128 == 0 else 64
    else:
        bn = min(128, N_local)
    while N_local % bn != 0 and bn > 32:
        bn //= 2

    bk = 64
    while K % bk != 0 and bk > 16:
        bk //= 2

    nw = 8 if bm * bn >= 256 * 256 else 4
    # Col-parallel: gm=8 balances M-tile locality with GEMM dispatch
    gm = 8

    num_m_tiles = M // bm
    num_m_tiles_local = M_local // bm
    num_tiles_n = math.ceil(N_local / bn)
    num_k_blocks = K // bk

    # 2. Per-tile roofline (use N_local since B is local)
    roofline_tflops, intensity, ridge, b_in_l2 = _tile_roofline(
        bm, bn, bk, M, K, N_local, dtype_bytes, peak_tflops, hbm_bw_gbps, l2_size
    )

    # 3. Communication: gather M_local*K from each remote rank
    total_remote_bytes = M_local * K * (world_size - 1) * dtype_bytes
    total_link_bw = link_bw * (world_size - 1)
    comm_time_ms = total_remote_bytes / (total_link_bw * 1e9) * 1e3

    # 4. Compute: C_local[M, N_local] = staged_a[M, K] @ B_local[K, N_local]
    total_flops = 2 * M * N_local * K
    compute_time_ms = total_flops / (roofline_tflops * 1e12) * 1e3

    ratio = comm_time_ms / compute_time_ms if compute_time_ms > 0 else 999

    # 5. k_per_flag: for col-parallel, use all K blocks per flag (1 flag per M-tile)
    kpf = num_k_blocks
    num_flag_groups_k = 1

    # 6. Pipeline stages: col-parallel benefits from 1 stage because
    # each M-block is independent — no cross-rank K accumulation.
    # All fetchers run in the first CU wave, GEMM fills subsequent waves.
    num_stages = 1
    m_per_stage = num_m_tiles
    gemm_tiles_per_stage = m_per_stage * num_tiles_n

    # 7. num_fetch_sms: use most CUs for fetching to finish ASAP
    # Empirically ~290/304 CUs on MI300X is the sweet spot
    nf = max(1, num_cus - 14)
    fsf = nf

    # 8. Per-WG timing
    gemm_wg_us_val = _gemm_wg_time_us(bm, bn, bk, K, num_flag_groups_k, roofline_tflops, num_cus)

    # 10. Per-WG fetch times
    total_fg_per_stage = num_flag_groups_k * m_per_stage
    fgs_per_wg_stg0 = max(1, math.ceil(total_fg_per_stage / fsf))
    fgs_per_wg_rest = max(1, math.ceil(total_fg_per_stage / nf))
    fetch_us_stg0 = _fetch_wg_time_us(bm, bk, kpf, world_size, link_bw, dtype_bytes, fgs_per_wg_stg0)
    fetch_us_rest = _fetch_wg_time_us(bm, bk, kpf, world_size, link_bw, dtype_bytes, fgs_per_wg_rest)

    # 11. Grid geometry
    first_stage_size = fsf + gemm_tiles_per_stage
    rest_stage_size = nf + gemm_tiles_per_stage
    grid_size = first_stage_size + rest_stage_size * max(0, num_stages - 1)
    total_fetch_wgs = fsf + nf * max(0, num_stages - 1)
    total_gemm_wgs = gemm_tiles_per_stage * num_stages

    # 12. Kernel time estimate
    avg_fetch_us = fsf * fetch_us_stg0 + nf * max(0, num_stages - 1) * fetch_us_rest
    avg_fetch_us /= max(total_fetch_wgs, 1)
    est_kernel_ms, est_ideal_ms = _estimate_kernel_time(
        total_gemm_wgs, gemm_wg_us_val, total_fetch_wgs, avg_fetch_us, num_cus, scheduling_factor
    )

    # 13. Pipeline estimate
    stage_m = m_per_stage * bm
    stage_comm_ms = stage_m * K * (world_size - 1) * dtype_bytes / (total_link_bw * 1e9) * 1e3 / world_size
    stage_compute_ms = 2 * stage_m * N_local * K / (roofline_tflops * 1e12) * 1e3
    pipeline_ms = stage_comm_ms + max(stage_comm_ms, stage_compute_ms) * max(0, num_stages - 1) + stage_compute_ms
    sequential_ms = comm_time_ms + compute_time_ms

    # 14. PyTorch estimate
    standalone_gemm_eff = 0.30
    standalone_tflops = roofline_tflops * standalone_gemm_eff
    standalone_gemm_ms = total_flops / (standalone_tflops * 1e12) * 1e3
    pytorch_est_ms = comm_time_ms + standalone_gemm_ms

    staged_a_gb = M * K * dtype_bytes / (1024**3)

    return dict(
        block_size_m=bm, block_size_n=bn, block_size_k=bk,
        group_size_m=gm, num_warps=nw,
        num_fetch_sms=nf, k_per_flag=kpf,
        num_fetch_stages=num_stages, first_stage_fetch_sms=fsf,
        K_local=K, M_local=M_local, N_local=N_local,
        num_m_tiles=num_m_tiles, num_tiles_n=num_tiles_n,
        num_k_blocks=num_k_blocks, num_flag_groups_k=num_flag_groups_k,
        m_per_stage=m_per_stage, gemm_tiles_per_stage=gemm_tiles_per_stage,
        grid_size=grid_size, total_fetch_wgs=total_fetch_wgs, total_gemm_wgs=total_gemm_wgs,
        roofline_tflops=roofline_tflops, tile_intensity=intensity,
        ridge_point=ridge, b_in_l2=b_in_l2,
        gemm_wg_us=gemm_wg_us_val,
        fetch_wg_us_stg0=fetch_us_stg0, fetch_wg_us_rest=fetch_us_rest,
        total_remote_bytes=total_remote_bytes, total_link_bw=total_link_bw,
        comm_time_ms=comm_time_ms, total_flops=total_flops,
        compute_time_ms=compute_time_ms, ratio=ratio,
        stage_comm_ms=stage_comm_ms, stage_compute_ms=stage_compute_ms,
        pipeline_ms=pipeline_ms, sequential_ms=sequential_ms,
        est_kernel_ms=est_kernel_ms, est_ideal_ms=est_ideal_ms,
        standalone_gemm_ms=standalone_gemm_ms, pytorch_est_ms=pytorch_est_ms,
        staged_a_gb=staged_a_gb, scheduling_factor=scheduling_factor,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Derive parameters for HBM-buffered all_gather_matmul.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-m", type=int, required=True, help="M dimension (rows of output)")
    parser.add_argument("-n", type=int, required=True, help="N dimension (cols of output)")
    parser.add_argument("-k", type=int, required=True, help="K dimension (total reduction dim)")
    parser.add_argument("--world_size", type=int, default=DEFAULT_WORLD_SIZE, help="Number of GPUs")
    parser.add_argument(
        "--link_bw",
        type=float,
        default=None,
        help="Per-link XGMI bandwidth in GB/s (one direction). Omit to auto-profile via GPU-to-GPU copies.",
    )
    parser.add_argument("--num_cus", type=int, default=DEFAULT_NUM_CUS, help="Number of compute units")
    parser.add_argument("--peak_tflops", type=float, default=DEFAULT_PEAK_TFLOPS_FP16, help="Peak fp16 TFLOPS")
    parser.add_argument("--hbm_bw", type=float, default=DEFAULT_HBM_BW_GBPS, help="HBM bandwidth in GB/s")
    parser.add_argument(
        "--scheduling_factor",
        type=float,
        default=DEFAULT_SCHEDULING_FACTOR,
        help="CU scheduling overhead factor (calibrated from traces)",
    )
    parser.add_argument(
        "--col_parallel", action="store_true",
        help="Derive for column-parallel (M-sharded A) instead of row-parallel (K-sharded A)",
    )

    args, passthrough = parser.parse_known_args()

    if not args.col_parallel and args.k % args.world_size != 0:
        parser.error(f"K ({args.k}) must be divisible by world_size ({args.world_size})")
    if args.col_parallel and args.m % args.world_size != 0:
        parser.error(f"M ({args.m}) must be divisible by world_size ({args.world_size}) for col-parallel")
    if args.col_parallel and args.n % args.world_size != 0:
        parser.error(f"N ({args.n}) must be divisible by world_size ({args.world_size}) for col-parallel")

    link_bw = args.link_bw
    bw_profiled = False
    if link_bw is None:
        try:
            link_bw = profile_link_bandwidth(args.world_size)
            bw_profiled = True
        except Exception as e:
            print(f"\n  Auto-profiling failed: {e}")
            print("  Falling back to --link_bw 50 (MI300X default)\n")
            link_bw = 50.0

    derive_fn = derive_col_parallel if args.col_parallel else derive
    p = derive_fn(
        args.m,
        args.n,
        args.k,
        args.world_size,
        link_bw,
        args.num_cus,
        args.peak_tflops,
        args.hbm_bw,
        DEFAULT_L2_SIZE_BYTES,
        args.scheduling_factor,
        dtype_bytes=2,
    )

    print_analysis(args.m, args.n, args.k, args.world_size, link_bw, p, passthrough, bw_profiled=bw_profiled)


if __name__ == "__main__":
    main()
