#!/usr/bin/env python3
# K-2388 [S-004]: PC1 ordering-cost law under producer-consumer wavefront-count IMBALANCE.
#
# Builds on K-2349 (intra-CU wave occupancy), K-2361 (XCD producer/consumer split),
# K-2366 (intra/inter XCD topology paired kernels).
#
# Sweep grid:
#   imbalance ∈ {(p=1,c=8),(2,8),(4,8),(8,4),(8,2),(8,1)}                 = 6
#   ordering  ∈ {XCHG_ACQREL, MAX_ACQUIRE, CAS_ACQREL, FADD_RELEASE}      = 4
#   (wgp,block) ∈ {(304,256),(304,1024),(1216,256),(1216,1024)}           = 4
#   buffer    ∈ {2 MiB (L2-resident), 32 MiB (HBM)}                       = 2
#   reps      = 25 (+4 warmup)
# Total cells = 6 × 4 × 4 × 2 = 192 cells per src->dst pair
# Pairs: (0->1) only to keep within budget — XCD topology is held fixed (intra-XCD
# producer/consumer effectively, both on rank 0; data lives on rank 1 for cross-GPU
# atomic).  Total rep-rows ≈ 192 × 25 = 4,800.
#
# `prod_waves_per_CU` = num_warps for producer kernel (1 WG per CU at wgp=304;
# 4 WG per CU at wgp=1216 — the per-CU wave count is num_warps × WGs/CU).
# Reported as `prod_num_warps` raw + derived `prod_waves_per_CU`.
#
# The PRODUCER is the kernel that issues the ordering primitive (cost-defining
# RMW with the named semantic). The CONSUMER does a paired ATOMIC_MAX read-back
# in the matching semantic to force the visibility chain.

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import triton.language as tl

import iris

torch.manual_seed(2388)


# ---------------------------------------------------------------------------
# Ordering primitives (4 classes spanning the K-2317 cost spectrum)
# Producer-side issue op + sem; consumer-side observe is atomic_max with the
# matching sem (acquire vs release vs acq_rel vs relaxed paired).
# ---------------------------------------------------------------------------

ORDERING_CLASSES = [
    # (name,             issue_op, issue_sem, observe_op, observe_sem)
    ("XCHG_ACQREL",     "xchg",   "acq_rel",  "max",      "acq_rel"),
    ("MAX_ACQUIRE",     "max",    "acquire",  "max",      "acquire"),
    ("CAS_ACQREL",      "cas",    "acq_rel",  "max",      "acq_rel"),
    ("FADD_RELEASE",    "fadd",   "release",  "max",      "acquire"),
]

IMBALANCES = [
    # (prod_num_warps, cons_num_warps)
    (1, 8),
    (2, 8),
    (4, 8),
    (8, 4),
    (8, 2),
    (8, 1),
]

WGP_BLOCK = [
    # (n_workgroups, block_size)
    (304,   256),
    (304,   1024),
    (1216,  256),
    (1216,  1024),
]

BUFFERS = [
    2 * 1024 * 1024,    # 2 MiB — L2-resident on MI300X (L2 = 4 MiB per XCD)
    32 * 1024 * 1024,   # 32 MiB — overflows L2, hits HBM
]

DTYPE     = torch.int32
N_WARMUP  = 4
N_REPEAT  = 25
SRC_RANK  = 0
DST_RANK  = 1
CUS_PER_GPU = 304


# ---------------------------------------------------------------------------
# Kernel codegen — per (issue_op, issue_sem) and per (observe_op, observe_sem)
# Triton requires constexpr tags so we generate a kernel per combo.
# CAS uses tiled inner loop (256-wide) per K-2354/K-2366 lowering bug workaround.
# ---------------------------------------------------------------------------

CAS_INNER = 256
CAS_SENTINEL = -2**31  # buffer is zeroed → CAS fails → no net write but RMW fires


_PROD_TEMPLATES = {
    "xchg": '''
@triton.jit
def kP_xchg_{sem}(buf, n, src_rank: tl.constexpr, dst_rank: tl.constexpr,
                  BLOCK_SIZE: tl.constexpr, heap):
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off = off % n
    iris.atomic_xchg(buf + off, 1, src_rank, dst_rank, heap, sem="{sem}", scope="sys")
''',
    "max": '''
@triton.jit
def kP_max_{sem}(buf, n, src_rank: tl.constexpr, dst_rank: tl.constexpr,
                 BLOCK_SIZE: tl.constexpr, heap):
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off = off % n
    iris.atomic_max(buf + off, 1, src_rank, dst_rank, heap, sem="{sem}", scope="sys")
''',
    "fadd": '''
@triton.jit
def kP_fadd_{sem}(buf, n, src_rank: tl.constexpr, dst_rank: tl.constexpr,
                  BLOCK_SIZE: tl.constexpr, heap):
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off = off % n
    iris.atomic_add(buf + off, 1, src_rank, dst_rank, heap, sem="{sem}", scope="sys")
''',
    "cas": '''
@triton.jit
def kP_cas_{sem}(buf, n, src_rank: tl.constexpr, dst_rank: tl.constexpr,
                 BLOCK_SIZE: tl.constexpr, heap):
    pid = tl.program_id(0)
    cmp = tl.full([{INNER}], {SENT}, dtype=tl.int32)
    new = tl.full([{INNER}], 1, dtype=tl.int32)
    base = pid * BLOCK_SIZE
    for i in tl.static_range(0, BLOCK_SIZE, {INNER}):
        offs = base + i + tl.arange(0, {INNER})
        offs = offs % n
        iris.atomic_cas(buf + offs, cmp, new, src_rank, dst_rank, heap,
                        sem="{sem}", scope="sys")
''',
}

_CONS_TEMPLATE = '''
@triton.jit
def kC_max_{sem}(buf, n, src_rank: tl.constexpr, dst_rank: tl.constexpr,
                 BLOCK_SIZE: tl.constexpr, heap):
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off = off % n
    iris.atomic_max(buf + off, 1, src_rank, dst_rank, heap, sem="{sem}", scope="sys")
'''


def _emit_kernels_module():
    here = Path(__file__).resolve().parent
    gen = here / "_k2388_kernels.py"
    src = [
        "# AUTOGEN — see k2388_pc_imbalance.py::_emit_kernels_module()",
        "import triton, triton.language as tl",
        "import iris",
    ]
    seen_p = set()
    seen_c = set()
    for cls in ORDERING_CLASSES:
        _, issue, isem, observe, osem = cls
        key_p = (issue, isem)
        if key_p not in seen_p:
            seen_p.add(key_p)
            tmpl = _PROD_TEMPLATES[issue]
            if issue == "cas":
                src.append(tmpl.format(sem=isem, INNER=CAS_INNER, SENT=CAS_SENTINEL))
            else:
                src.append(tmpl.format(sem=isem))
        key_c = (observe, osem)
        if key_c not in seen_c:
            seen_c.add(key_c)
            src.append(_CONS_TEMPLATE.format(sem=osem))
    gen.write_text("\n".join(src) + "\n")
    return gen


_GEN_PATH = _emit_kernels_module()
sys.path.insert(0, str(_GEN_PATH.parent))
import _k2388_kernels as _K_MOD  # noqa: E402


def producer_kernel(issue_op: str, sem: str):
    return getattr(_K_MOD, f"kP_{issue_op}_{sem}")


def consumer_kernel(observe_op: str, sem: str):
    return getattr(_K_MOD, f"kC_{observe_op}_{sem}")


def all_cells(smoke: bool = False):
    if smoke:
        # 4 cells: one per ordering class at smallest geometry
        return [(cls, IMBALANCES[2], WGP_BLOCK[0], BUFFERS[0])
                for cls in ORDERING_CLASSES]
    cells = []
    for cls in ORDERING_CLASSES:
        for imb in IMBALANCES:
            for wb in WGP_BLOCK:
                for buf in BUFFERS:
                    cells.append((cls, imb, wb, buf))
    return cells


def run_one_cell(shmem, cls, imb, wb, buffer_bytes, buf_pool):
    """Time the paired producer→consumer kernel sequence per call."""
    _, issue, isem, observe, osem = cls
    p_nw, c_nw = imb
    n_wg, BS = wb
    elem_bytes = torch.tensor([], dtype=DTYPE).element_size()
    n_elements = buffer_bytes // elem_bytes
    buf = buf_pool[buffer_bytes]
    cur_rank = shmem.get_rank()

    kP = producer_kernel(issue, isem)
    kC = consumer_kernel(observe, osem)
    grid = (n_wg,)

    def fn():
        if cur_rank == SRC_RANK:
            kP[grid](buf, n_elements, SRC_RANK, DST_RANK, BS,
                     shmem.get_heap_bases(), num_warps=p_nw)
            kC[grid](buf, n_elements, SRC_RANK, DST_RANK, BS,
                     shmem.get_heap_bases(), num_warps=c_nw)

    def preamble():
        if cur_rank == SRC_RANK:
            buf.fill_(0)

    ok = torch.tensor([1], device="cuda", dtype=torch.int32)
    if cur_rank == SRC_RANK:
        try:
            preamble()
            fn()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"[FAIL-COMPILE] cls={cls[0]} imb={imb} wb={wb} buf={buffer_bytes>>20}MiB :: "
                  f"{type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)
            ok[0] = 0
    dist.broadcast(ok, src=SRC_RANK)
    if int(ok.item()) == 0:
        return float("nan"), []

    try:
        shmem.barrier()
        ms_all = iris.do_bench(fn, barrier_fn=shmem.barrier,
                               preamble_fn=preamble,
                               n_warmup=N_WARMUP, n_repeat=N_REPEAT,
                               return_mode="all")
        ms_mean = float(np.mean(ms_all)) if ms_all is not None else float("nan")
        ms_list = [float(x) for x in ms_all] if ms_all is not None else []
    except Exception as e:
        if cur_rank == SRC_RANK:
            print(f"[FAIL-BENCH] cls={cls[0]} imb={imb} wb={wb} :: "
                  f"{type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)
        ms_mean = float("nan")
        ms_list = []

    return ms_mean, ms_list


def _worker(local_rank: int, world_size: int, init_url: str, args: dict):
    dist.init_process_group(backend="nccl", init_method=init_url,
                            world_size=world_size, rank=local_rank,
                            device_id=torch.device(f"cuda:{local_rank}"))
    torch.cuda.set_device(local_rank)
    shmem = iris.iris(args["heap_size"])
    cur_rank = shmem.get_rank()

    elem_bytes = torch.tensor([], dtype=DTYPE).element_size()
    buf_pool = {}
    for buf_b in BUFFERS:
        buf_pool[buf_b] = shmem.zeros(buf_b // elem_bytes, device="cuda", dtype=DTYPE)
    if cur_rank == 0:
        total_mb = sum(BUFFERS) // (1 << 20)
        print(f"[K-2388] pre-allocated {len(buf_pool)} buffers, total {total_mb} MiB/rank",
              flush=True)

    cells = all_cells(smoke=args["smoke"])
    if args["chunk_total"] > 1:
        i0 = (len(cells) * args["chunk"]) // args["chunk_total"]
        i1 = (len(cells) * (args["chunk"] + 1)) // args["chunk_total"]
        cells = cells[i0:i1]
        if cur_rank == 0:
            print(f"[K-2388] chunk {args['chunk']+1}/{args['chunk_total']}: cells [{i0}:{i1}] "
                  f"({len(cells)} of total)", flush=True)

    if cur_rank == 0:
        print(f"[K-2388] sweeping {len(cells)} cells, world_size={world_size}", flush=True)

    rows = []
    rep_rows = []
    t0 = time.time()
    for i, (cls, imb, wb, bufb) in enumerate(cells):
        ms_mean, ms_list = run_one_cell(shmem, cls, imb, wb, bufb, buf_pool)

        ms_t = torch.tensor([ms_mean], device="cuda", dtype=torch.float64)
        dist.broadcast(ms_t, src=SRC_RANK)
        ms_b = float(ms_t.item())

        if cur_rank == 0:
            cls_name, issue, isem, observe, osem = cls
            p_nw, c_nw = imb
            n_wg, BS = wb
            us_per_call = ms_b * 1e3
            wgs_per_cu = n_wg / CUS_PER_GPU
            row = dict(
                ordering_class=cls_name,
                issue_op=issue, issue_sem=isem,
                observe_op=observe, observe_sem=osem,
                prod_num_warps=p_nw, cons_num_warps=c_nw,
                prod_waves_per_CU=p_nw * wgs_per_cu,
                cons_waves_per_CU=c_nw * wgs_per_cu,
                imbalance_ratio=p_nw / c_nw,
                n_workgroups=n_wg, wgs_per_cu=wgs_per_cu,
                block_size=BS, buffer_bytes=bufb,
                buffer_class="L2" if bufb <= (4 << 20) else "HBM",
                n_elements=bufb // 4, dtype="int32",
                src_rank=SRC_RANK, dst_rank=DST_RANK,
                n_repeat=N_REPEAT, n_warmup=N_WARMUP,
                world_size=world_size,
                ms_mean=ms_b, us_per_call=us_per_call,
                node=os.environ.get("HOSTNAME", "?"),
                ts=time.time(),
            )
            rows.append(row)
            for rep_idx, rep_ms in enumerate(ms_list):
                rep_rows.append(dict(
                    ordering_class=cls_name,
                    prod_num_warps=p_nw, cons_num_warps=c_nw,
                    n_workgroups=n_wg, block_size=BS, buffer_bytes=bufb,
                    rep=rep_idx, ms=rep_ms,
                    node=os.environ.get("HOSTNAME", "?"), ts=time.time(),
                ))
            if i % 16 == 0 or i == len(cells) - 1:
                el = time.time() - t0
                eta = (el / max(i + 1, 1)) * (len(cells) - i - 1)
                print(f"  [{i+1:4d}/{len(cells)}] cls={cls_name:14s} imb=({p_nw},{c_nw}) "
                      f"wgp={n_wg:5d} bs={BS:5d} buf={bufb>>20:2d}MiB "
                      f"us={us_per_call:9.2f} el={el:.0f}s eta={eta:.0f}s", flush=True)
        shmem.barrier()

    if cur_rank == 0:
        df = pd.DataFrame(rows)
        rep_df = pd.DataFrame(rep_rows)
        out_path = Path(args["out"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        df.to_csv(out_path.with_suffix(".csv"), index=False)
        rep_path = out_path.with_name(out_path.stem + "_reps.parquet")
        rep_df.to_parquet(rep_path, index=False)
        rep_df.to_csv(rep_path.with_suffix(".csv"), index=False)
        n_nan = int(df["us_per_call"].isna().sum())
        print(f"[done] {len(df)} cells, {len(rep_df)} reps → {out_path} (NaN cells: {n_nan})",
              flush=True)

    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-ranks", type=int, default=8)
    ap.add_argument("--heap-size", type=int, default=1 << 31)
    ap.add_argument("--out", type=str,
                    default="/home/ryaswann/mc2-workspaces/K-2388/output/k2388_pc_imbalance.parquet")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--chunk-total", type=int, default=1)
    ap.add_argument("--port", type=int, default=29588)
    args = vars(ap.parse_args())

    init_url = f"tcp://127.0.0.1:{args['port']}"
    mp.spawn(fn=_worker, args=(args["num_ranks"], init_url, args),
             nprocs=args["num_ranks"], join=True)


if __name__ == "__main__":
    main()
