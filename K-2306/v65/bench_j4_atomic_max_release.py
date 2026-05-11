#!/usr/bin/env python3
# K-2306: J4 ATOMIC_MAX_RELEASE microbenchmark on MI300X (gfx942)
#
# Third leg of the MAX-family memory-ordering tetrad
# (H4 RELAXED K-2297 -> I4 ACQUIRE K-2301 -> J4 RELEASE [this] -> K4 ACQREL).
#
# Mirrors K-2297 H4 RELAXED methodology exactly, swapping sem='relaxed'->'release'.
# Buffer is zeroed in preamble; val=INT_MIN so max(0, INT_MIN) = 0 (no net write
# but the L2 atomic-MAX unit still executes the compare RMW under release ordering).
#
# Total rows: 3 block_sizes x 2 dtypes x 3 scopes x 64 pairs x 25 reps = 28,800.

import argparse
import csv
import os
import socket
import time
from itertools import product

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import triton.language as tl

import iris


PRIM_ID = "J4"
PRIM_NAME = "ATOMIC_MAX_RELEASE"
SEM = "release"


@triton.jit
def atomic_max_release_kernel(
    buf,
    n,
    src_rank: tl.constexpr,
    dst_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCOPE: tl.constexpr,
    SENTINEL: tl.constexpr,
    heap_bases,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    # Buffer is zeroed in preamble; SENTINEL=INT_MIN so max(0, INT_MIN) = 0
    # (no net write but the L2 atomic-MAX unit must still execute the compare RMW
    # under release ordering, exposing the per-op cost vs H4 RELAXED / I4 ACQUIRE).
    val = tl.full([BLOCK_SIZE], SENTINEL, dtype=buf.type.element_ty)
    iris.atomic_max(buf + offs, val, src_rank, dst_rank, heap_bases, mask=mask, sem="release", scope=SCOPE)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--reps", type=int, default=25)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--num-ranks", type=int, default=8)
    p.add_argument("--heap-size", type=int, default=1 << 31)  # 2 GiB
    p.add_argument("--buffer-bytes", type=int, default=1 << 24)  # 16 MiB
    p.add_argument("--block-sizes", default="256,1024,4096")
    p.add_argument("--dtypes", default="int32,int64")
    p.add_argument("--scopes", default="cta,gpu,sys")
    p.add_argument("--pairs", default="")  # "src:dst,src:dst,..." (empty = full grid)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def cell_iter(args):
    block_sizes = [int(b) for b in args.block_sizes.split(",")]
    dtypes = args.dtypes.split(",")
    scopes = args.scopes.split(",")
    if args.pairs:
        pairs = [tuple(int(x) for x in pr.split(":")) for pr in args.pairs.split(",")]
    else:
        pairs = [(s, d) for s in range(args.num_ranks) for d in range(args.num_ranks)]
    if args.smoke:
        block_sizes = block_sizes[:1]
        dtypes = dtypes[:1]
        scopes = scopes[:1]
        pairs = pairs[:4]
    for bs, dt, sc, (s, d) in product(block_sizes, dtypes, scopes, pairs):
        yield bs, dt, sc, s, d


def torch_dtype(name):
    return {"int32": torch.int32, "int64": torch.int64}[name]


def sentinel_for(dt_name):
    # MAX sentinel: INT_MIN so max(0, INT_MIN) = 0 (no net write, atomic still executes)
    return {"int32": -(2**31), "int64": -(2**63)}[dt_name]


def _worker(local_rank, world_size, init_url, args, hostname, gpu_arch_str, run_id):
    backend = "nccl"
    dist.init_process_group(
        backend=backend, init_method=init_url, world_size=world_size, rank=local_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    shmem = iris.iris(args.heap_size)
    rank = shmem.get_rank()
    nranks = shmem.get_num_ranks()
    if rank == 0:
        print(f"[bench] world={nranks} arch={gpu_arch_str} host={hostname}", flush=True)

    out_dir = os.path.dirname(args.out) or "."
    base, ext = os.path.splitext(os.path.basename(args.out))
    per_rank_path = os.path.join(out_dir, f"{base}.rank{rank}{ext}")
    os.makedirs(out_dir, exist_ok=True)
    f = open(per_rank_path, "w", newline="")
    wr = csv.writer(f)
    wr.writerow([
        "run_id", "primitive_id", "primitive_name", "sem", "scope",
        "block_size", "dtype", "src_rank", "dst_rank",
        "buffer_bytes", "n_elements", "rep_idx", "time_ms",
        "bandwidth_gibps", "world_size", "gpu_arch", "hostname", "ts_unix",
    ])
    f.flush()

    # Pre-allocate one symmetric buffer per dtype up front and reuse across cells.
    pre_bufs = {}
    for dt_name in set(args.dtypes.split(",")):
        dt = torch_dtype(dt_name)
        elem_size = torch.tensor([], dtype=dt).element_size()
        n_elem = args.buffer_bytes // elem_size
        pre_bufs[dt_name] = shmem.zeros(n_elem, dtype=dt, device="cuda")
    if rank == 0:
        print(f"[bench] preallocated buffers: {list(pre_bufs.keys())}", flush=True)

    for cell_idx, (bs, dt_name, sc, s, d) in enumerate(cell_iter(args)):
        dt = torch_dtype(dt_name)
        elem_size = torch.tensor([], dtype=dt).element_size()
        n_elem = args.buffer_bytes // elem_size
        buf = pre_bufs[dt_name]
        sentinel = sentinel_for(dt_name)
        grid = lambda meta: (triton.cdiv(n_elem, meta["BLOCK_SIZE"]),)

        def run_once():
            if rank == s:
                atomic_max_release_kernel[grid](
                    buf, n_elem, s, d, bs, sc, sentinel, shmem.get_heap_bases(),
                )

        def preamble():
            buf.zero_()  # buf=0, val=INT_MIN -> max stays 0 (no net write, atomic still executes)

        # warmup once before do_bench
        run_once()
        shmem.barrier()

        try:
            times_ms = iris.do_bench(
                run_once,
                barrier_fn=shmem.barrier,
                preamble_fn=preamble,
                n_warmup=args.warmup,
                n_repeat=args.reps,
                return_mode="all",
            )
        except Exception as e:
            if rank == 0:
                print(f"[bench] FAIL cell ({bs},{dt_name},{sc},{s}->{d}): {e}", flush=True)
            shmem.barrier()
            continue

        total_bytes = n_elem * elem_size
        # Only the source rank actually launches the kernel — only it records.
        if rank == s:
            ts = time.time()
            for i, t_ms in enumerate(times_ms):
                if t_ms <= 0:
                    bw = 0.0
                else:
                    bw = (total_bytes * 2) / (t_ms * 1e-3) / 2**30  # RMW => 2x
                wr.writerow([
                    run_id, PRIM_ID, PRIM_NAME, SEM, sc,
                    bs, dt_name, s, d,
                    args.buffer_bytes, n_elem, i, f"{t_ms:.6f}",
                    f"{bw:.4f}", nranks, gpu_arch_str, hostname, f"{ts:.3f}",
                ])
            f.flush()
            if rank == 0 and cell_idx % 25 == 0:
                print(f"[bench] cell {cell_idx} ({bs},{dt_name},{sc},{s}->{d}) mean_ms={np.mean(times_ms):.4f}", flush=True)

        shmem.barrier()

    f.close()
    if rank == 0:
        print(f"[bench] DONE rank0 wrote {per_rank_path}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


def main():
    args = parse()
    hostname = socket.gethostname()
    gpu_arch = torch.cuda.get_device_properties(0).gcnArchName
    run_id = f"K-2306-{int(time.time())}-{hostname}"
    init_url = "tcp://127.0.0.1:29603"
    mp.spawn(
        fn=_worker,
        args=(args.num_ranks, init_url, args, hostname, gpu_arch, run_id),
        nprocs=args.num_ranks,
        join=True,
    )


if __name__ == "__main__":
    main()
