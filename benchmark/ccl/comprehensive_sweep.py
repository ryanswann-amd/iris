#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Comprehensive iris-ccl vs RCCL sweep across the 1 KiB ... 1 GiB message range.

Measures wall-clock latency (mean of n_repeat samples) for the four core
collectives on a single MI300X node:

    iris.ccl.{all_reduce, all_gather, reduce_scatter, all_to_all}

vs. the equivalent RCCL primitives exposed through ``torch.distributed``.
The script is the single source of truth for the static defaults table baked
into ``iris/ccl/config.py`` — it can be invoked in two modes:

* **Validation** (default): run iris.ccl with ``Config()`` defaults and
  ``torch.distributed`` and report iris-vs-RCCL ratios. Used to demonstrate
  the "within 10 %" claim across the entire size range.

* **Tuning** (``--mode=tune``): sweep candidate kernel configs per
  (collective, message-size bucket, dtype) and emit a JSON record of the
  fastest config in each cell. The output of this mode is hand-converted
  into the ``_DEFAULTS_TABLE`` constant in ``iris/ccl/config.py``.

Both modes share the same launcher/timer infrastructure to keep timing
consistent across runs.

Usage
-----

Single-node, 8-rank run::

    torchrun --nproc_per_node=8 benchmark/ccl/comprehensive_sweep.py \\
        --benchmark_rccl --benchmark_iris \\
        --output_csv output/sweep.csv \\
        --output_plots_dir output/plots

Tuning sweep (writes one JSON per (collective, dtype))::

    torchrun --nproc_per_node=8 benchmark/ccl/comprehensive_sweep.py \\
        --mode tune --output_dir output/tune

The script intentionally uses ``torchrun`` directly instead of
``iris.bench.main`` so that it is easy to rerun, parse, and plot offline.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist

import iris
from iris.ccl import Config
from iris.ccl.config import _DEFAULT_ARCH, default_config

logger = logging.getLogger("iris.ccl.sweep")


# ── Utilities ─────────────────────────────────────────────────────────────


_DTYPE_ALIASES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def _parse_dtype(spec: str) -> torch.dtype:
    """Map a CLI dtype string to a torch.dtype."""
    if spec.lower() not in _DTYPE_ALIASES:
        raise ValueError(f"Unknown dtype {spec!r}; expected one of {sorted(_DTYPE_ALIASES)}.")
    return _DTYPE_ALIASES[spec.lower()]


def _dtype_str(dtype: torch.dtype) -> str:
    """Map a torch.dtype to a short CLI-friendly string."""
    return {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(dtype, str(dtype))


def _power_of_two_bytes(start_kib: int = 1, stop_gib: int = 1) -> list[int]:
    """Return powers-of-two byte counts from ``start_kib`` KiB to ``stop_gib`` GiB inclusive."""
    start_bytes = start_kib * 1024
    stop_bytes = stop_gib * 1024 * 1024 * 1024
    sizes = []
    n = start_bytes
    while n <= stop_bytes:
        sizes.append(n)
        n *= 2
    return sizes


def _shape_for_message(total_bytes: int, dtype: torch.dtype, world_size: int, collective: str) -> tuple[int, int]:
    """Pick a (M, N) tensor shape that yields the requested *per-rank* input bytes.

    * For ``all_reduce``, ``all_gather``, ``reduce_scatter`` — input is
      ``M x N`` and total bytes are ``M * N * elem_size``.
    * For ``all_to_all`` — input is ``M x (N * world_size)`` so total bytes
      are ``M * N * world_size * elem_size``.

    The shape is chosen so N is a power of two between 16 and 8192 and the
    full tensor is exactly ``total_bytes`` (rounded up to the smallest
    power-of-two M that satisfies the constraint).
    """
    elem_size = torch.tensor([], dtype=dtype).element_size()
    total_elems = total_bytes // elem_size
    if total_elems <= 0:
        raise ValueError(f"total_bytes={total_bytes} too small for dtype={dtype}")

    if collective == "all_to_all":
        per_rank_elems = total_elems // world_size
        if per_rank_elems <= 0:
            raise ValueError(
                f"total_bytes={total_bytes} below the per-rank floor for all_to_all with world_size={world_size}"
            )
    else:
        per_rank_elems = total_elems

    # Pick N (per-rank) — clamp to [16, 8192] and a power of two so kernel
    # block sizes and stride alignments are always satisfied.
    n_target = min(8192, max(16, per_rank_elems))
    n = 1 << int(math.log2(n_target))
    if n > per_rank_elems:
        n = max(16, 1 << int(math.log2(max(16, per_rank_elems))))

    m = max(1, per_rank_elems // n)
    # Guard: ensure exact byte count (drop to nearest power-of-two M that fits)
    if m * n * elem_size > total_bytes:
        m = max(1, total_bytes // (n * elem_size))

    return int(m), int(n)


def _input_output_shapes(
    collective: str,
    m: int,
    n: int,
    world_size: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ``(input_shape, output_shape)`` for the given collective."""
    if collective == "all_reduce" or collective == "reduce_scatter":
        return (m, n), (m, n)
    if collective == "all_gather":
        return (m, n), (world_size * m, n)
    if collective == "all_to_all":
        # The harness's per-rank tensor of size ``m * n`` is treated as one
        # ``world_size``-way concatenation: ``(m, n // world_size)`` per peer.
        if n % world_size:
            raise ValueError(
                f"all_to_all requires N={n} to be divisible by world_size={world_size}; adjust the sweep grid."
            )
        per_peer_n = n // world_size
        return (m, n), (m, n)  # both (m, world_size * per_peer_n) where total N == n
    raise ValueError(f"Unknown collective {collective!r}")


def _collective_bus_bytes(collective: str, m: int, n: int, world_size: int, dtype: torch.dtype) -> int:
    """Bytes moved across the bus per call (used to compute bus-bandwidth GB/s).

    Numbers track NCCL's documented bus-bandwidth conventions so iris and
    RCCL rows are directly comparable.
    """
    elem = torch.tensor([], dtype=dtype).element_size()
    payload = m * n * elem
    if collective == "all_reduce":
        return int(payload * 2 * (world_size - 1) / world_size)
    if collective == "all_gather":
        return int(payload * (world_size - 1))
    if collective == "reduce_scatter":
        return int(payload * (world_size - 1))
    if collective == "all_to_all":
        # Each rank sends (W-1)/W of its input; multiply by W ranks gives (W-1) payloads
        return int(payload * (world_size - 1) / world_size)
    raise ValueError(f"Unknown collective {collective!r}")


# ── Iris launchers ────────────────────────────────────────────────────────


@dataclass
class _IrisPool:
    """Pre-allocated symmetric buffers reused across all sweep cells.

    The bump allocator inside iris's symmetric heap does not release memory,
    so we allocate once at the largest size we will ever need and reshape
    views per cell. Buffers are sized in bytes (1D ``int8``) so we can re-view
    them as either fp16 or bf16 — both have a 2-byte element size.
    """

    ctx: object
    inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    outputs: dict[str, torch.Tensor] = field(default_factory=dict)


def _build_iris_pool(ctx, world_size: int, max_bytes: int) -> _IrisPool:
    """Pre-allocate the per-collective symmetric buffers.

    Sizes are derived from the largest message in the sweep — currently 1 GiB
    per-rank (so 8 GiB output for all-gather at world_size=8). The buffers are
    fp16 because that is the largest dtype in the sweep grid; bf16 cells reuse
    the same byte-backed memory via ``view(torch.bfloat16)``.
    """
    pool = _IrisPool(ctx=ctx)
    max_elems_fp16 = max_bytes // 2  # fp16 element size

    # all_reduce: input + output, both shape (M, N)
    pool.inputs["all_reduce"] = ctx.zeros((max_elems_fp16,), dtype=torch.float16)
    pool.outputs["all_reduce"] = ctx.zeros((max_elems_fp16,), dtype=torch.float16)

    # all_gather: output is world_size × input
    pool.inputs["all_gather"] = ctx.zeros((max_elems_fp16,), dtype=torch.float16)
    pool.outputs["all_gather"] = ctx.zeros((max_elems_fp16 * world_size,), dtype=torch.float16)

    # reduce_scatter: input shape (M, N), output shape (M, N) (matches iris API).
    pool.inputs["reduce_scatter"] = ctx.zeros((max_elems_fp16,), dtype=torch.float16)
    pool.outputs["reduce_scatter"] = ctx.zeros((max_elems_fp16,), dtype=torch.float16)

    # all_to_all: input/output shape (M, N) where N is the per-peer width × world_size.
    pool.inputs["all_to_all"] = ctx.zeros((max_elems_fp16,), dtype=torch.float16)
    pool.outputs["all_to_all"] = ctx.zeros((max_elems_fp16,), dtype=torch.float16)

    return pool


@dataclass
class _IrisHandles:
    ctx: object
    inp: torch.Tensor
    out: torch.Tensor


def _view_pool_tensor(pool_tensor: torch.Tensor, shape: tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
    """Re-view a fp16 pool tensor as ``dtype`` and reshape to ``shape``."""
    needed = shape[0] * shape[1]
    if dtype == torch.float16:
        return pool_tensor[:needed].view(shape)
    if dtype == torch.bfloat16:
        # Same element size as fp16 — view-cast then reshape.
        return pool_tensor.view(torch.bfloat16)[:needed].view(shape)
    raise ValueError(f"Unsupported dtype {dtype}")


def _build_iris_handles(
    pool: _IrisPool, collective: str, m: int, n: int, world_size: int, dtype: torch.dtype
) -> _IrisHandles:
    in_shape, out_shape = _input_output_shapes(collective, m, n, world_size)
    inp = _view_pool_tensor(pool.inputs[collective], in_shape, dtype)
    out = _view_pool_tensor(pool.outputs[collective], out_shape, dtype)
    inp.fill_(float(pool.ctx.get_rank() + 1))
    return _IrisHandles(ctx=pool.ctx, inp=inp, out=out)


def _make_iris_runner(
    pool: _IrisPool,
    collective: str,
    m: int,
    n: int,
    world_size: int,
    dtype: torch.dtype,
    config: Config | None,
) -> tuple[Callable[[], None], Callable[[], None]]:
    """Return ``(run, preamble)`` callables for iris timing."""
    ctx = pool.ctx
    handles = _build_iris_handles(pool, collective, m, n, world_size, dtype)

    if collective == "all_reduce":
        cfg = config if config is not None else default_config("all_reduce", m * n * handles.inp.element_size())
        workspace = ctx.ccl.all_reduce_preamble(handles.out, handles.inp, config=cfg)

        def run() -> None:
            ctx.ccl.all_reduce(handles.out, handles.inp, config=cfg, workspace=workspace)

        def preamble() -> None:
            handles.out.zero_()
            ctx.ccl.all_reduce_preamble(handles.out, handles.inp, config=cfg, workspace=workspace)

    elif collective == "all_gather":
        cfg = config if config is not None else default_config("all_gather", m * n * handles.inp.element_size())

        def run() -> None:
            ctx.ccl.all_gather(handles.out, handles.inp, config=cfg)

        def preamble() -> None:
            handles.out.zero_()

    elif collective == "reduce_scatter":
        cfg = config if config is not None else default_config("reduce_scatter", m * n * handles.inp.element_size())

        def run() -> None:
            ctx.ccl.reduce_scatter(handles.out, handles.inp, config=cfg)

        def preamble() -> None:
            handles.out.zero_()

    elif collective == "all_to_all":
        cfg = config if config is not None else default_config("all_to_all", m * n * handles.inp.element_size())

        def run() -> None:
            ctx.ccl.all_to_all(handles.out, handles.inp, config=cfg)

        def preamble() -> None:
            handles.out.zero_()

    else:
        raise ValueError(f"Unknown collective {collective!r}")

    return run, preamble


# ── RCCL launchers ────────────────────────────────────────────────────────


@dataclass
class _RcclPool:
    """Pre-allocated CUDA tensors for RCCL, mirroring the iris pool sizes."""

    inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    outputs: dict[str, torch.Tensor] = field(default_factory=dict)


def _build_rccl_pool(world_size: int, max_bytes: int) -> _RcclPool:
    """Pre-allocate the per-collective torch tensors for the RCCL baseline."""
    pool = _RcclPool()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    max_elems_fp16 = max_bytes // 2

    pool.inputs["all_reduce"] = torch.zeros(max_elems_fp16, dtype=torch.float16, device=device)
    pool.outputs["all_reduce"] = pool.inputs["all_reduce"]  # in-place semantics

    pool.inputs["all_gather"] = torch.zeros(max_elems_fp16, dtype=torch.float16, device=device)
    pool.outputs["all_gather"] = torch.zeros(max_elems_fp16 * world_size, dtype=torch.float16, device=device)

    pool.inputs["reduce_scatter"] = torch.zeros(max_elems_fp16, dtype=torch.float16, device=device)
    pool.outputs["reduce_scatter"] = torch.zeros(max_elems_fp16, dtype=torch.float16, device=device)

    pool.inputs["all_to_all"] = torch.zeros(max_elems_fp16, dtype=torch.float16, device=device)
    pool.outputs["all_to_all"] = torch.zeros(max_elems_fp16, dtype=torch.float16, device=device)

    return pool


def _make_rccl_runner(
    pool: _RcclPool,
    collective: str,
    m: int,
    n: int,
    world_size: int,
    dtype: torch.dtype,
) -> tuple[Callable[[], None], Callable[[], None]]:
    """Return ``(run, preamble)`` callables for the RCCL baseline."""
    rank = dist.get_rank()

    if collective == "all_reduce":
        tensor = _view_pool_tensor(pool.inputs["all_reduce"], (m, n), dtype)
        tensor.fill_(float(rank + 1))

        def run() -> None:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        def preamble() -> None:
            tensor.fill_(float(rank + 1))

    elif collective == "all_gather":
        in_t = _view_pool_tensor(pool.inputs["all_gather"], (m, n), dtype)
        out_t = _view_pool_tensor(pool.outputs["all_gather"], (world_size * m, n), dtype)
        in_t.fill_(float(rank + 1))
        out_slices = [out_t[i * m : (i + 1) * m] for i in range(world_size)]

        def run() -> None:
            dist.all_gather(out_slices, in_t)

        def preamble() -> None:
            out_t.zero_()

    elif collective == "reduce_scatter":
        if m % world_size:
            raise ValueError(
                f"reduce_scatter requires M={m} to be divisible by world_size={world_size}; adjust the sweep grid."
            )
        in_t = _view_pool_tensor(pool.inputs["reduce_scatter"], (m, n), dtype)
        out_t = _view_pool_tensor(pool.outputs["reduce_scatter"], (m // world_size, n), dtype)
        in_t.fill_(float(rank + 1))
        in_slices = [in_t[i * (m // world_size) : (i + 1) * (m // world_size)] for i in range(world_size)]

        def run() -> None:
            dist.reduce_scatter(out_t, in_slices, op=dist.ReduceOp.SUM)

        def preamble() -> None:
            out_t.zero_()

    elif collective == "all_to_all":
        per = n // world_size
        if per * world_size != n:
            raise ValueError(f"all_to_all requires N={n} to be divisible by world_size={world_size}")
        in_t = _view_pool_tensor(pool.inputs["all_to_all"], (m, n), dtype)
        out_t = _view_pool_tensor(pool.outputs["all_to_all"], (m, n), dtype)
        in_t.fill_(float(rank + 1))
        # NCCL all_to_all_single uses a single contiguous tensor for both sides;
        # fall back to dist.all_to_all with per-chunk views to match the iris API.
        in_chunks = [in_t[:, i * per : (i + 1) * per].contiguous() for i in range(world_size)]
        out_chunks = [out_t[:, i * per : (i + 1) * per].contiguous() for i in range(world_size)]

        def run() -> None:
            dist.all_to_all(out_chunks, in_chunks)

        def preamble() -> None:
            out_t.zero_()
            for i, c in enumerate(in_chunks):
                c.fill_(float(rank * 1000 + i))

    else:
        raise ValueError(f"Unknown collective {collective!r}")

    return run, preamble


# ── Timing core ────────────────────────────────────────────────────────────


def _timed_event_loop(
    run: Callable[[], None], preamble: Callable[[], None], n_warmup: int, n_repeat: int
) -> list[float]:
    """Time ``run`` ``n_repeat`` times after ``n_warmup`` warmups, returning per-call ms.

    Uses the same barrier+cache-clear discipline as ``iris.do_bench`` so
    iris and RCCL rows are directly comparable.
    """
    cache_bytes = 256 * 1024 * 1024
    cache = torch.empty(cache_bytes, dtype=torch.int8, device=f"cuda:{torch.cuda.current_device()}")

    def _clear_cache() -> None:
        cache.zero_()

    def _barrier() -> None:
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.synchronize()

    _barrier()
    preamble()
    run()
    _barrier()

    for _ in range(n_warmup):
        _barrier()
        preamble()
        _clear_cache()
        _barrier()
        run()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]

    for i in range(n_repeat):
        _barrier()
        preamble()
        _clear_cache()
        _barrier()
        starts[i].record()
        run()
        ends[i].record()

    _barrier()

    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


# ── Validation sweep ───────────────────────────────────────────────────────


@dataclass
class _Row:
    collective: str
    impl: str
    dtype: str
    total_bytes: int
    M: int
    N: int
    world_size: int
    mean_ms: float
    min_ms: float
    median_ms: float
    bus_gbps: float


def _validation_grid(
    world_size: int, collectives: list[str], dtypes: list[torch.dtype]
) -> list[tuple[int, torch.dtype, str]]:
    sizes = _power_of_two_bytes(start_kib=1, stop_gib=1)
    cells: list[tuple[int, torch.dtype, str]] = []
    for total_bytes, dtype in itertools.product(sizes, dtypes):
        for collective in collectives:
            cells.append((total_bytes, dtype, collective))
    return cells


def _run_validation(args: argparse.Namespace, ctx) -> list[_Row]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    collectives = [c.strip() for c in args.collectives.split(",") if c.strip()]
    dtypes = [_parse_dtype(s) for s in args.dtypes.split(",")] if args.dtypes else [torch.float16, torch.bfloat16]
    cells = _validation_grid(world_size, collectives, dtypes)

    pool = _build_iris_pool(ctx, world_size, max_bytes=1 << 30)
    rccl_pool = _build_rccl_pool(world_size, max_bytes=1 << 30) if "rccl" in args.impls else None

    rows: list[_Row] = []
    for total_bytes, dtype, collective in cells:
        try:
            m, n = _shape_for_message(total_bytes, dtype, world_size, collective)
        except ValueError as exc:
            if rank == 0:
                logger.warning("Skipping cell %s/%s/%d B: %s", collective, _dtype_str(dtype), total_bytes, exc)
            continue

        # Skip cells that violate kernel/algorithm shape constraints.
        if collective == "reduce_scatter" and m % world_size:
            # RCCL requires M divisible by world_size for reduce_scatter; iris doesn't,
            # but for an apples-to-apples comparison we skip cells RCCL can't run.
            continue
        if collective == "all_to_all" and (n % world_size):
            continue

        for impl in args.impls:
            try:
                if impl == "iris":
                    run, preamble = _make_iris_runner(pool, collective, m, n, world_size, dtype, config=None)
                elif impl == "rccl":
                    run, preamble = _make_rccl_runner(rccl_pool, collective, m, n, world_size, dtype)
                else:
                    raise ValueError(f"Unknown impl {impl!r}")
            except ValueError as exc:
                if rank == 0:
                    logger.warning(
                        "Skipping cell %s/%s/%s/%d B: %s",
                        impl,
                        collective,
                        _dtype_str(dtype),
                        total_bytes,
                        exc,
                    )
                continue

            times = _timed_event_loop(run, preamble, args.n_warmup, args.n_repeat)
            del run, preamble
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            mean = statistics.mean(times)
            mn = min(times)
            med = statistics.median(times)
            bus_bytes = _collective_bus_bytes(collective, m, n, world_size, dtype)
            bw = (bus_bytes / 1e9) / (mean / 1e3) if mean > 0 else 0.0
            rows.append(
                _Row(
                    collective=collective,
                    impl=impl,
                    dtype=_dtype_str(dtype),
                    total_bytes=total_bytes,
                    M=m,
                    N=n,
                    world_size=world_size,
                    mean_ms=mean,
                    min_ms=mn,
                    median_ms=med,
                    bus_gbps=bw,
                )
            )
            if rank == 0:
                logger.info(
                    "%-15s %-4s %-12d %-6d %-6d %-4s mean=%.4f ms bus=%.2f GB/s",
                    collective,
                    impl,
                    total_bytes,
                    m,
                    n,
                    _dtype_str(dtype),
                    mean,
                    bw,
                )
    return rows


# ── Tuning sweep ───────────────────────────────────────────────────────────


def _candidate_configs(collective: str, m: int, n: int, world_size: int) -> list[Config]:
    """Build a small grid of candidate Configs for the tuner.

    Restricted to options that actually move the needle on MI300X based on
    prior research (S-001/S-003): comm_sms (occupancy), block_size_n
    (vectorization), variant (algorithm), and num_warps (latency hiding).
    """
    candidates: list[Config] = []

    comm_sms_choices = [32, 64, 96, 128, 192, 224, 256, 304]
    block_n_choices = [16, 32, 64, 128, 256, 512]
    block_m_choices = [8, 16, 32, 64]
    num_warps_choices = [4, 8]

    if collective == "all_reduce":
        for variant in ("two_shot", "atomic", "one_shot"):
            for comm_sms in comm_sms_choices:
                for block_m in block_m_choices:
                    for block_n in block_n_choices:
                        if block_n > n:
                            continue
                        if block_m > m and m > 1:
                            continue
                        if variant == "two_shot" and (block_n % world_size):
                            # Two-shot stripes columns across ranks; needs N divisible by W.
                            continue
                        for nw in num_warps_choices:
                            try:
                                cfg = Config(
                                    block_size_m=max(8, min(block_m, max(1, m))),
                                    block_size_n=block_n,
                                    comm_sms=comm_sms,
                                    all_reduce_variant=variant,
                                    all_reduce_distribution=1,
                                    num_warps=nw,
                                )
                            except ValueError:
                                continue
                            candidates.append(cfg)

    elif collective == "all_gather":
        for variant in ("persistent", "partitioned"):
            for comm_sms in comm_sms_choices:
                if variant == "partitioned" and comm_sms % world_size:
                    continue
                for block_m in block_m_choices:
                    for block_n in block_n_choices:
                        if block_n > n:
                            continue
                        for nw in num_warps_choices:
                            try:
                                cfg = Config(
                                    block_size_m=max(8, min(block_m, max(1, m))),
                                    block_size_n=block_n,
                                    comm_sms=comm_sms,
                                    all_gather_variant=variant,
                                    num_warps=nw,
                                )
                            except ValueError:
                                continue
                            candidates.append(cfg)

    elif collective == "reduce_scatter":
        for comm_sms in comm_sms_choices:
            for block_m in block_m_choices:
                for block_n in block_n_choices:
                    if block_n > n:
                        continue
                    for nw in num_warps_choices:
                        try:
                            cfg = Config(
                                block_size_m=max(8, min(block_m, max(1, m))),
                                block_size_n=block_n,
                                comm_sms=comm_sms,
                                all_reduce_distribution=1,
                                num_warps=nw,
                            )
                        except ValueError:
                            continue
                        candidates.append(cfg)

    elif collective == "all_to_all":
        for comm_sms in comm_sms_choices:
            for block_m in block_m_choices:
                for block_n in block_n_choices:
                    if block_n > n:
                        continue
                    for nw in num_warps_choices:
                        try:
                            cfg = Config(
                                block_size_m=max(8, min(block_m, max(1, m))),
                                block_size_n=block_n,
                                comm_sms=comm_sms,
                                num_warps=nw,
                            )
                        except ValueError:
                            continue
                        candidates.append(cfg)

    return candidates


def _run_tuning(args: argparse.Namespace, ctx) -> dict:
    """Sweep candidate configs per (collective, dtype, total_bytes) and return best."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    pool = _build_iris_pool(ctx, world_size, max_bytes=1 << 30)

    sizes = _power_of_two_bytes(start_kib=1, stop_gib=1)
    if args.tune_sizes:
        sizes = [int(s) for s in args.tune_sizes.split(",")]

    dtypes = (
        [_parse_dtype(s) for s in args.tune_dtypes.split(",")] if args.tune_dtypes else [torch.float16, torch.bfloat16]
    )
    collectives = args.collectives.split(",")

    best: dict = {"arch": _DEFAULT_ARCH, "world_size": world_size, "entries": []}

    for collective in collectives:
        for dtype in dtypes:
            for total_bytes in sizes:
                try:
                    m, n = _shape_for_message(total_bytes, dtype, world_size, collective)
                except ValueError:
                    continue
                if collective == "reduce_scatter" and m % world_size:
                    continue
                if collective == "all_to_all" and (n % world_size):
                    continue

                cands = _candidate_configs(collective, m, n, world_size)
                # Cap candidates per cell to keep the sweep tractable.
                if args.max_candidates_per_cell and len(cands) > args.max_candidates_per_cell:
                    cands = cands[:: max(1, len(cands) // args.max_candidates_per_cell)]
                    cands = cands[: args.max_candidates_per_cell]

                cell_best: tuple[float, Config] | None = None
                for cfg in cands:
                    try:
                        run, preamble = _make_iris_runner(pool, collective, m, n, world_size, dtype, config=cfg)
                    except (ValueError, RuntimeError):
                        continue
                    try:
                        times = _timed_event_loop(run, preamble, args.tune_warmup, args.tune_repeat)
                    except Exception:
                        continue
                    median = statistics.median(times)
                    if cell_best is None or median < cell_best[0]:
                        cell_best = (median, cfg)

                if cell_best is None:
                    continue
                t, cfg = cell_best
                best["entries"].append(
                    {
                        "collective": collective,
                        "dtype": _dtype_str(dtype),
                        "total_bytes": total_bytes,
                        "M": m,
                        "N": n,
                        "median_ms": t,
                        "config": _config_to_dict(cfg, collective),
                    }
                )
                if rank == 0:
                    logger.info(
                        "TUNE %-15s %-4s %-12d -> %.4f ms cfg=%s",
                        collective,
                        _dtype_str(dtype),
                        total_bytes,
                        t,
                        _config_to_dict(cfg, collective),
                    )
    return best


def _config_to_dict(cfg: Config, collective: str) -> dict:
    d = {
        "block_size_m": cfg.block_size_m,
        "block_size_n": cfg.block_size_n,
        "comm_sms": cfg.comm_sms,
        "swizzle_size": cfg.swizzle_size,
        "num_warps": cfg.num_warps,
    }
    if collective == "all_reduce":
        d["variant"] = cfg.all_reduce_variant
        d["distribution"] = cfg.all_reduce_distribution
        d["num_rings"] = cfg.all_reduce_num_rings
    elif collective == "all_gather":
        d["variant"] = cfg.all_gather_variant
    elif collective == "reduce_scatter":
        d["variant"] = cfg.reduce_scatter_variant
        d["distribution"] = cfg.all_reduce_distribution
    return d


# ── Output ────────────────────────────────────────────────────────────────


def _write_csv(rows: list[_Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "collective",
                "impl",
                "dtype",
                "total_bytes",
                "M",
                "N",
                "world_size",
                "mean_ms",
                "min_ms",
                "median_ms",
                "bus_gbps",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.collective,
                    r.impl,
                    r.dtype,
                    r.total_bytes,
                    r.M,
                    r.N,
                    r.world_size,
                    f"{r.mean_ms:.6f}",
                    f"{r.min_ms:.6f}",
                    f"{r.median_ms:.6f}",
                    f"{r.bus_gbps:.4f}",
                ]
            )


def _emit_plots(rows: list[_Row], out_dir: Path) -> None:
    """Write one PNG per (collective, dtype) showing iris vs RCCL bandwidth and ratio."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    by_key: dict[tuple[str, str], dict[str, list[_Row]]] = {}
    for r in rows:
        by_key.setdefault((r.collective, r.dtype), {}).setdefault(r.impl, []).append(r)

    for (collective, dtype), impls in by_key.items():
        fig, (ax_bw, ax_ratio) = plt.subplots(1, 2, figsize=(12, 4.5))
        for impl in ("iris", "rccl"):
            if impl not in impls:
                continue
            rs = sorted(impls[impl], key=lambda r: r.total_bytes)
            xs = [r.total_bytes for r in rs]
            ys = [r.bus_gbps for r in rs]
            ax_bw.plot(xs, ys, marker="o", label=impl)
        ax_bw.set_xscale("log", base=2)
        ax_bw.set_xlabel("Per-rank message size (bytes)")
        ax_bw.set_ylabel("Bus bandwidth (GB/s)")
        ax_bw.set_title(f"{collective} ({dtype}) — bandwidth")
        ax_bw.legend()
        ax_bw.grid(True, which="both", alpha=0.3)

        if "iris" in impls and "rccl" in impls:
            iris_map = {r.total_bytes: r.mean_ms for r in impls["iris"]}
            rccl_map = {r.total_bytes: r.mean_ms for r in impls["rccl"]}
            xs = sorted(set(iris_map) & set(rccl_map))
            ratios = [iris_map[x] / rccl_map[x] for x in xs]
            ax_ratio.plot(xs, ratios, marker="o", color="tab:purple")
            ax_ratio.axhline(1.0, color="black", linestyle="--", alpha=0.5)
            ax_ratio.axhline(1.10, color="tab:red", linestyle=":", label="+10 %")
            ax_ratio.set_xscale("log", base=2)
            ax_ratio.set_xlabel("Per-rank message size (bytes)")
            ax_ratio.set_ylabel("iris time / RCCL time")
            ax_ratio.set_title(f"{collective} ({dtype}) — ratio")
            ax_ratio.legend()
            ax_ratio.grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / f"{collective}_{dtype}.png", dpi=110)
        plt.close(fig)


def _emit_summary(rows: list[_Row], path: Path) -> None:
    """Write a JSON summary of iris-vs-RCCL ratios per (collective, dtype, total_bytes)."""
    iris_map: dict[tuple[str, str, int], _Row] = {}
    rccl_map: dict[tuple[str, str, int], _Row] = {}
    for r in rows:
        key = (r.collective, r.dtype, r.total_bytes)
        if r.impl == "iris":
            iris_map[key] = r
        elif r.impl == "rccl":
            rccl_map[key] = r

    summary: list[dict] = []
    for key in sorted(set(iris_map) & set(rccl_map)):
        ir, rr = iris_map[key], rccl_map[key]
        ratio = ir.mean_ms / rr.mean_ms if rr.mean_ms > 0 else float("inf")
        summary.append(
            {
                "collective": key[0],
                "dtype": key[1],
                "total_bytes": key[2],
                "iris_mean_ms": ir.mean_ms,
                "rccl_mean_ms": rr.mean_ms,
                "iris_over_rccl": ratio,
                "iris_bus_gbps": ir.bus_gbps,
                "rccl_bus_gbps": rr.bus_gbps,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({"rows": summary}, f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=["validate", "tune"], default="validate")
    p.add_argument("--benchmark_iris", action="store_true", help="Include iris.ccl in the validation sweep")
    p.add_argument("--benchmark_rccl", action="store_true", help="Include RCCL in the validation sweep")
    p.add_argument(
        "--collectives",
        default="all_reduce,all_gather,reduce_scatter,all_to_all",
        help="Comma-separated list of collectives to sweep",
    )
    p.add_argument("--n_warmup", type=int, default=10, help="Validation warmup iterations")
    p.add_argument("--n_repeat", type=int, default=20, help="Validation timed iterations")
    p.add_argument("--dtypes", default=None, help="Comma-separated dtypes for validation (default: fp16,bf16)")
    p.add_argument("--tune_warmup", type=int, default=3, help="Tuning warmup iterations")
    p.add_argument("--tune_repeat", type=int, default=5, help="Tuning timed iterations")
    p.add_argument("--tune_sizes", default=None, help="Comma-separated byte sizes to tune (default: 1KiB..1GiB pow2)")
    p.add_argument("--tune_dtypes", default=None, help="Comma-separated dtypes to tune (default: fp16,bf16)")
    p.add_argument(
        "--max_candidates_per_cell",
        type=int,
        default=64,
        help="Max number of configs to try per (collective, dtype, size) tuning cell",
    )
    p.add_argument("--output_csv", default=None, help="CSV output path (validation mode)")
    p.add_argument("--output_summary", default=None, help="JSON summary path (validation mode)")
    p.add_argument("--output_plots_dir", default=None, help="Directory for per-cell PNG plots")
    p.add_argument("--output_dir", default="output/tune", help="Tuning output directory")
    p.add_argument(
        "--heap_size",
        type=int,
        # 32 GiB: 1 GiB input + 8 GiB output (all-gather) + 6 GiB for the other collectives
        # + workspace allocated inside preamble.
        default=32 * (1 << 30),
        help="Iris symmetric heap size in bytes",
    )
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [rank?] %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if not args.benchmark_iris and not args.benchmark_rccl and args.mode == "validate":
        # Default both on for the validation run.
        args.benchmark_iris = True
        args.benchmark_rccl = True
    args.impls = []
    if args.benchmark_iris:
        args.impls.append("iris")
    if args.benchmark_rccl:
        args.impls.append("rccl")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    rank_fmt = logging.Formatter(f"%(asctime)s [rank{rank}] %(levelname)s %(name)s: %(message)s")
    for handler in logging.getLogger().handlers:
        handler.setFormatter(rank_fmt)

    ctx = iris.iris(args.heap_size)
    if rank == 0:
        logger.info("Comprehensive sweep starting: mode=%s impls=%s", args.mode, args.impls)

    t0 = time.time()
    if args.mode == "validate":
        rows = _run_validation(args, ctx)
        if rank == 0:
            csv_path = Path(args.output_csv) if args.output_csv else Path("output/sweep.csv")
            _write_csv(rows, csv_path)
            logger.info("Wrote %d rows to %s", len(rows), csv_path)

            if args.output_summary:
                summary_path = Path(args.output_summary)
            else:
                summary_path = csv_path.with_name(csv_path.stem + "_summary.json")
            _emit_summary(rows, summary_path)
            logger.info("Wrote summary to %s", summary_path)

            if args.output_plots_dir:
                plots_dir = Path(args.output_plots_dir)
                _emit_plots(rows, plots_dir)
                logger.info("Wrote plots to %s", plots_dir)
    else:
        best = _run_tuning(args, ctx)
        if rank == 0:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            out_path = out_dir / f"tune_{ts}.json"
            with out_path.open("w") as f:
                json.dump(best, f, indent=2)
            logger.info("Wrote tuning report to %s", out_path)

    if rank == 0:
        logger.info("Sweep complete in %.1f s", time.time() - t0)
    ctx.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
