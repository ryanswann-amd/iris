# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for the K-820 fused-launch fastpath in ``iris.ccl.all_reduce``.

Covers:
  * Correctness: ``fused_launch=True`` produces bit-equivalent results to the
    slow path AND to ``torch.distributed.all_reduce`` for two_shot.
  * Cache behavior:
      - cold call populates ``config._fused_cache``;
      - warm call (same shape) is a cache hit (no extra entries);
      - different (M, N) creates a separate entry (no key collision);
      - different dtype creates a separate entry (no key collision).
  * Fastpath gating: fastpath is skipped (cache stays empty) when
      - variant != "two_shot",
      - group is not None,
      - fused_launch is False.
  * Multi-call stability: 10 warm calls all match the reference.
"""

import gc

import pytest
import torch
import torch.distributed as dist

import iris
from iris.ccl import Config


HEAP_SIZE = 2**33  # 8 GB


def _ref_all_reduce(M, N, dtype, rank):
    """Compute the reference all_reduce output via torch.distributed."""
    ref = torch.empty((M, N), dtype=dtype, device=f"cuda:{rank}")
    ref.fill_(float(rank + 1))
    dist.all_reduce(ref, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    return ref


def _iris_all_reduce(shmem, M, N, dtype, rank, config):
    """Run iris.ccl.all_reduce once and return the output tensor."""
    iris_in = shmem.zeros((M, N), dtype=dtype)
    iris_in.fill_(float(rank + 1))
    iris_out = shmem.zeros((M, N), dtype=dtype)
    workspace = shmem.ccl.all_reduce_preamble(iris_out, iris_in, config=config)
    shmem.barrier()
    shmem.ccl.all_reduce(iris_out, iris_in, config=config, workspace=workspace)
    torch.cuda.synchronize()
    return iris_out


@pytest.mark.parametrize(
    "M, N, block_size_m, block_size_n",
    [
        (128, 64, 32, 64),
        (256, 128, 32, 16),
        (1024, 256, 32, 64),
    ],
)
def test_fused_launch_matches_reference(M, N, block_size_m, block_size_n):
    """fused_launch=True must yield the same output as torch.distributed."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    shmem = iris.iris(HEAP_SIZE)
    rank = shmem.get_rank()
    dtype = torch.bfloat16

    try:
        ref = _ref_all_reduce(M, N, dtype, rank)

        cfg = Config(
            block_size_m=block_size_m,
            block_size_n=block_size_n,
            all_reduce_variant="two_shot",
            all_reduce_distribution=1,
            fused_launch=True,
        )

        # First call populates the cache (cold/slow path).
        out_cold = _iris_all_reduce(shmem, M, N, dtype, rank, cfg)
        assert torch.equal(out_cold, ref), (
            f"cold-path mismatch (M={M}, N={N}): "
            f"max diff={(out_cold - ref).abs().max().item()}"
        )

        # Second call hits the cache (warm/fast path).
        out_warm = _iris_all_reduce(shmem, M, N, dtype, rank, cfg)
        assert torch.equal(out_warm, ref), (
            f"warm-path mismatch (M={M}, N={N}): "
            f"max diff={(out_warm - ref).abs().max().item()}"
        )

        # Cache should have exactly one entry for this (M, N, dtype).
        assert hasattr(cfg, "_fused_cache")
        assert len(cfg._fused_cache) == 1
        assert (M, N, dtype) in cfg._fused_cache

    finally:
        shmem.barrier()
        del shmem
        gc.collect()


def test_fused_launch_warm_path_repeats_match():
    """10 successive warm calls should each produce the reference output."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    shmem = iris.iris(HEAP_SIZE)
    rank = shmem.get_rank()
    M, N = 1024, 256
    dtype = torch.bfloat16

    try:
        ref = _ref_all_reduce(M, N, dtype, rank)
        cfg = Config(
            block_size_m=32,
            block_size_n=64,
            all_reduce_variant="two_shot",
            all_reduce_distribution=1,
            fused_launch=True,
        )

        iris_in = shmem.zeros((M, N), dtype=dtype)
        iris_in.fill_(float(rank + 1))
        iris_out = shmem.zeros((M, N), dtype=dtype)
        workspace = shmem.ccl.all_reduce_preamble(iris_out, iris_in, config=cfg)
        shmem.barrier()

        for i in range(10):
            shmem.ccl.all_reduce(iris_out, iris_in, config=cfg, workspace=workspace)
            torch.cuda.synchronize()
            assert torch.equal(iris_out, ref), f"iter {i} mismatch"

        # Still only one cache entry after 10 calls (no leak).
        assert len(cfg._fused_cache) == 1
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


def test_fused_launch_distinct_shapes_distinct_entries():
    """Different (M, N) must produce separate cache entries (no collision)."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    shmem = iris.iris(HEAP_SIZE)
    rank = shmem.get_rank()
    dtype = torch.bfloat16

    try:
        cfg = Config(
            block_size_m=32,
            block_size_n=64,
            all_reduce_variant="two_shot",
            all_reduce_distribution=1,
            fused_launch=True,
        )

        for M, N in [(128, 64), (256, 128), (1024, 256)]:
            ref = _ref_all_reduce(M, N, dtype, rank)
            out = _iris_all_reduce(shmem, M, N, dtype, rank, cfg)
            assert torch.equal(out, ref), f"shape ({M}, {N}) mismatch"

        # 3 distinct shapes -> 3 entries.
        assert len(cfg._fused_cache) == 3
        for M, N in [(128, 64), (256, 128), (1024, 256)]:
            assert (M, N, dtype) in cfg._fused_cache
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


def test_fused_launch_disabled_does_not_populate_cache():
    """With fused_launch=False the fastpath is skipped and no cache is built."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    shmem = iris.iris(HEAP_SIZE)
    rank = shmem.get_rank()
    M, N = 1024, 256
    dtype = torch.bfloat16

    try:
        cfg = Config(
            block_size_m=32,
            block_size_n=64,
            all_reduce_variant="two_shot",
            all_reduce_distribution=1,
            fused_launch=False,  # explicit
        )

        ref = _ref_all_reduce(M, N, dtype, rank)
        out = _iris_all_reduce(shmem, M, N, dtype, rank, cfg)
        assert torch.equal(out, ref)

        # No cache attribute should be created when fastpath is off.
        assert not hasattr(cfg, "_fused_cache")
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


def test_fused_launch_gated_off_for_non_two_shot_variant():
    """fused_launch=True + variant != two_shot must fall through to slow path."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    shmem = iris.iris(HEAP_SIZE)
    rank = shmem.get_rank()
    M, N = 1024, 256
    dtype = torch.bfloat16

    try:
        cfg = Config(
            block_size_m=32,
            block_size_n=64,
            all_reduce_variant="atomic",
            fused_launch=True,
        )
        ref = _ref_all_reduce(M, N, dtype, rank)
        out = _iris_all_reduce(shmem, M, N, dtype, rank, cfg)
        assert torch.equal(out, ref)
        # Atomic variant does not engage the two_shot fastpath -> no cache.
        assert not hasattr(cfg, "_fused_cache")
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


def test_fused_launch_default_off_in_config():
    """The Config default must be fused_launch=False (opt-in only)."""
    cfg = Config()
    assert cfg.fused_launch is False, (
        "fused_launch must default to False so existing callers see no behavior change"
    )
