# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for the K-820/K-861 per-Config fused-launch fastpath.

These tests are intentionally narrow: they assert *behaviour of the cache
itself*, not numerical correctness of the underlying collectives (which is
covered by ``test_all_to_all.py`` / ``test_all_gather.py`` /
``test_all_reduce.py``).  Specifically each test verifies:

1. **Positive hit path** — a second call with the same Config + shape +
   dtype hits the cache.  We assert this two ways:
     * the module-level ``record_hit`` counter increments
     * the cached closure object identity is reused (same ``id``)
   Either alone would let a silent-miss bug pass; together they only pass
   if the wrapper actually took the fastpath branch.

2. **Negative path / cache miss on key change** — a call with a different
   ``(M, N, dtype)`` builds a new entry rather than incorrectly reusing a
   stale closure.  This is the regression test for an over-broad cache
   key (the original K-820 bug class).
"""

import pytest
import torch
import torch.distributed as dist
import iris
from iris.ccl import Config
from iris.ccl import launch_cache


def _maybe_skip():
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")


def _fresh_shmem():
    return iris.iris(2**32)


@pytest.fixture(autouse=True)
def _reset_cache_stats():
    launch_cache.reset_stats()
    yield
    launch_cache.reset_stats()


# --------------------------------------------------------------------------
# all_to_all
# --------------------------------------------------------------------------
def test_all_to_all_fastpath_hit_and_miss():
    _maybe_skip()
    shmem = _fresh_shmem()
    rank = shmem.get_rank()
    world = shmem.get_num_ranks()
    dtype = torch.bfloat16

    cfg = Config(block_size_m=32, block_size_n=64)
    M, N = 64, 64
    inp = shmem.zeros((M, N * world), dtype=dtype)
    out = shmem.zeros((M, N * world), dtype=dtype)
    inp.fill_(float(rank + 1))
    shmem.barrier()

    # Cold call: miss + populate
    shmem.ccl.all_to_all(out, inp, config=cfg)
    torch.cuda.synchronize()
    s1 = launch_cache.get_stats()
    assert s1["hits"] == 0 and s1["misses"] == 1, f"cold call: {s1}"
    cache = cfg._iris_launch_cache
    assert len(cache) == 1
    cached_obj = next(iter(cache.values()))
    cached_id = id(cached_obj)

    # Warm call (same key): MUST hit.
    shmem.ccl.all_to_all(out, inp, config=cfg)
    torch.cuda.synchronize()
    s2 = launch_cache.get_stats()
    assert s2["hits"] == 1 and s2["misses"] == 1, f"warm call: {s2}"
    # Cached closure object identity is reused (proves we took the branch,
    # not a silent miss that re-built and re-stored an equivalent closure).
    assert id(next(iter(cache.values()))) == cached_id, "closure replaced on hit"
    assert len(cache) == 1, "cache size grew on a hit"

    # Negative path: shape change must miss and add a new entry.
    M2 = M * 2
    inp2 = shmem.zeros((M2, N * world), dtype=dtype)
    out2 = shmem.zeros((M2, N * world), dtype=dtype)
    shmem.barrier()
    shmem.ccl.all_to_all(out2, inp2, config=cfg)
    torch.cuda.synchronize()
    s3 = launch_cache.get_stats()
    assert s3["misses"] == 2, f"shape change must miss: {s3}"
    assert len(cache) == 2, f"shape change must add entry, got {len(cache)}"

    # Negative path: dtype change must also miss.
    inp3 = shmem.zeros((M, N * world), dtype=torch.float16)
    out3 = shmem.zeros((M, N * world), dtype=torch.float16)
    shmem.barrier()
    shmem.ccl.all_to_all(out3, inp3, config=cfg)
    torch.cuda.synchronize()
    s4 = launch_cache.get_stats()
    assert s4["misses"] == 3, f"dtype change must miss: {s4}"
    assert len(cache) == 3

    shmem.barrier()
    del shmem


# --------------------------------------------------------------------------
# all_gather
# --------------------------------------------------------------------------
def test_all_gather_fastpath_hit_and_miss():
    _maybe_skip()
    shmem = _fresh_shmem()
    rank = shmem.get_rank()
    world = shmem.get_num_ranks()
    dtype = torch.bfloat16

    cfg = Config(block_size_m=32, block_size_n=64)
    M, N = 64, 64
    inp = shmem.zeros((M, N), dtype=dtype)
    out = shmem.zeros((M * world, N), dtype=dtype)
    inp.fill_(float(rank + 1))
    shmem.barrier()

    shmem.ccl.all_gather(out, inp, config=cfg)
    torch.cuda.synchronize()
    assert launch_cache.get_stats() == {"hits": 0, "misses": 1}
    cache = cfg._iris_launch_cache
    cached_id = id(next(iter(cache.values())))

    shmem.ccl.all_gather(out, inp, config=cfg)
    torch.cuda.synchronize()
    s = launch_cache.get_stats()
    assert s["hits"] == 1, f"warm call did not hit: {s}"
    assert id(next(iter(cache.values()))) == cached_id

    # Different M -> miss.
    inp2 = shmem.zeros((M * 2, N), dtype=dtype)
    out2 = shmem.zeros((M * 2 * world, N), dtype=dtype)
    shmem.barrier()
    shmem.ccl.all_gather(out2, inp2, config=cfg)
    torch.cuda.synchronize()
    assert launch_cache.get_stats()["misses"] == 2
    assert len(cache) == 2

    shmem.barrier()
    del shmem


# --------------------------------------------------------------------------
# reduce_scatter
# --------------------------------------------------------------------------
def test_reduce_scatter_fastpath_hit_and_miss():
    _maybe_skip()
    shmem = _fresh_shmem()
    rank = shmem.get_rank()
    world = shmem.get_num_ranks()
    dtype = torch.bfloat16

    cfg = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=1)
    M, N = 64, 64
    inp = shmem.zeros((M, N), dtype=dtype)
    out = shmem.zeros((M, N), dtype=dtype)
    inp.fill_(float(rank + 1))
    shmem.barrier()

    shmem.ccl.reduce_scatter(out, inp, config=cfg)
    torch.cuda.synchronize()
    assert launch_cache.get_stats() == {"hits": 0, "misses": 1}
    cache = cfg._iris_launch_cache
    cached_id = id(next(iter(cache.values())))

    shmem.ccl.reduce_scatter(out, inp, config=cfg)
    torch.cuda.synchronize()
    s = launch_cache.get_stats()
    assert s["hits"] == 1, f"warm call did not hit: {s}"
    assert id(next(iter(cache.values()))) == cached_id

    # Negative path: change dtype.
    inp2 = shmem.zeros((M, N), dtype=torch.float16)
    out2 = shmem.zeros((M, N), dtype=torch.float16)
    shmem.barrier()
    shmem.ccl.reduce_scatter(out2, inp2, config=cfg)
    torch.cuda.synchronize()
    assert launch_cache.get_stats()["misses"] == 2
    assert len(cache) == 2

    shmem.barrier()
    del shmem


# --------------------------------------------------------------------------
# all_reduce two_shot (fastpath eligible variant)
# --------------------------------------------------------------------------
def test_all_reduce_two_shot_fastpath_hit_and_miss():
    _maybe_skip()
    shmem = _fresh_shmem()
    rank = shmem.get_rank()
    world = shmem.get_num_ranks()
    dtype = torch.bfloat16

    cfg = Config(
        all_reduce_variant="two_shot",
        block_size_m=32,
        block_size_n=64,
        all_reduce_distribution=0,
    )
    M, N = 64, 64
    inp = shmem.zeros((M, N), dtype=dtype)
    out = shmem.zeros((M, N), dtype=dtype)
    inp.fill_(float(rank + 1))
    shmem.barrier()

    shmem.ccl.all_reduce(out, inp, config=cfg)
    torch.cuda.synchronize()
    assert launch_cache.get_stats() == {"hits": 0, "misses": 1}
    cache = cfg._iris_launch_cache
    cached_id = id(next(iter(cache.values())))

    shmem.ccl.all_reduce(out, inp, config=cfg)
    torch.cuda.synchronize()
    s = launch_cache.get_stats()
    assert s["hits"] == 1, f"warm call did not hit: {s}"
    assert id(next(iter(cache.values()))) == cached_id

    # Numerical sanity check: cached call should still produce sum-of-ranks.
    expected = sum(r + 1 for r in range(world))
    got = out[0, 0].item()
    assert abs(got - expected) < 1e-3, f"AR two_shot wrong sum on cached call: {got} vs {expected}"

    # Different shape -> new entry.
    inp2 = shmem.zeros((M * 2, N), dtype=dtype)
    out2 = shmem.zeros((M * 2, N), dtype=dtype)
    shmem.barrier()
    shmem.ccl.all_reduce(out2, inp2, config=cfg)
    torch.cuda.synchronize()
    assert launch_cache.get_stats()["misses"] == 2
    assert len(cache) == 2

    shmem.barrier()
    del shmem


# --------------------------------------------------------------------------
# all_reduce with non-fastpath variant must NEVER touch the cache.
# (Atomic / ring / spinlock are conservatively excluded; this protects
# against a future bug that accidentally caches a stateful workspace.)
# --------------------------------------------------------------------------
def test_all_reduce_atomic_does_not_use_fastpath():
    _maybe_skip()
    shmem = _fresh_shmem()
    rank = shmem.get_rank()
    world = shmem.get_num_ranks()
    dtype = torch.bfloat16

    cfg = Config(
        all_reduce_variant="atomic",
        block_size_m=32,
        block_size_n=64,
        all_reduce_distribution=1,
    )
    M, N = 64, 64
    inp = shmem.zeros((M, N), dtype=dtype)
    out = shmem.zeros((M, N), dtype=dtype)
    inp.fill_(float(rank + 1))
    shmem.barrier()

    shmem.ccl.all_reduce(out, inp, config=cfg)
    torch.cuda.synchronize()
    shmem.ccl.all_reduce(out, inp, config=cfg)
    torch.cuda.synchronize()

    s = launch_cache.get_stats()
    assert s["hits"] == 0, f"atomic variant must not use fastpath cache: {s}"
    # Either no cache attribute was ever attached, or it stayed empty.
    assert getattr(cfg, "_iris_launch_cache", None) in (None, {}), "atomic variant must not populate cache"

    shmem.barrier()
    del shmem
