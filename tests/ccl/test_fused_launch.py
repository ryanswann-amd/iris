# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
K-871: Tests for the fused-launch fastpath ported from K-820 to
all_gather, reduce_scatter, and all_to_all collectives.

These tests run under torchrun (distributed pytest harness in
``tests/run_tests_distributed.py``).
"""

import pytest
import torch
import torch.distributed as dist

import iris
from iris.ccl import Config
from iris.ccl.triton._fused_launch_cache import (
    _LaunchDescriptor,
    fused_launch_enabled,
    get_or_build_cache,
)


# --------------------------------------------------------------------------
# Sanity / unit tests (no GPU required for the cache plumbing itself)
# --------------------------------------------------------------------------


def test_fused_launch_default_off_in_config():
    """Config().fused_launch must default to False so that the fastpath is
    opt-in and existing callers see byte-for-byte identical behaviour."""
    config = Config()
    assert config.fused_launch is False


def test_fused_launch_env_var_recognized(monkeypatch):
    """``IRIS_CCL_FUSED_LAUNCH=1`` (and a few synonyms) should activate the
    fastpath even when ``Config(fused_launch=False)``."""
    for val in ("1", "true", "yes", "on", "TRUE", "Yes"):
        monkeypatch.setenv("IRIS_CCL_FUSED_LAUNCH", val)
        assert fused_launch_enabled(), f"value {val!r} should enable fastpath"
    monkeypatch.setenv("IRIS_CCL_FUSED_LAUNCH", "0")
    assert not fused_launch_enabled()
    monkeypatch.delenv("IRIS_CCL_FUSED_LAUNCH", raising=False)
    assert not fused_launch_enabled()


def test_get_or_build_cache_lazy_and_per_config():
    """Cache is created lazily on first lookup and stored on the Config
    instance (per-Config scoping prevents cross-workload pollution)."""
    config_a = Config(fused_launch=True)
    config_b = Config(fused_launch=True)

    assert getattr(config_a, "_fused_cache", None) is None
    cache_a = get_or_build_cache(config_a)
    assert isinstance(cache_a, dict)
    assert config_a._fused_cache is cache_a

    cache_b = get_or_build_cache(config_b)
    assert cache_b is not cache_a, "different Configs must have separate caches"

    # Repeat get returns the same object.
    assert get_or_build_cache(config_a) is cache_a


def test_descriptor_invoke_calls_kernel_with_io_then_args():
    """``_LaunchDescriptor.invoke`` must call the captured kernel with the
    runtime ``(input, output)`` tensors prepended to the captured args."""
    captured_grid = []
    captured_args = []
    captured_kwargs = []

    class _FakeKernel:
        def __getitem__(self, grid):
            captured_grid.append(grid)

            def _launch(*args, **kwargs):
                captured_args.append(args)
                captured_kwargs.append(kwargs)

            return _launch

    desc = _LaunchDescriptor(
        kernel_fn=_FakeKernel(),
        grid=(64,),
        args_after_io=(7, 8, 9),
        kwargs={"num_warps": 4},
    )
    desc.invoke("INPUT", "OUTPUT")

    assert captured_grid == [(64,)]
    assert captured_args == [("INPUT", "OUTPUT", 7, 8, 9)]
    assert captured_kwargs == [{"num_warps": 4}]


# --------------------------------------------------------------------------
# Distributed correctness tests (require torchrun + GPUs).
# These mirror the K-820 test pattern: cold call populates the cache, warm
# calls hit the fastpath, and outputs must match a reference implementation
# (PyTorch / iris baseline).
# --------------------------------------------------------------------------


def _require_dist():
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")


def test_all_gather_fused_matches_reference():
    """Cold + warm fastpath all_gather must match torch.all_gather."""
    _require_dist()
    heap_size = 1 << 32  # 4 GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    M, N = 256, 64
    dtype = torch.bfloat16

    pt_in = torch.full((M, N), float(rank + 1), dtype=dtype, device=f"cuda:{rank}")
    pt_out = torch.zeros(world_size * M, N, dtype=dtype, device=f"cuda:{rank}")
    shmem.barrier()
    dist.all_gather_into_tensor(pt_out, pt_in)
    torch.cuda.synchronize()

    iris_in = shmem.zeros((M, N), dtype=dtype)
    iris_in.copy_(pt_in)
    iris_out = shmem.zeros((world_size * M, N), dtype=dtype)

    config = Config(block_size_m=32, block_size_n=64, fused_launch=True)

    # Cold call (descriptor populated as side-effect).
    shmem.barrier()
    shmem.ccl.all_gather(iris_out, iris_in, config=config)
    torch.cuda.synchronize()
    assert torch.equal(iris_out, pt_out), "cold-call output must match"

    # Warm calls — fastpath now active.
    cache = get_or_build_cache(config)
    assert ("all_gather", M, N, dtype) in cache, "descriptor must be cached"

    for _ in range(10):
        iris_out.zero_()
        shmem.barrier()
        shmem.ccl.all_gather(iris_out, iris_in, config=config)
        torch.cuda.synchronize()
        assert torch.equal(iris_out, pt_out), "warm-path output must match"


def test_reduce_scatter_fused_matches_reference():
    """Cold + warm fastpath reduce_scatter must match the iris baseline."""
    _require_dist()
    heap_size = 1 << 32
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    M, N = 256, 64
    dtype = torch.bfloat16

    iris_in = shmem.zeros((M, N), dtype=dtype)
    # Fill with deterministic pattern.
    iris_in.copy_(torch.full((M, N), float(rank + 1), dtype=dtype, device=f"cuda:{rank}"))

    # Reference path: baseline reduce_scatter (no fused_launch).
    ref_out = shmem.zeros((M, N), dtype=dtype)
    config_ref = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=1)
    shmem.barrier()
    shmem.ccl.reduce_scatter(ref_out, iris_in, config=config_ref)
    torch.cuda.synchronize()
    ref = ref_out.clone()

    # Fused-launch path.
    iris_out = shmem.zeros((M, N), dtype=dtype)
    config_fused = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=1, fused_launch=True)

    # Cold + warm.
    for i in range(5):
        iris_out.zero_()
        shmem.barrier()
        shmem.ccl.reduce_scatter(iris_out, iris_in, config=config_fused)
        torch.cuda.synchronize()
        assert torch.equal(iris_out, ref), f"iter {i}: fused vs baseline mismatch"

    cache = get_or_build_cache(config_fused)
    assert ("reduce_scatter", M, N, dtype) in cache


def test_all_to_all_fused_matches_reference():
    """Cold + warm fastpath all_to_all must match the iris baseline."""
    _require_dist()
    heap_size = 1 << 32
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    M, N_per = 128, 64
    total_N = N_per * world_size
    dtype = torch.bfloat16

    # Initialise each per-rank chunk distinctly so all_to_all permutes data.
    iris_in = shmem.zeros((M, total_N), dtype=dtype)
    full = torch.empty((M, total_N), dtype=dtype, device=f"cuda:{rank}")
    for i in range(world_size):
        full[:, i * N_per : (i + 1) * N_per] = float(rank * world_size + i + 1)
    iris_in.copy_(full)

    # Reference (baseline iris path).
    ref_out = shmem.zeros((M, total_N), dtype=dtype)
    config_ref = Config(block_size_m=32, block_size_n=128)
    shmem.barrier()
    shmem.ccl.all_to_all(ref_out, iris_in, config=config_ref)
    torch.cuda.synchronize()
    ref = ref_out.clone()

    iris_out = shmem.zeros((M, total_N), dtype=dtype)
    config_fused = Config(block_size_m=32, block_size_n=128, fused_launch=True)

    for i in range(5):
        iris_out.zero_()
        shmem.barrier()
        shmem.ccl.all_to_all(iris_out, iris_in, config=config_fused)
        torch.cuda.synchronize()
        assert torch.equal(iris_out, ref), f"iter {i}: fused vs baseline mismatch"

    cache = get_or_build_cache(config_fused)
    assert ("all_to_all", M, total_N, dtype) in cache


def test_fused_launch_distinct_shapes_distinct_entries():
    """Different ``(M, N)`` shapes for the same collective produce separate
    cache entries."""
    _require_dist()
    heap_size = 1 << 32
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    dtype = torch.bfloat16

    config = Config(block_size_m=32, block_size_n=64, fused_launch=True)

    for M, N in [(128, 64), (256, 64), (512, 64)]:
        iris_in = shmem.zeros((M, N), dtype=dtype)
        iris_in.copy_(torch.full((M, N), float(rank + 1), dtype=dtype, device=f"cuda:{rank}"))
        iris_out = shmem.zeros((world_size * M, N), dtype=dtype)
        shmem.barrier()
        shmem.ccl.all_gather(iris_out, iris_in, config=config)
        torch.cuda.synchronize()

    cache = get_or_build_cache(config)
    keys = [k for k in cache if k[0] == "all_gather"]
    assert len(keys) == 3, f"expected 3 entries, got {keys}"


def test_fused_launch_disabled_does_not_populate_cache():
    """``fused_launch=False`` (default) must leave ``_fused_cache`` unset."""
    _require_dist()
    heap_size = 1 << 32
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    dtype = torch.bfloat16

    config = Config(block_size_m=32, block_size_n=64)  # default fused_launch=False
    iris_in = shmem.zeros((128, 64), dtype=dtype)
    iris_in.copy_(torch.full((128, 64), float(rank + 1), dtype=dtype, device=f"cuda:{rank}"))
    iris_out = shmem.zeros((world_size * 128, 64), dtype=dtype)
    shmem.barrier()
    shmem.ccl.all_gather(iris_out, iris_in, config=config)
    torch.cuda.synchronize()

    assert getattr(config, "_fused_cache", None) is None


def test_fused_launch_distinct_dtypes_distinct_entries():
    """bf16 vs fp16 for the same shape produce distinct cache entries."""
    _require_dist()
    heap_size = 1 << 32
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    config = Config(block_size_m=32, block_size_n=64, fused_launch=True)

    for dtype in (torch.bfloat16, torch.float16):
        iris_in = shmem.zeros((128, 64), dtype=dtype)
        iris_in.copy_(torch.full((128, 64), float(rank + 1), dtype=dtype, device=f"cuda:{rank}"))
        iris_out = shmem.zeros((world_size * 128, 64), dtype=dtype)
        shmem.barrier()
        shmem.ccl.all_gather(iris_out, iris_in, config=config)
        torch.cuda.synchronize()

    cache = get_or_build_cache(config)
    keys = [k for k in cache if k[0] == "all_gather" and k[1] == 128 and k[2] == 64]
    assert len(keys) == 2, f"expected 2 dtype-keyed entries, got {keys}"
