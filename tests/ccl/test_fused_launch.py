# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
K-871: Tests for the fused-launch fastpath ported from K-820 to
all_gather, reduce_scatter, and all_to_all collectives.

Distributed tests run under torchrun via tests/run_tests_distributed.py.
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


# ---------------------------------------------------------------------------
# Unit tests (no GPU required for the cache plumbing itself).
# ---------------------------------------------------------------------------


def test_fused_launch_default_off_in_config():
    """Config().fused_launch must default to False — fastpath is opt-in."""
    assert Config().fused_launch is False


@pytest.mark.parametrize("val,expected", [
    ("1", True), ("true", True), ("yes", True), ("on", True),
    ("TRUE", True), ("Yes", True), ("0", False), ("", False),
])
def test_fused_launch_env_var_recognized(monkeypatch, val, expected):
    """``IRIS_CCL_FUSED_LAUNCH=...`` synonyms recognised."""
    if val:
        monkeypatch.setenv("IRIS_CCL_FUSED_LAUNCH", val)
    else:
        monkeypatch.delenv("IRIS_CCL_FUSED_LAUNCH", raising=False)
    assert fused_launch_enabled() is expected


def test_get_or_build_cache_lazy_and_per_config():
    """Cache is lazy + per-Config (no cross-workload pollution)."""
    config_a = Config(fused_launch=True)
    config_b = Config(fused_launch=True)

    assert getattr(config_a, "_fused_cache", None) is None
    cache_a = get_or_build_cache(config_a)
    assert isinstance(cache_a, dict)
    assert config_a._fused_cache is cache_a
    assert get_or_build_cache(config_a) is cache_a  # idempotent
    assert get_or_build_cache(config_b) is not cache_a  # per-Config


def test_descriptor_invoke_calls_kernel_with_io_then_args():
    """``_LaunchDescriptor.invoke`` prepends (input, output) to captured args."""
    captured = []

    class _FakeKernel:
        def __getitem__(self, grid):
            def _launch(*args, **kwargs):
                captured.append((grid, args, kwargs))
            return _launch

    desc = _LaunchDescriptor(
        kernel_fn=_FakeKernel(), grid=(64,),
        args_after_io=(7, 8, 9), kwargs={"num_warps": 4},
    )
    desc.invoke("INPUT", "OUTPUT")
    assert captured == [((64,), ("INPUT", "OUTPUT", 7, 8, 9), {"num_warps": 4})]


# ---------------------------------------------------------------------------
# Distributed correctness tests (require torchrun + GPUs). Mirror K-820:
# cold call populates cache, warm calls hit fastpath, outputs must match
# a baseline (PyTorch / iris-baseline-config) reference.
# ---------------------------------------------------------------------------


def _require_dist():
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")


def _setup_dist_ctx():
    """Shared boilerplate: 4GB iris ctx + (rank, world)."""
    shmem = iris.iris(1 << 32)
    return shmem, shmem.get_rank(), shmem.get_num_ranks()


def _run_collective_loop(shmem, op, iris_out, iris_in, ref, *, iters, config, key, **op_kwargs):
    """Run ``op`` ``iters`` times and assert bitwise match against ``ref``
    on every call. Verify the descriptor cache contains ``key`` afterward."""
    for i in range(iters):
        iris_out.zero_()
        shmem.barrier()
        op(iris_out, iris_in, config=config, **op_kwargs)
        torch.cuda.synchronize()
        assert torch.equal(iris_out, ref), f"iter {i}: fused vs baseline mismatch"
    cache = get_or_build_cache(config)
    assert key in cache, f"descriptor for {key} must be cached"


def test_all_gather_fused_matches_reference():
    """Cold + warm fastpath all_gather must match torch.all_gather."""
    _require_dist()
    shmem, rank, world_size = _setup_dist_ctx()
    M, N, dtype = 256, 64, torch.bfloat16

    pt_in = torch.full((M, N), float(rank + 1), dtype=dtype, device=f"cuda:{rank}")
    pt_out = torch.zeros(world_size * M, N, dtype=dtype, device=f"cuda:{rank}")
    shmem.barrier()
    dist.all_gather_into_tensor(pt_out, pt_in)
    torch.cuda.synchronize()

    iris_in = shmem.zeros((M, N), dtype=dtype); iris_in.copy_(pt_in)
    iris_out = shmem.zeros((world_size * M, N), dtype=dtype)
    config = Config(block_size_m=32, block_size_n=64, fused_launch=True)

    _run_collective_loop(
        shmem, shmem.ccl.all_gather, iris_out, iris_in, pt_out,
        iters=11, config=config, key=("all_gather", M, N, dtype),
    )


def test_reduce_scatter_fused_matches_reference():
    """Cold + warm fastpath reduce_scatter must match the iris baseline."""
    _require_dist()
    shmem, rank, world_size = _setup_dist_ctx()
    M, N, dtype = 256, 64, torch.bfloat16

    iris_in = shmem.zeros((M, N), dtype=dtype)
    iris_in.copy_(torch.full((M, N), float(rank + 1), dtype=dtype, device=f"cuda:{rank}"))

    ref_out = shmem.zeros((M, N), dtype=dtype)
    ref_cfg = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=1)
    shmem.barrier(); shmem.ccl.reduce_scatter(ref_out, iris_in, config=ref_cfg)
    torch.cuda.synchronize()
    ref = ref_out.clone()

    iris_out = shmem.zeros((M, N), dtype=dtype)
    fused_cfg = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=1, fused_launch=True)
    _run_collective_loop(
        shmem, shmem.ccl.reduce_scatter, iris_out, iris_in, ref,
        iters=5, config=fused_cfg, key=("reduce_scatter", M, N, dtype),
    )


def test_all_to_all_fused_matches_reference():
    """Cold + warm fastpath all_to_all must match the iris baseline."""
    _require_dist()
    shmem, rank, world_size = _setup_dist_ctx()
    M, N_per, dtype = 128, 64, torch.bfloat16
    total_N = N_per * world_size

    full = torch.empty((M, total_N), dtype=dtype, device=f"cuda:{rank}")
    for i in range(world_size):
        full[:, i * N_per:(i + 1) * N_per] = float(rank * world_size + i + 1)
    iris_in = shmem.zeros((M, total_N), dtype=dtype); iris_in.copy_(full)

    ref_out = shmem.zeros((M, total_N), dtype=dtype)
    shmem.barrier(); shmem.ccl.all_to_all(ref_out, iris_in, config=Config(block_size_m=32, block_size_n=128))
    torch.cuda.synchronize()
    ref = ref_out.clone()

    iris_out = shmem.zeros((M, total_N), dtype=dtype)
    fused_cfg = Config(block_size_m=32, block_size_n=128, fused_launch=True)
    _run_collective_loop(
        shmem, shmem.ccl.all_to_all, iris_out, iris_in, ref,
        iters=5, config=fused_cfg, key=("all_to_all", M, total_N, dtype),
    )


def test_fused_launch_distinct_shapes_distinct_entries():
    """Distinct (M, N) shapes for the same collective produce separate entries."""
    _require_dist()
    shmem, rank, world_size = _setup_dist_ctx()
    dtype = torch.bfloat16
    config = Config(block_size_m=32, block_size_n=64, fused_launch=True)

    for M, N in [(128, 64), (256, 64), (512, 64)]:
        iris_in = shmem.zeros((M, N), dtype=dtype)
        iris_in.copy_(torch.full((M, N), float(rank + 1), dtype=dtype, device=f"cuda:{rank}"))
        iris_out = shmem.zeros((world_size * M, N), dtype=dtype)
        shmem.barrier(); shmem.ccl.all_gather(iris_out, iris_in, config=config)
        torch.cuda.synchronize()

    keys = [k for k in get_or_build_cache(config) if k[0] == "all_gather"]
    assert len(keys) == 3, f"expected 3 entries, got {keys}"


def test_fused_launch_disabled_does_not_populate_cache(monkeypatch):
    """``fused_launch=False`` (default) leaves ``_fused_cache`` unset.

    Explicitly clears ``IRIS_CCL_FUSED_LAUNCH`` so the test holds even when
    the surrounding shell exports the env-var override.
    """
    _require_dist()
    monkeypatch.delenv("IRIS_CCL_FUSED_LAUNCH", raising=False)
    shmem, rank, world_size = _setup_dist_ctx()
    dtype = torch.bfloat16
    config = Config(block_size_m=32, block_size_n=64)  # default fused_launch=False

    iris_in = shmem.zeros((128, 64), dtype=dtype)
    iris_in.copy_(torch.full((128, 64), float(rank + 1), dtype=dtype, device=f"cuda:{rank}"))
    iris_out = shmem.zeros((world_size * 128, 64), dtype=dtype)
    shmem.barrier(); shmem.ccl.all_gather(iris_out, iris_in, config=config)
    torch.cuda.synchronize()

    assert getattr(config, "_fused_cache", None) is None


def test_fused_launch_distinct_dtypes_distinct_entries():
    """bf16 vs fp16 for the same shape produce distinct cache entries."""
    _require_dist()
    shmem, rank, world_size = _setup_dist_ctx()
    config = Config(block_size_m=32, block_size_n=64, fused_launch=True)

    for dtype in (torch.bfloat16, torch.float16):
        iris_in = shmem.zeros((128, 64), dtype=dtype)
        iris_in.copy_(torch.full((128, 64), float(rank + 1), dtype=dtype, device=f"cuda:{rank}"))
        iris_out = shmem.zeros((world_size * 128, 64), dtype=dtype)
        shmem.barrier(); shmem.ccl.all_gather(iris_out, iris_in, config=config)
        torch.cuda.synchronize()

    keys = [k for k in get_or_build_cache(config) if k[0] == "all_gather" and k[1] == 128 and k[2] == 64]
    assert len(keys) == 2, f"expected 2 dtype-keyed entries, got {keys}"
