# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Unit tests for the analytical reference-output helper used by
``benchmark/ccl/comprehensive_sweep.py`` to verify iris kernel correctness.

The helper is a pure-Python tensor builder (no GPU, no torch.distributed),
so we can pin its mathematical contract on a CPU CI host. The companion
test ``test_default_config.py`` already provides the ``tritonblas`` and
``iris.hip.get_num_xcc`` stubs we need to import the iris package.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


# Reuse the import-side ``tritonblas`` stub installed by test_default_config.
# Pytest collects in alphabetical order, but importing the helper twice is
# idempotent and we want this file to be self-contained when run in isolation.
class _PermissiveModule(types.ModuleType):
    def __getattr__(self, name):  # noqa: D401 - sentinel
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        sentinel = type(name, (), {})
        setattr(self, name, sentinel)
        return sentinel


if "tritonblas" not in sys.modules:  # pragma: no cover - import-side stub
    _tb = _PermissiveModule("tritonblas")
    _tb_kernels = _PermissiveModule("tritonblas.kernels")
    _tb_stages = _PermissiveModule("tritonblas.kernels.stages")
    _tb.kernels = _tb_kernels
    _tb_kernels.stages = _tb_stages
    sys.modules["tritonblas"] = _tb
    sys.modules["tritonblas.kernels"] = _tb_kernels
    sys.modules["tritonblas.kernels.stages"] = _tb_stages


torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def reference_output():
    """Load ``reference_output`` from ``benchmark/ccl/comprehensive_sweep.py``.

    The benchmark script is not a regular package — load it via importlib so
    the unit test does not have to depend on a benchmark-dir entry point.
    """
    repo_root = Path(__file__).resolve().parents[2]
    sweep_path = repo_root / "benchmark" / "ccl" / "comprehensive_sweep.py"
    spec = importlib.util.spec_from_file_location("iris_ccl_sweep", sweep_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec_module so ``dataclass`` field-annotation resolution
    # (which does ``sys.modules.get(cls.__module__).__dict__``) finds the
    # module while the module body is running.
    sys.modules["iris_ccl_sweep"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("iris_ccl_sweep", None)
        raise
    return module.reference_output


def test_reference_output_all_reduce(reference_output):
    """all_reduce reference: every element equals sum_{r=0..W-1} (r+1) = W(W+1)/2."""
    out = reference_output("all_reduce", m=4, n=8, world_size=8, rank=3, dtype=torch.float32, device="cpu")
    assert out.shape == (4, 8)
    expected = float(8 * 9 / 2)  # 36
    assert torch.all(out == expected)


def test_reference_output_reduce_scatter(reference_output):
    """reduce_scatter reference: same closed-form sum as all_reduce, rank-independent."""
    for rank in range(8):
        out = reference_output("reduce_scatter", m=4, n=8, world_size=8, rank=rank, dtype=torch.float32, device="cpu")
        assert out.shape == (4, 8)
        assert torch.all(out == 36.0)


def test_reference_output_all_gather(reference_output):
    """all_gather reference: row-block i is filled with (i + 1)."""
    world_size = 4
    m = 2
    n = 5
    for rank in range(world_size):
        out = reference_output(
            "all_gather", m=m, n=n, world_size=world_size, rank=rank, dtype=torch.float32, device="cpu"
        )
        assert out.shape == (world_size * m, n)
        for src in range(world_size):
            block = out[src * m : (src + 1) * m]
            assert torch.all(block == float(src + 1)), f"rank={rank} src={src} block={block}"


def test_reference_output_all_to_all(reference_output):
    """all_to_all reference: column chunk i is filled with (i + 1)."""
    world_size = 4
    m = 3
    per = 2
    n = world_size * per
    out = reference_output("all_to_all", m=m, n=n, world_size=world_size, rank=2, dtype=torch.float32, device="cpu")
    assert out.shape == (m, n)
    for src in range(world_size):
        chunk = out[:, src * per : (src + 1) * per]
        assert torch.all(chunk == float(src + 1)), f"src={src} chunk={chunk}"


def test_reference_output_all_to_all_n_not_divisible_raises(reference_output):
    """all_to_all requires N divisible by world_size — surfaces shape-grid bugs early."""
    with pytest.raises(ValueError, match="divisible by world_size"):
        reference_output("all_to_all", m=2, n=7, world_size=4, rank=0, dtype=torch.float32, device="cpu")


def test_reference_output_unknown_collective_raises(reference_output):
    with pytest.raises(ValueError, match="Unknown collective"):
        reference_output("broadcast", m=2, n=2, world_size=2, rank=0, dtype=torch.float32, device="cpu")
