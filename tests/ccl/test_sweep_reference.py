# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Unit tests for the analytical reference-output helper used by
``benchmark/ccl/comprehensive_sweep.py`` to verify iris kernel correctness.

The helper is a pure-Python tensor builder (no GPU, no torch.distributed),
so we can pin its mathematical contract on a CPU CI host. The companion
test ``test_default_config.py`` already provides the ``tritonblas`` and
``iris.hip.get_num_xcc`` stubs we need to import the iris package.

This module also pins the load-bearing safety net the harness adds on top
of ``reference_output``: the pure ``_compare_to_reference`` helper, the new
``correct``/``max_abs_err`` CSV columns, and ``_correctness_exit_code``
(non-zero exit on any iris correctness failure).
"""

from __future__ import annotations

import csv
import importlib.util
import logging
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
def sweep_module():
    """Load ``benchmark/ccl/comprehensive_sweep.py`` as an importable module.

    The benchmark script is not a regular package — load it via importlib so
    the unit tests do not have to depend on a benchmark-dir entry point.
    Returned module exposes ``reference_output``, ``_compare_to_reference``,
    ``_correctness_exit_code``, ``_write_csv``, and ``_Row``.
    """
    if "iris_ccl_sweep" in sys.modules:  # pragma: no cover - per-session reuse
        return sys.modules["iris_ccl_sweep"]
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
    return module


@pytest.fixture(scope="module")
def reference_output(sweep_module):
    """Convenience handle to ``sweep_module.reference_output``."""
    return sweep_module.reference_output


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


# --------------------------------------------------------------------------
# Verifier safety-net tests: pin the harness's fail-loud contract on the
# pure-tensor ``_compare_to_reference`` helper, the new CSV columns, and
# the non-zero exit-code path.
# --------------------------------------------------------------------------


def test_compare_to_reference_passing_tensor(sweep_module):
    """A bit-perfect iris output reports correct=True with max_abs_err=0."""
    expected = sweep_module.reference_output(
        "all_reduce", m=4, n=8, world_size=8, rank=0, dtype=torch.float32, device="cpu"
    )
    correct, max_abs_err = sweep_module._compare_to_reference(
        expected.clone(), "all_reduce", m=4, n=8, world_size=8, rank=0, dtype=torch.float32
    )
    assert correct is True
    assert max_abs_err == 0.0


def test_compare_to_reference_failing_tensor(sweep_module):
    """A deliberately wrong iris output reports correct=False with populated max_abs_err."""
    expected = sweep_module.reference_output(
        "all_reduce", m=4, n=8, world_size=8, rank=0, dtype=torch.float32, device="cpu"
    )
    actual = expected.clone()
    actual[0, 0] = expected[0, 0] + 5.0  # well above 1e-2 tolerance
    correct, max_abs_err = sweep_module._compare_to_reference(
        actual, "all_reduce", m=4, n=8, world_size=8, rank=0, dtype=torch.float32
    )
    assert correct is False
    assert max_abs_err == pytest.approx(5.0)


def test_compare_to_reference_within_tolerance(sweep_module):
    """A perturbation just under the dtype tolerance still reports correct=True."""
    expected = sweep_module.reference_output(
        "all_reduce", m=2, n=2, world_size=4, rank=0, dtype=torch.float32, device="cpu"
    )
    actual = expected.clone()
    actual[0, 0] = expected[0, 0] + 1e-3  # below the 1e-2 default tolerance
    correct, max_abs_err = sweep_module._compare_to_reference(
        actual, "all_reduce", m=2, n=2, world_size=4, rank=0, dtype=torch.float32
    )
    assert correct is True
    assert max_abs_err == pytest.approx(1e-3, rel=1e-3)


def _row(
    sweep_module,
    *,
    impl: str,
    correct: bool | None,
    max_abs_err: float = -1.0,
    collective: str = "all_reduce",
    total_bytes: int = 1024,
):
    """Build a minimal ``_Row`` for the CSV / exit-code assertions."""
    return sweep_module._Row(
        collective=collective,
        impl=impl,
        dtype="fp16",
        total_bytes=total_bytes,
        M=4,
        N=8,
        world_size=8,
        mean_ms=1.0,
        min_ms=0.9,
        median_ms=1.0,
        bus_gbps=10.0,
        correct=correct,
        max_abs_err=max_abs_err,
    )


def test_write_csv_emits_correct_and_max_abs_err_columns(sweep_module, tmp_path):
    """``_write_csv`` writes the new ``correct`` + ``max_abs_err`` columns per row."""
    rows = [
        _row(sweep_module, impl="iris", correct=True, max_abs_err=0.0),
        _row(sweep_module, impl="iris", correct=False, max_abs_err=4.2, total_bytes=2048),
        _row(sweep_module, impl="rccl", correct=None, max_abs_err=-1.0),
    ]
    csv_path = tmp_path / "out.csv"
    sweep_module._write_csv(rows, csv_path)
    with csv_path.open() as f:
        records = list(csv.DictReader(f))
    assert "correct" in records[0]
    assert "max_abs_err" in records[0]
    assert records[0]["correct"] == "true"
    assert records[0]["max_abs_err"] == "0"
    assert records[1]["correct"] == "false"
    assert float(records[1]["max_abs_err"]) == pytest.approx(4.2)
    # RCCL rows leave the verifier columns empty (sentinel for "not checked").
    assert records[2]["correct"] == ""
    assert records[2]["max_abs_err"] == ""


def test_correctness_exit_code_zero_when_all_pass(sweep_module):
    """No iris failures → exit code 0."""
    rows = [
        _row(sweep_module, impl="iris", correct=True, max_abs_err=0.0),
        _row(sweep_module, impl="rccl", correct=None),
    ]
    assert sweep_module._correctness_exit_code(rows) == 0


def test_correctness_exit_code_nonzero_on_iris_failure(sweep_module, caplog):
    """Any iris row with correct=False → exit code 1 + a logged FAIL summary."""
    rows = [
        _row(sweep_module, impl="iris", correct=True, max_abs_err=0.0),
        _row(sweep_module, impl="iris", correct=False, max_abs_err=3.5, total_bytes=4096),
        _row(sweep_module, impl="rccl", correct=None),
    ]
    test_logger = logging.getLogger("iris.ccl.sweep.test_exit")
    with caplog.at_level(logging.ERROR, logger=test_logger.name):
        assert sweep_module._correctness_exit_code(rows, logger=test_logger) == 1
    # The summary line names the failing cell so a CI log scrape can find it.
    messages = [rec.getMessage() for rec in caplog.records if rec.name == test_logger.name]
    assert any("iris correctness failed" in m and "all_reduce" in m and "4096B" in m for m in messages), messages


def test_correctness_exit_code_silent_when_no_logger(sweep_module, caplog):
    """Failure path stays quiet when no logger is supplied (rank>0 in real use)."""
    rows = [_row(sweep_module, impl="iris", correct=False, max_abs_err=1.0)]
    with caplog.at_level(logging.ERROR):
        assert sweep_module._correctness_exit_code(rows, logger=None) == 1
    assert not any("iris correctness failed" in rec.getMessage() for rec in caplog.records)
