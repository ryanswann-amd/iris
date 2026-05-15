# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Unit tests for the static per-(arch, collective, message-size) defaults table
that powers ``iris.ccl.{all_reduce, all_gather, reduce_scatter, all_to_all}``
when the user calls them without an explicit ``Config``.

These tests are deliberately GPU-free **and** ROCm-free: they load
``iris/ccl/config.py`` directly via ``importlib`` so the import chain never
touches the top-level ``iris`` package (which would dlopen
``libamdhip64.so``) or ``iris.ccl`` (which imports ``triton``). The result
is a pure-Python lookup test that runs on any CPU CI host.

The public-API wiring tests at the bottom of the file fall back to plain
text inspection of the four collective stub source files for the same
reason — we don't want to require a GPU runtime just to verify that a
``config=None`` branch still calls into the table.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import types
from typing import Any

import pytest


# --------------------------------------------------------------------------
# Direct module loader — bypass ``iris/__init__.py`` and ``iris/ccl/__init__.py``
# --------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "iris" / "ccl" / "config.py"
_CCL_DIR = _REPO_ROOT / "iris" / "ccl"


def _load_config_module() -> types.ModuleType:
    """Load ``iris/ccl/config.py`` as a leaf module without touching iris/__init__.py.

    The defaults table is pure Python — there is no reason for a unit-test
    job to need ``libamdhip64.so`` or ``triton`` on the host just to assert
    on it. Doing this via ``importlib.util.spec_from_file_location`` keeps
    the test self-contained.
    """
    # Stub the top-level ``iris`` package and ``iris.hip`` so the eventual
    # ``import iris`` inside ``Config.__post_init__`` resolves without
    # dlopening HIP. We only stub if not already present so a real ROCm CI
    # host that has imported the proper iris first still works.
    if "iris" not in sys.modules:
        iris_stub = types.ModuleType("iris")
        iris_stub.__path__ = [str(_REPO_ROOT / "iris")]  # mark as package
        sys.modules["iris"] = iris_stub
    if "iris.hip" not in sys.modules:
        hip_stub = types.ModuleType("iris.hip")
        hip_stub.get_num_xcc = lambda *args, **kwargs: 8
        sys.modules["iris.hip"] = hip_stub
        sys.modules["iris"].hip = hip_stub  # type: ignore[attr-defined]

    spec = importlib.util.spec_from_file_location(
        "iris_ccl_config_under_test",
        _CONFIG_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def config_module() -> types.ModuleType:
    return _load_config_module()


COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")


# --------------------------------------------------------------------------
# Schema invariants
# --------------------------------------------------------------------------


def test_table_has_gfx942(config_module):
    """gfx942 (MI300X) must always have an entry — it's the canonical AMD arch."""
    assert "gfx942" in config_module._DEFAULTS_TABLE


@pytest.mark.parametrize("arch", ["gfx942"])
def test_table_covers_all_collectives(config_module, arch):
    table = config_module._DEFAULTS_TABLE[arch]
    assert set(table.keys()) >= set(COLLECTIVES)


@pytest.mark.parametrize("arch", ["gfx942"])
@pytest.mark.parametrize("coll", COLLECTIVES)
def test_table_buckets_sorted_and_terminate_at_inf(config_module, arch, coll):
    """Buckets must be sorted ascending by ``max_bytes`` and end at +inf."""
    buckets = config_module._DEFAULTS_TABLE[arch][coll]
    assert buckets, f"no buckets for {arch}/{coll}"
    edges = [b[0] for b in buckets]
    assert edges == sorted(edges), f"buckets unsorted for {arch}/{coll}: {edges}"
    assert math.isinf(edges[-1]), f"final bucket must be float('inf'), got {edges[-1]}"


# --------------------------------------------------------------------------
# Lookup behaviour
# --------------------------------------------------------------------------


def test_lookup_unknown_collective_raises(config_module):
    with pytest.raises(ValueError, match="Unknown collective"):
        config_module.lookup_defaults("foo", 1024, arch="gfx942")


def test_lookup_negative_bytes_raises(config_module):
    with pytest.raises(ValueError, match="non-negative"):
        config_module.lookup_defaults("all_reduce", -1, arch="gfx942")


def test_lookup_unknown_arch_falls_back(config_module):
    """Unknown arch must fall back to the default arch instead of raising."""
    overrides = config_module.lookup_defaults("all_reduce", 1024, arch="gfx_unknown")
    assert overrides, "fallback to default arch must yield non-empty overrides"


def test_lookup_zero_bytes_resolves_to_first_bucket(config_module):
    """``message_bytes == 0`` must hit the smallest bucket, not crash."""
    for coll in COLLECTIVES:
        overrides = config_module.lookup_defaults(coll, 0, arch="gfx942")
        first_bucket = config_module._DEFAULTS_TABLE["gfx942"][coll][0][1]
        assert overrides == first_bucket


@pytest.mark.parametrize("coll", COLLECTIVES)
def test_lookup_returns_first_matching_bucket(config_module, coll):
    """Walking buckets must pick the smallest ``max_bytes >= message_bytes``."""
    buckets = config_module._DEFAULTS_TABLE["gfx942"][coll]

    # Just below the first edge → first bucket
    first_edge = buckets[0][0]
    overrides = config_module.lookup_defaults(coll, max(1, first_edge - 1), arch="gfx942")
    assert overrides == buckets[0][1]

    # Just above the first edge → second bucket
    if len(buckets) >= 2:
        overrides = config_module.lookup_defaults(coll, first_edge + 1, arch="gfx942")
        assert overrides == buckets[1][1]
        # And in the "huge message" tail → final bucket
        overrides = config_module.lookup_defaults(coll, 1 << 33, arch="gfx942")
        assert overrides == buckets[-1][1]


def test_lookup_returns_a_copy(config_module):
    """Mutating the returned override dict must not corrupt the table."""
    overrides = config_module.lookup_defaults("all_reduce", 1024, arch="gfx942")
    overrides["comm_sms"] = -1
    fresh = config_module.lookup_defaults("all_reduce", 1024, arch="gfx942")
    assert fresh["comm_sms"] != -1, "table state leaked across lookups"


# --------------------------------------------------------------------------
# Auto-detect path — no torch / no GPU
# --------------------------------------------------------------------------


def test_detect_arch_falls_back_when_torch_missing(config_module, monkeypatch):
    """``_detect_arch`` must gracefully return ``_DEFAULT_ARCH`` if torch is
    unavailable or doesn't expose CUDA/HIP — this is the CPU CI path."""
    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated CPU CI environment")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    assert config_module._detect_arch() == config_module._DEFAULT_ARCH


def test_lookup_with_arch_none_uses_detector(config_module, monkeypatch):
    """``arch=None`` must consult ``_detect_arch`` rather than hardcoding gfx942."""
    monkeypatch.setattr(config_module, "_detect_arch", lambda: "gfx_made_up_for_test")
    # Unknown arch falls back to the default gfx942 table — so this still
    # returns a populated dict, proving the detector was actually invoked.
    overrides = config_module.lookup_defaults("all_reduce", 1024, arch=None)
    assert overrides == config_module._DEFAULTS_TABLE["gfx942"]["all_reduce"][0][1]


# --------------------------------------------------------------------------
# Config integration — every table row must produce a valid Config.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("coll", COLLECTIVES)
def test_default_config_is_valid_for_every_bucket(config_module, coll):
    """Every entry in the table must yield a Config that survives validation."""
    buckets = config_module._DEFAULTS_TABLE["gfx942"][coll]
    for max_bytes, _ in buckets:
        # Pick a message size in the middle of this bucket.
        size = 1024 if math.isinf(max_bytes) else max(1, int(max_bytes) - 1)
        cfg = config_module.default_config(coll, size, arch="gfx942")
        assert cfg.comm_sms > 0
        assert cfg.block_size_m > 0
        assert cfg.block_size_n > 0
        assert cfg.num_warps > 0
        # ring slice must divide block_size_n (Config validator enforces this)
        assert cfg.block_size_n % cfg.all_reduce_ring_slice_n == 0


def test_variant_field_routes_per_collective(config_module):
    """``variant`` in the override dict must map to the collective-specific
    Config field (``all_reduce_variant`` / ``all_gather_variant`` / ...)."""
    cfg = config_module.default_config("all_reduce", 8 * 1024, arch="gfx942")
    assert cfg.all_reduce_variant in {"atomic", "ring", "two_shot", "one_shot", "spinlock"}

    cfg = config_module.default_config("all_gather", 8 * 1024, arch="gfx942")
    assert cfg.all_gather_variant in {"persistent", "partitioned"}

    cfg = config_module.default_config("reduce_scatter", 8 * 1024, arch="gfx942")
    assert cfg.reduce_scatter_variant == "two_shot"


def test_default_config_rejects_unknown_table_keys(config_module, monkeypatch):
    """A typo in the defaults table (e.g. ``coomm_sms`` instead of
    ``comm_sms``) must fail loudly at ``default_config`` time rather than
    silently falling through and being lost."""
    monkeypatch.setattr(
        config_module,
        "lookup_defaults",
        lambda *a, **k: {"definitely_not_a_real_field": 1},
    )
    with pytest.raises(ValueError, match="Unknown defaults-table key"):
        config_module.default_config("all_reduce", 1024, arch="gfx942")


def test_default_config_isolates_per_collective_keys(config_module, monkeypatch):
    """``num_rings`` is meaningful only to ``all_reduce``. If it leaks into
    another collective's bucket the typo guard must catch it instead of
    silently producing an incorrect Config."""
    overrides_with_ar_only_key: dict[str, Any] = {"num_rings": 2, "comm_sms": 64}
    monkeypatch.setattr(
        config_module,
        "lookup_defaults",
        lambda *a, **k: overrides_with_ar_only_key,
    )
    # all_reduce: ``num_rings`` legally maps to ``all_reduce_num_rings``.
    cfg = config_module.default_config("all_reduce", 1024, arch="gfx942")
    assert cfg.all_reduce_num_rings == 2
    # all_gather: same key is rejected — no silent cross-collective leakage.
    with pytest.raises(ValueError, match="Unknown defaults-table key"):
        config_module.default_config("all_gather", 1024, arch="gfx942")


# --------------------------------------------------------------------------
# Public-API wiring — the four collective stubs must consult the table when
# ``config=None`` rather than falling back to their old hard-coded literals.
#
# We do this with plain text inspection (not import) so the test stays
# CPU-friendly even though the stubs themselves transitively pull in
# triton + iris kernels.
# --------------------------------------------------------------------------


def _read(stub_name: str) -> str:
    return (_CCL_DIR / f"{stub_name}.py").read_text()


@pytest.mark.parametrize("stub", ["all_reduce", "all_gather", "reduce_scatter", "all_to_all"])
def test_public_apis_route_through_default_config(stub):
    """Each collective stub must wire its ``config=None`` branch through
    ``iris.ccl.config.default_config``. A regression here would silently
    revert to the old hard-coded ``Config(...)`` defaults and defeat the
    entire point of the table.
    """
    src = _read(stub)
    assert "default_config" in src, (
        f"iris/ccl/{stub}.py no longer wires its config=None branch through "
        "default_config — this regresses the static tuning table."
    )
    # Sanity: the message-size argument must come from the input tensor,
    # not be hardcoded — otherwise every call would land in the same bucket.
    assert "input_tensor.numel() * input_tensor.element_size()" in src, (
        f"iris/ccl/{stub}.py no longer derives message_bytes from the input "
        "tensor — the table lookup will pick the wrong bucket."
    )


def test_triton_all_reduce_preamble_uses_default_config():
    """The all_reduce preamble has its own ``config=None`` branch (it can be
    called standalone to allocate a workspace) — it must also consult the
    table or the workspace will be sized for the dataclass defaults instead
    of the per-message-size sweet spot."""
    src = (_REPO_ROOT / "iris" / "ccl" / "triton" / "all_reduce.py").read_text()
    assert "default_config(" in src, "triton/all_reduce.py preamble must use default_config()"


def test_public_apis_export_default_helpers():
    """``iris.ccl`` must re-export ``default_config`` / ``lookup_defaults``
    so callers can pre-build a ``Config`` for a known message size without
    reaching into ``iris.ccl.config``."""
    src = (_CCL_DIR / "__init__.py").read_text()
    for name in ("default_config", "lookup_defaults", "Config"):
        assert name in src, f"iris.ccl.__init__ must re-export {name}"
