# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Unit tests for the static per-(arch, collective, message-size) defaults table
that powers ``iris.ccl.{all_reduce, all_gather, reduce_scatter, all_to_all}``
when the user calls them without an explicit ``Config``.

These tests intentionally don't require a GPU: the defaults table is a pure
Python lookup and ``Config.__post_init__`` only needs ``iris.hip.get_num_xcc``,
which is patched here. Skip-importing the GPU back-ends keeps the unit-test
job lightweight and lets CI catch regressions in the table without
re-allocating a node.
"""

from __future__ import annotations

import math
import sys
import types

import pytest


# --------------------------------------------------------------------------
# Optional-dependency stubs
# --------------------------------------------------------------------------


# ``iris/__init__.py`` eagerly imports ``iris.ops``, which in turn requires
# ``tritonblas``. The defaults table is a pure-Python lookup that doesn't
# touch any of that — install a sentinel module so the import chain
# resolves on hosts without tritonblas (CPU CI, doc builders, ...).
class _PermissiveModule(types.ModuleType):
    """A ``types.ModuleType`` that returns a sentinel for any attribute.

    ``iris/__init__.py`` eagerly imports ``iris.ops``, which transitively
    pulls a handful of ``tritonblas.kernels.stages.*`` symbols (``GemmContext``,
    ``ScheduleContext``, ``make_tensor_view``, ``Tile``, ...). The unit tests
    here don't touch any of that, so a permissive stub avoids requiring the
    real ``tritonblas`` install on CPU CI hosts.
    """

    def __getattr__(self, name):  # noqa: D401 - sentinel
        # Any unknown name resolves to a no-op class so ``from X import Y``
        # succeeds without us enumerating the API surface up front. Skip
        # dunders so introspection (``__file__``, ``__path__``, ...) takes
        # the normal AttributeError path expected by importlib + inspect.
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


# --------------------------------------------------------------------------
# Test fixtures
# --------------------------------------------------------------------------

# We lazily import iris.ccl.config inside fixtures so that the iris.hip.get_num_xcc
# stub can be applied via monkeypatch *before* Config validation runs.


@pytest.fixture(autouse=True)
def _stub_xcc(monkeypatch):
    """Patch ``iris.hip.get_num_xcc`` so Config validation works on CI hosts."""
    import iris  # noqa: WPS433 - import inside fixture is intentional

    monkeypatch.setattr(iris.hip, "get_num_xcc", lambda *args, **kwargs: 8)
    yield


@pytest.fixture
def config_module():
    from iris.ccl import config as cfg

    return cfg


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
        second_edge = buckets[1][0]
        overrides = config_module.lookup_defaults(coll, first_edge + 1, arch="gfx942")
        assert overrides == buckets[1][1]
        # And in the "huge message" tail → final bucket
        overrides = config_module.lookup_defaults(coll, 1 << 33, arch="gfx942")
        assert overrides == buckets[-1][1]


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
    # all_reduce → all_reduce_variant
    cfg = config_module.default_config("all_reduce", 8 * 1024, arch="gfx942")
    assert cfg.all_reduce_variant in {"atomic", "ring", "two_shot", "one_shot", "spinlock"}
    # all_gather → all_gather_variant
    cfg = config_module.default_config("all_gather", 8 * 1024, arch="gfx942")
    assert cfg.all_gather_variant in {"persistent", "partitioned"}
    # reduce_scatter → reduce_scatter_variant ("two_shot" is the only legal
    # value at present; this guards against the table accidentally pinning it
    # to something the kernel doesn't implement).
    cfg = config_module.default_config("reduce_scatter", 8 * 1024, arch="gfx942")
    assert cfg.reduce_scatter_variant == "two_shot"


# --------------------------------------------------------------------------
# Public-API wiring — the four collective stubs must consult the table when
# ``config=None`` rather than falling back to their old hard-coded literals.
# --------------------------------------------------------------------------


def test_public_apis_import_default_config():
    """The four public-API stubs must reference ``default_config``.

    A regression where one of them silently went back to the old hard-coded
    ``Config(block_size_m=32, block_size_n=64, ...)`` literal would defeat
    the entire purpose of this PR. Source-string sanity check guards against
    that — much cheaper than a GPU integration test.
    """
    import inspect
    from iris.ccl import all_reduce as ar
    from iris.ccl import all_gather as ag
    from iris.ccl import reduce_scatter as rs
    from iris.ccl import all_to_all as a2a

    for mod in (ar, ag, rs, a2a):
        src = inspect.getsource(mod)
        assert "default_config" in src, (
            f"{mod.__name__} no longer wires its config=None branch through "
            "iris.ccl.config.default_config — this regresses the static tuning "
            "table introduced for K-7224."
        )
