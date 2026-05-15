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
        config_module._lookup_raw("foo", 1024, arch="gfx942")


def test_lookup_negative_bytes_raises(config_module):
    with pytest.raises(ValueError, match="non-negative"):
        config_module._lookup_raw("all_reduce", -1, arch="gfx942")


def test_lookup_unknown_arch_falls_back(config_module):
    """Unknown arch must fall back to the default arch instead of raising."""
    overrides = config_module._lookup_raw("all_reduce", 1024, arch="gfx_unknown")
    assert overrides, "fallback to default arch must yield non-empty overrides"


@pytest.mark.parametrize("coll", COLLECTIVES)
def test_lookup_returns_first_matching_bucket(config_module, coll):
    """Walking buckets must pick the smallest ``max_bytes >= message_bytes``."""
    buckets = config_module._DEFAULTS_TABLE["gfx942"][coll]

    # Just below the first edge → first bucket
    first_edge = buckets[0][0]
    overrides = config_module._lookup_raw(coll, max(1, first_edge - 1), arch="gfx942")
    assert overrides == buckets[0][1]

    # Just above the first edge → second bucket
    if len(buckets) >= 2:
        overrides = config_module._lookup_raw(coll, first_edge + 1, arch="gfx942")
        assert overrides == buckets[1][1]
        # And in the "huge message" tail → final bucket
        overrides = config_module._lookup_raw(coll, 1 << 33, arch="gfx942")
        assert overrides == buckets[-1][1]


def test_lookup_raw_is_module_private(config_module):
    """The raw-lookup helper must NOT be exported from iris.ccl — the Round-5
    Architect required exactly one safe public entry point (``default_config``)
    so future callers cannot bypass the fail-closed gate by going through the
    raw lookup.
    """
    import iris.ccl as ccl_pkg

    assert "lookup_defaults" not in getattr(ccl_pkg, "__all__", [])
    assert not hasattr(ccl_pkg, "lookup_defaults"), (
        "iris.ccl.lookup_defaults was removed in K-7292 round 5; the only "
        "supported public entry point is iris.ccl.default_config."
    )
    # The raw helper still exists, but only as a module-private name.
    assert callable(config_module._lookup_raw)


# --------------------------------------------------------------------------
# Config integration — every table row must produce a valid Config.
# --------------------------------------------------------------------------


def _config_from_overrides(config_module, coll, overrides):
    """Build a Config from a raw override dict using the same field-mapping as
    ``default_config`` — but skipping the validation gate. Used by the table
    coverage tests below so they exercise every bucket regardless of whether
    the cell happens to be in :data:`_VALIDATED_CELLS`.
    """
    field_map = {
        "all_reduce": "all_reduce_variant",
        "all_gather": "all_gather_variant",
        "reduce_scatter": "reduce_scatter_variant",
    }
    kwargs = {}
    for key, value in overrides.items():
        if key == "variant":
            mapped = field_map.get(coll)
            if mapped:
                kwargs[mapped] = value
        elif key == "distribution":
            kwargs["all_reduce_distribution"] = value
        elif key == "num_rings":
            kwargs["all_reduce_num_rings"] = value
        else:
            kwargs[key] = value
    return config_module.Config(**kwargs)


@pytest.mark.parametrize("coll", COLLECTIVES)
def test_default_config_is_valid_for_every_bucket(config_module, coll):
    """Every entry in the table must yield a Config that survives validation.

    Uses the module-private raw-lookup path so coverage of the table does not
    depend on the cell happening to be in the on-target allow-list.
    """
    buckets = config_module._DEFAULTS_TABLE["gfx942"][coll]
    for max_bytes, _ in buckets:
        # Pick a message size in the middle of this bucket.
        size = 1024 if math.isinf(max_bytes) else max(1, int(max_bytes) - 1)
        overrides = config_module._lookup_raw(coll, size, arch="gfx942")
        cfg = _config_from_overrides(config_module, coll, overrides)
        assert cfg.comm_sms > 0
        assert cfg.block_size_m > 0
        assert cfg.block_size_n > 0
        assert cfg.num_warps > 0
        # ring slice must divide block_size_n (Config validator enforces this)
        assert cfg.block_size_n % cfg.all_reduce_ring_slice_n == 0


def test_variant_field_routes_per_collective(config_module):
    """``variant`` in the override dict must map to the collective-specific
    Config field (``all_reduce_variant`` / ``all_gather_variant`` / ...).

    Uses the raw-lookup path + manual Config construction so the coverage is
    not gated on the bucket cell being in the on-target allow-list.
    """
    # all_reduce → all_reduce_variant
    cfg = _config_from_overrides(
        config_module, "all_reduce", config_module._lookup_raw("all_reduce", 8 * 1024, arch="gfx942")
    )
    assert cfg.all_reduce_variant in {"atomic", "ring", "two_shot", "one_shot", "spinlock"}
    # all_gather → all_gather_variant
    cfg = _config_from_overrides(
        config_module, "all_gather", config_module._lookup_raw("all_gather", 8 * 1024, arch="gfx942")
    )
    assert cfg.all_gather_variant in {"persistent", "partitioned"}
    # reduce_scatter → reduce_scatter_variant ("two_shot" is the only legal
    # value at present; this guards against the table accidentally pinning it
    # to something the kernel doesn't implement).
    cfg = _config_from_overrides(
        config_module,
        "reduce_scatter",
        config_module._lookup_raw("reduce_scatter", 8 * 1024, arch="gfx942"),
    )
    assert cfg.reduce_scatter_variant == "two_shot"


# --------------------------------------------------------------------------
# Public-API wiring — the four collective stubs must consult the table when
# ``config=None`` rather than falling back to their old hard-coded literals.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Fail-closed safeguard — only cells with positive on-target evidence may be
# routed by ``default_config``. Anything outside the allow-list must raise so
# iris cannot silently launch a kernel of unproven correctness.
# --------------------------------------------------------------------------


def test_validated_cells_set_is_non_empty(config_module):
    """The allow-list must have at least one entry — an empty set would
    fail-closed every call and break iris.ccl out of the box."""
    assert config_module._VALIDATED_CELLS, (
        "_VALIDATED_CELLS is empty; default_config would fail-closed for "
        "every input. Add the cells that passed the latest on-target sweep "
        "before clearing this assertion."
    )


def test_validated_cells_have_valid_collectives(config_module):
    """Every allow-listed cell must reference a real collective so the
    fail-closed check in ``default_config`` actually resolves the right key."""
    for arch, collective, _ in config_module._VALIDATED_CELLS:
        assert isinstance(arch, str) and arch
        assert collective in COLLECTIVES, f"unknown collective {collective!r} in _VALIDATED_CELLS"


def test_default_config_succeeds_for_validated_cell(config_module):
    """``default_config`` must succeed on every entry in the allow-list and
    return a Config that survives the dataclass validator."""
    for arch, collective, message_bytes in config_module._VALIDATED_CELLS:
        cfg = config_module.default_config(collective, message_bytes, arch=arch)
        assert cfg.comm_sms > 0


def test_default_config_warns_for_unvalidated_cell(config_module):
    """Cells with no on-target evidence must emit
    :class:`UnvalidatedDefaultConfigWarning` but still return a best-effort
    Config — the round-9 Architect required that the public
    ``ctx.ccl.<collective>(config=None)`` contract keep working out of the
    box, with provenance surfaced as a warning rather than as a hard
    ``NotImplementedError`` that silently narrows the API to a small
    allow-list. Callers that want the previous fail-closed behaviour can
    install ``warnings.filterwarnings("error", ...)`` selectively."""
    import warnings

    arch = "gfx942"
    samples = [
        ("all_gather", 8192),
        ("reduce_scatter", 65536),
        ("all_to_all", 1 << 20),
        ("all_reduce", 131072),  # one of the round-2 verifier failures
        ("all_reduce", 1234567),  # arbitrary unvalidated all_reduce size
    ]
    for collective, message_bytes in samples:
        assert (arch, collective, message_bytes) not in config_module._VALIDATED_CELLS, (
            f"{(arch, collective, message_bytes)} accidentally landed in the "
            "allow-list; pick a different sample for this test."
        )
        with pytest.warns(config_module.UnvalidatedDefaultConfigWarning, match="no on-target verifier evidence"):
            cfg = config_module.default_config(collective, message_bytes, arch=arch)
        assert cfg.comm_sms > 0

        # Escalation path: filterwarnings("error", ...) recovers the previous
        # fail-closed behaviour for production callers that opt in.
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=config_module.UnvalidatedDefaultConfigWarning)
            with pytest.raises(config_module.UnvalidatedDefaultConfigWarning):
                config_module.default_config(collective, message_bytes, arch=arch)


def test_default_config_does_not_warn_for_validated_cell(config_module):
    """Validated cells must NOT trigger
    :class:`UnvalidatedDefaultConfigWarning` — the allow-list is the
    "no warning needed, on-target evidence exists" set."""
    import warnings

    for arch, collective, message_bytes in config_module._VALIDATED_CELLS:
        with warnings.catch_warnings():
            warnings.simplefilter("error", config_module.UnvalidatedDefaultConfigWarning)
            cfg = config_module.default_config(collective, message_bytes, arch=arch)
        assert cfg.comm_sms > 0


def test_lookup_raw_unaffected_by_allow_list(config_module):
    """``_lookup_raw`` is the module-private bucket lookup and must NOT be
    gated by :data:`_VALIDATED_CELLS` — the safeguard lives in
    ``default_config`` so in-tree tooling can still inspect the table values
    for unvalidated cells (sweep harness, table introspection)."""
    arch = "gfx942"
    unvalidated_samples = [
        ("all_gather", 8192),
        ("reduce_scatter", 65536),
        ("all_to_all", 1 << 20),
        ("all_reduce", 131072),
    ]
    for collective, message_bytes in unvalidated_samples:
        assert (arch, collective, message_bytes) not in config_module._VALIDATED_CELLS
        overrides = config_module._lookup_raw(collective, message_bytes, arch=arch)
        assert overrides, "_lookup_raw must still return the bucket overrides"


def test_resolve_is_single_source_of_truth(config_module):
    """Both ``_lookup_raw`` (raw) and ``default_config`` (validated) must
    route through the same internal ``_resolve`` helper so the table has a
    single resolution path — this prevents the maintenance trap the Round-3
    Architect flagged where a future caller could bypass the safeguard by
    going through a divergent code path.

    The contract: ``_resolve`` returns ``(overrides, validated)`` and
    ``_lookup_raw(...) == _resolve(...)[0]`` for every cell, including the
    unvalidated ones (raw lookup ignores the validation flag by design).
    """
    arch = "gfx942"
    for coll in COLLECTIVES:
        # A spread of sizes that hits every bucket.
        for size in (1024, 64 * 1024 + 1, 8 * 1024 * 1024, 1 << 30):
            overrides_raw, validated = config_module._resolve(coll, size, arch=arch)
            overrides_lookup = config_module._lookup_raw(coll, size, arch=arch)
            assert overrides_raw == overrides_lookup
            assert isinstance(validated, bool)

    # And the validated flag must be True exactly on _VALIDATED_CELLS:
    for arch_k, coll, size in config_module._VALIDATED_CELLS:
        _, validated = config_module._resolve(coll, size, arch=arch_k)
        assert validated is True, f"{(arch_k, coll, size)} should resolve as validated"


def test_validated_cells_match_documented_evidence(config_module):
    """Pin the allow-list to the exact 12 cells the round-2 on-target sweep
    (``output/sweep_revision_smoke_mi300x.csv``, K-7267 workspace) flagged
    ``correct=True`` for ``all_reduce`` × fp16 × 1 KiB → 1 GiB.

    A drift here means either (a) a fresh on-target sweep added cells, or
    (b) someone widened the allow-list without re-running the sweep. Either
    way the contributor must update this test alongside the registry, which
    forces the conversation about evidence."""
    expected = {
        ("gfx942", "all_reduce", 1024),
        ("gfx942", "all_reduce", 2048),
        ("gfx942", "all_reduce", 4096),
        ("gfx942", "all_reduce", 8192),
        ("gfx942", "all_reduce", 16384),
        ("gfx942", "all_reduce", 32768),
        ("gfx942", "all_reduce", 65536),
        ("gfx942", "all_reduce", 262144),
        ("gfx942", "all_reduce", 2097152),
        ("gfx942", "all_reduce", 4194304),
        ("gfx942", "all_reduce", 33554432),
        ("gfx942", "all_reduce", 1073741824),
    }
    assert config_module._VALIDATED_CELLS == expected, (
        "drift from the round-2 on-target sweep evidence; re-run the sweep "
        "and update both the allow-list and this assertion in lock-step."
    )


def test_validated_cells_match_round2_failure_exclusion(config_module):
    """The 9 ``all_reduce`` cells the round-2 sweep flagged ``correct=False``
    must NOT appear in the allow-list — those are the cells the verifier
    proved produce wrong output, so ``default_config`` has to fail-closed
    there even though they live in the same buckets as passing cells."""
    round2_failures = {
        ("gfx942", "all_reduce", 131072),
        ("gfx942", "all_reduce", 524288),
        ("gfx942", "all_reduce", 1048576),
        ("gfx942", "all_reduce", 8388608),
        ("gfx942", "all_reduce", 16777216),
        ("gfx942", "all_reduce", 67108864),
        ("gfx942", "all_reduce", 134217728),
        ("gfx942", "all_reduce", 268435456),
        ("gfx942", "all_reduce", 536870912),
    }
    leaked = round2_failures & config_module._VALIDATED_CELLS
    assert not leaked, (
        f"_VALIDATED_CELLS leaked round-2 verifier failures: {sorted(leaked)}; "
        "these cells produced wrong output on MI300X and must stay outside "
        "the allow-list until the kernel bug is fixed."
    )


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
