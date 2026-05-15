# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Validation gate for the iris.ccl static defaults table.

The defaults table in :mod:`iris.ccl.config` is a pure piecewise-constant
``(arch, collective, message_size_bucket)`` → kernel-knob lookup. This
module owns the **separate** concern of "has a given cell been proven
correct on real hardware?" — the round-10 Architect required these two
concerns live in distinct modules so the lookup table cannot accidentally
be mistaken for the policy decision about unvalidated cells.

Responsibilities exposed here:

- :class:`UnvalidatedDefaultConfigWarning` — the warning subclass
  ``warn_if_unvalidated`` emits and that production callers can escalate
  via ``warnings.filterwarnings("error", ...)``.
- :data:`_VALIDATED_CELLS` — the registry of cells with positive
  on-target evidence (round-2 ``output/sweep_revision_smoke_mi300x.csv``
  in workspace K-7267).
- :func:`is_validated` — pure predicate; safe to call from anywhere
  (tests, sweep harness, telemetry, ...) without side effects.
- :func:`warn_if_unvalidated` — the policy hook each public collective
  call site invokes explicitly so the "warn-and-pray" decision is
  visible at the four entry points instead of buried inside a generic
  helper. Callers that already know the cell is validated (e.g. the
  sweep harness re-running a verified row) can skip the call entirely.

Keeping the gate out of :mod:`iris.ccl.config` means downstream consumers
of the lookup table (sweep harness, table-introspection tests, future
analyzers) never inherit an undocumented warn-and-pray contract — they
opt into the policy by importing it from this module.
"""

import warnings


class UnvalidatedDefaultConfigWarning(UserWarning):
    """Raised when a caller resolves a defaults-table cell with no on-target evidence.

    Surfacing the missing-evidence signal as a dedicated warning subclass lets
    production callers either ignore it (default Python behaviour, preserves
    the "out of the box" PRD contract) or escalate it to an error via
    ``warnings.filterwarnings("error", category=UnvalidatedDefaultConfigWarning)``
    to recover a fail-closed behaviour selectively.
    """


# ── Registry of cells with positive on-target evidence ────────────────────
#
# ``benchmark/ccl/comprehensive_sweep.py`` (round 2 evidence,
# ``output/sweep_revision_smoke_mi300x.csv`` in workspace K-7267) runs every
# iris cell through ``_compare_to_reference`` against the analytical reduction
# answer. The set below enumerates exactly the ``(arch, collective, bytes)``
# cells that **passed** that verifier on MI300X — i.e. the cells for which we
# can prove the table-selected variant produces correct output. Cells outside
# this set (every ``all_gather`` / ``reduce_scatter`` / ``all_to_all`` cell,
# every bf16 cell, plus the 9 ``all_reduce`` × fp16 cells the round-2 sweep
# flagged ``correct=False``) are currently unvalidated.
_VALIDATED_CELLS: set[tuple[str, str, int]] = {
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


def is_validated(arch: str, collective: str, message_bytes: int) -> bool:
    """Pure predicate: has this cell been proven correct on real hardware?

    Args:
        arch: Architecture string (``gfx942`` ...).
        collective: One of ``all_reduce``, ``all_gather``, ``reduce_scatter``,
            ``all_to_all``.
        message_bytes: Per-rank input tensor size in bytes (the same key the
            sweep harness uses to record evidence).

    Returns:
        True iff ``(arch, collective, message_bytes)`` is in
        :data:`_VALIDATED_CELLS`.
    """
    return (arch, collective, message_bytes) in _VALIDATED_CELLS


def warn_if_unvalidated(arch: str, collective: str, message_bytes: int) -> None:
    """Emit :class:`UnvalidatedDefaultConfigWarning` for unvalidated cells.

    This is the explicit policy hook the four public collective entry points
    invoke before falling back to ``iris.ccl.config.default_config``: keeping
    the warn at the call site (instead of inside the generic
    ``default_config`` helper) means the warn-and-pray contract is visible
    in each of the four files a reviewer audits, and downstream callers of
    the raw lookup never inherit it implicitly.

    Args:
        arch: Architecture string (``gfx942`` ...).
        collective: One of ``all_reduce``, ``all_gather``, ``reduce_scatter``,
            ``all_to_all``.
        message_bytes: Per-rank input tensor size in bytes.
    """
    if is_validated(arch, collective, message_bytes):
        return
    warnings.warn(
        f"iris.ccl.{collective} default Config for {message_bytes} B on {arch} "
        "has no on-target verifier evidence: this cell is not in "
        "iris.ccl.validation._VALIDATED_CELLS. The collective will run with "
        "the best-effort table-selected Config; pass an explicit Config(...) "
        "or extend the allow-list with fresh on-target evidence "
        "(benchmark/ccl/comprehensive_sweep.py) to silence this warning. "
        "See iris/ccl/validation.py::_VALIDATED_CELLS and output/revision-notes.md "
        "for the validation matrix.",
        UnvalidatedDefaultConfigWarning,
        stacklevel=3,
    )
