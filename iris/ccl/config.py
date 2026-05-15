# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Configuration structures for iris-ccl collective operations.

This module exposes :class:`Config` (the public knob structure) plus a static
per-architecture defaults table consulted by every collective when the user
calls ``ctx.ccl.<collective>(...)`` without an explicit ``config`` override.

The table maps ``(arch, collective, message_size_bucket)`` to a small set of
kernel knobs (variant, ``comm_sms``, block sizes, ``num_warps``, ...). Buckets
are right-edge inclusive; lookups walk the buckets in order and take the
first whose ``max_bytes`` is ``>=`` the requested message size, so the table
is effectively a piecewise-constant map.

The default values below were produced by the canonical sweep harness at
``benchmark/ccl/comprehensive_sweep.py --mode tune`` on an MI300X (gfx942)
8-rank node and committed alongside the harness so the table can be
reproduced from source.
"""

from dataclasses import dataclass
from typing import Any

# NOTE: ``iris`` is intentionally NOT imported at module load time. The
# defaults table and lookup helpers are pure-Python and must remain
# importable on CPU-only hosts (CI, doc builders, ...) where loading the
# top-level ``iris`` package would pull in ``libamdhip64.so`` and fail.
# ``Config.__post_init__`` performs a deferred ``import iris`` for the only
# real iris dependency (``iris.hip.get_num_xcc``).


# ── Architecture detection ────────────────────────────────────────────────


_DEFAULT_ARCH = "gfx942"  # MI300X
_KNOWN_ARCHS = ("gfx942", "gfx950", "gfx90a")


def _detect_arch() -> str:
    """Return a short architecture string (``gfx942``, ``gfx950`` ...) for the local GPU.

    Falls back to ``_DEFAULT_ARCH`` when auto-detection fails (CPU-only env
    or pre-init contexts), so the table lookup always resolves.
    """
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            name = (getattr(props, "gcnArchName", "") or "").lower()
            for stem in _KNOWN_ARCHS:
                if stem in name:
                    return stem
    except Exception:
        pass
    return _DEFAULT_ARCH


# ── Static defaults table ──────────────────────────────────────────────────


# Each list entry is a bucket: ``(max_bytes, params)``. A request for
# ``message_bytes`` selects the first bucket with ``max_bytes >=
# message_bytes``; the trailing bucket uses ``float('inf')`` as the catch-all
# upper edge.
#
# Tunables (only the keys present in the bucket override the corresponding
# Config field — every other field keeps its dataclass default):
#
# - ``variant``           — collective-specific variant name
# - ``comm_sms``          — number of SMs assigned to the comm kernel
# - ``block_size_m``      — M-dim tile size
# - ``block_size_n``      — N-dim tile size
# - ``num_warps``         — warps per workgroup
# - ``swizzle_size``      — chiplet-aware swizzle group
# - ``distribution``      — two-shot all-reduce / reduce-scatter row layout
# - ``num_rings``         — concurrent rings for ring all-reduce
#
# Buckets reflect the empirically observed sweet-spot transitions on
# MI300X: small messages (≤ 64 KiB) prefer one-shot/atomic algorithms with
# fewer SMs because launch + barrier latency dominates; medium messages
# (64 KiB ... 4 MiB) move to two-shot and a wider tile; large messages
# (≥ 4 MiB) saturate the XGMI fabric and prefer the maximum SM count with
# wide tiles to amortize per-iteration bookkeeping.
_KIB = 1024
_MIB = 1024 * 1024
_GIB = 1024 * 1024 * 1024


_DEFAULTS_TABLE: dict[str, dict[str, list[tuple[float, dict[str, Any]]]]] = {
    "gfx942": {
        # all_reduce: tuner picks one_shot with small tiles up to ~1 MiB then
        # switches to two_shot. comm_sms drops as message grows because the
        # work-per-tile rises and over-subscription only adds barrier cost.
        "all_reduce": [
            (
                _MIB,
                {
                    "variant": "one_shot",
                    "comm_sms": 128,
                    "block_size_m": 8,
                    "block_size_n": 32,
                    "num_warps": 4,
                    "swizzle_size": 4,
                    "distribution": 1,
                },
            ),
            (
                16 * _MIB,
                {
                    "variant": "two_shot",
                    "comm_sms": 128,
                    "block_size_m": 8,
                    "block_size_n": 32,
                    "num_warps": 4,
                    "swizzle_size": 4,
                    "distribution": 1,
                },
            ),
            (
                float("inf"),
                {
                    "variant": "two_shot",
                    "comm_sms": 64,
                    "block_size_m": 8,
                    "block_size_n": 32,
                    "num_warps": 4,
                    "swizzle_size": 4,
                    "distribution": 1,
                },
            ),
        ],
        # all_gather: partitioned wins at small/medium sizes, persistent wins
        # for the bandwidth-bound tail (≥ 8 MiB).
        "all_gather": [
            (
                64 * _KIB,
                {
                    "variant": "persistent",
                    "comm_sms": 128,
                    "block_size_m": 8,
                    "block_size_n": 128,
                    "num_warps": 4,
                    "swizzle_size": 4,
                },
            ),
            (
                _MIB,
                {
                    "variant": "partitioned",
                    "comm_sms": 128,
                    "block_size_m": 8,
                    "block_size_n": 128,
                    "num_warps": 4,
                    "swizzle_size": 4,
                },
            ),
            (
                8 * _MIB,
                {
                    "variant": "partitioned",
                    "comm_sms": 128,
                    "block_size_m": 32,
                    "block_size_n": 256,
                    "num_warps": 4,
                    "swizzle_size": 4,
                },
            ),
            (
                float("inf"),
                {
                    "variant": "persistent",
                    "comm_sms": 128,
                    "block_size_m": 32,
                    "block_size_n": 32,
                    "num_warps": 4,
                    "swizzle_size": 4,
                },
            ),
        ],
        # reduce_scatter: only "two_shot" exists; mirror the all_reduce
        # tile-size sweet spots since both kernels do the same per-tile
        # reduction work.
        "reduce_scatter": [
            (
                _MIB,
                {
                    "variant": "two_shot",
                    "comm_sms": 128,
                    "block_size_m": 8,
                    "block_size_n": 64,
                    "num_warps": 4,
                    "swizzle_size": 4,
                    "distribution": 1,
                },
            ),
            (
                16 * _MIB,
                {
                    "variant": "two_shot",
                    "comm_sms": 128,
                    "block_size_m": 16,
                    "block_size_n": 128,
                    "num_warps": 4,
                    "swizzle_size": 4,
                    "distribution": 1,
                },
            ),
            (
                float("inf"),
                {
                    "variant": "two_shot",
                    "comm_sms": 128,
                    "block_size_m": 32,
                    "block_size_n": 128,
                    "num_warps": 4,
                    "swizzle_size": 4,
                    "distribution": 1,
                },
            ),
        ],
        # all_to_all: persistent kernel issues per-peer payloads, so wider
        # block_size_n vectorizes the per-peer copy. Larger messages need
        # more SMs to overlap the 7 peer streams.
        "all_to_all": [
            (
                64 * _KIB,
                {
                    "comm_sms": 64,
                    "block_size_m": 4,
                    "block_size_n": 128,
                    "num_warps": 4,
                    "swizzle_size": 4,
                },
            ),
            (
                4 * _MIB,
                {
                    "comm_sms": 128,
                    "block_size_m": 8,
                    "block_size_n": 256,
                    "num_warps": 4,
                    "swizzle_size": 4,
                },
            ),
            (
                float("inf"),
                {
                    "comm_sms": 256,
                    "block_size_m": 16,
                    "block_size_n": 256,
                    "num_warps": 4,
                    "swizzle_size": 4,
                },
            ),
        ],
    },
}


_VALID_COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")


def lookup_defaults(collective: str, message_bytes: int, arch: str | None = None) -> dict[str, Any]:
    """Look up the static defaults for ``(arch, collective, message_bytes)``.

    Args:
        collective: One of :data:`_VALID_COLLECTIVES`.
        message_bytes: Per-rank input tensor size in bytes.
        arch: Architecture string (``gfx942`` ...). When ``None``, auto-detected
            via :func:`_detect_arch`; falls back to :data:`_DEFAULT_ARCH` if the
            architecture is not present in the table.

    Returns:
        The override dict for the smallest bucket whose ``max_bytes`` is
        ``>= message_bytes``. An empty dict is returned if the collective
        is unknown — callers should treat that as "use Config dataclass
        defaults".

    Raises:
        ValueError: If ``collective`` is not one of the supported names.
    """
    if collective not in _VALID_COLLECTIVES:
        raise ValueError(f"Unknown collective {collective!r}; expected one of {_VALID_COLLECTIVES}")

    if message_bytes < 0:
        raise ValueError(f"message_bytes must be non-negative, got {message_bytes}")

    arch_key = arch or _detect_arch()
    table = _DEFAULTS_TABLE.get(arch_key) or _DEFAULTS_TABLE.get(_DEFAULT_ARCH)
    if not table:
        return {}
    buckets = table.get(collective)
    if not buckets:
        return {}
    for max_bytes, overrides in buckets:
        if message_bytes <= max_bytes:
            return dict(overrides)
    # Fallthrough: use the last bucket.
    return dict(buckets[-1][1])


# Per-collective mapping from "table key" → "Config dataclass field". Keys
# that don't appear in a collective's mapping (and aren't in
# :data:`_COMMON_PASSTHROUGH`) are not silently ignored — ``default_config``
# raises so a typo in the defaults table (e.g. ``coomm_sms``) is caught at
# lookup time, not lost. This also prevents an ``all_reduce``-only override
# (such as ``num_rings``) from accidentally being applied to ``all_gather``.
_FIELD_MAP: dict[str, dict[str, str]] = {
    "all_reduce": {
        "variant": "all_reduce_variant",
        "distribution": "all_reduce_distribution",
        "num_rings": "all_reduce_num_rings",
        "ring_slice_n": "all_reduce_ring_slice_n",
    },
    "all_gather": {
        "variant": "all_gather_variant",
    },
    "reduce_scatter": {
        "variant": "reduce_scatter_variant",
        # The two-shot reduce-scatter kernel reads
        # ``config.all_reduce_distribution`` (see
        # iris/ccl/triton/reduce_scatter.py) so we expose the same
        # ``distribution`` alias here.
        "distribution": "all_reduce_distribution",
    },
    "all_to_all": {},
}

# Keys that map straight onto identically-named ``Config`` fields and are
# accepted by every collective.
_COMMON_PASSTHROUGH = frozenset(
    {
        "block_size_m",
        "block_size_n",
        "swizzle_size",
        "comm_sms",
        "num_xcds",
        "chunk_size",
        "use_gluon",
        "num_stages",
        "num_warps",
        "threads_per_warp",
        "waves_per_eu",
    }
)


def default_config(collective: str, message_bytes: int, arch: str | None = None) -> "Config":
    """Build a :class:`Config` populated from the static defaults table.

    Args:
        collective: Collective name (see :data:`_VALID_COLLECTIVES`).
        message_bytes: Per-rank input tensor size in bytes; selects the bucket.
        arch: Optional architecture override. Auto-detected if ``None``.

    Returns:
        A :class:`Config` instance with the table-selected overrides applied.
        Caller may further mutate the returned object before handing it to a
        kernel launch; the dataclass is mutable by design.

    Raises:
        ValueError: If ``collective`` is unknown or if the table contains a
            key that doesn't map to any known ``Config`` field for that
            collective (typo guard for ``_DEFAULTS_TABLE``).
    """
    overrides = lookup_defaults(collective, message_bytes, arch=arch)
    per_collective = _FIELD_MAP[collective]  # collective already validated above
    kwargs: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in per_collective:
            kwargs[per_collective[key]] = value
        elif key in _COMMON_PASSTHROUGH:
            kwargs[key] = value
        else:
            raise ValueError(
                f"Unknown defaults-table key {key!r} for collective {collective!r}; "
                "expected a Config field name or one of the per-collective aliases "
                f"({sorted(per_collective)})."
            )
    return Config(**kwargs)


@dataclass
class Config:
    """
    Configuration parameters for iris-ccl collective operations.

    This configuration struct encapsulates common kernel parameters that can be
    set once and reused across multiple collective calls, similar to the
    origami config pattern from ROCm libraries. When a user invokes a
    collective without supplying an explicit ``Config``, the public API
    consults a static defaults table (see :func:`default_config`) keyed by
    architecture, collective, and per-rank message-size bucket — the values
    in that table were tuned on MI300X (gfx942) using
    ``benchmark/ccl/comprehensive_sweep.py``.

    Args:
        block_size_m: Block size for the M dimension tiling (default: 32).
        block_size_n: Block size for the N dimension tiling (default: 64).
        swizzle_size: Number of tiles to swizzle/group together for
                     better memory access patterns (default: 4).
        comm_sms: Number of SMs (Streaming Multiprocessors) to use for
                 the communication kernel (default: 64).
        num_xcds: Number of XCCs. If None, auto-detected from system (default: None).
        chunk_size: Number of tiles per chiplet chunk; auto-derived from
                    ``swizzle_size`` and ``comm_sms`` when None (default: None).
        use_gluon: If True, use Gluon-based implementation (default: False).
                   Gluon provides better control over warp-level traffic shaping.
        all_gather_variant: Variant for all-gather operation (default: "persistent").
                            Options: "persistent", "partitioned".
                           - "persistent": Each PID handles multiple tiles and sends to all ranks
                           - "partitioned": PIDs partitioned across ranks, eliminates inner loop
        all_reduce_variant: Variant for all-reduce operation (default: "two_shot").
                            Options: "atomic", "ring", "two_shot", "one_shot", "spinlock".
        all_reduce_distribution: Distribution for two-shot all-reduce (default: 1).
                                 0 for striding, 1 for block distribution.
        all_reduce_num_rings: Concurrent rings in ring-based all-reduce (default: 1).
        all_reduce_ring_slice_n: Column slice size for ring reduce-scatter/all-gather
                                 (default: auto = ``block_size_n``).
        reduce_scatter_variant: Variant for reduce-scatter operation (default: "two_shot").
                                Only "two_shot" is supported.
        num_stages: Number of pipeline stages for the kernel (default: 1).
        num_warps: Number of warps per workgroup (default: 4). For gluon kernels,
                   this also sets WARPS_PER_CTA in the BlockedLayout. The product
                   threads_per_warp * num_warps determines the minimum tile size
                   (block_size_m * block_size_n for flat-2D, or block_size_n for 1D).
        threads_per_warp: Threads per warp/wavefront (default: 64). Must match the
                          hardware wavefront size: 64 for AMD GPUs, 32 for NVIDIA.
        waves_per_eu: Waves per execution unit hint for occupancy (default: 0, auto).

    Example:
        >>> import iris
        >>> from iris.ccl import Config
        >>> ctx = iris.iris()
        >>> config = Config(
        ...     block_size_m=128,
        ...     block_size_n=32,
        ...     swizzle_size=8,
        ...     comm_sms=64,
        ...     use_gluon=True
        ... )
        >>> ctx.ccl.all_to_all(output_tensor, input_tensor, config=config)

        >>> # All-reduce with ring variant
        >>> config = Config(all_reduce_variant="ring")
        >>> ctx.ccl.all_reduce(output_tensor, input_tensor, config=config)

        >>> # All-gather with partitioned variant
        >>> config = Config(all_gather_variant="partitioned")
        >>> ctx.ccl.all_gather(output_tensor, input_tensor, config=config)
    """

    block_size_m: int = 32
    block_size_n: int = 64
    swizzle_size: int = 4
    comm_sms: int = 64
    num_xcds: int | None = None
    chunk_size: int | None = None
    use_gluon: bool = False
    all_gather_variant: str = "persistent"
    all_reduce_variant: str = "two_shot"
    all_reduce_distribution: int = 1
    all_reduce_num_rings: int = 1
    all_reduce_ring_slice_n: int | None = None
    reduce_scatter_variant: str = "two_shot"
    num_stages: int = 1
    num_warps: int = 4
    threads_per_warp: int = 64
    waves_per_eu: int = 0

    def __post_init__(self):
        """Validate and auto-detect num_xcds if not set."""
        if self.num_xcds is None:
            # Deferred import: keeps ``iris.ccl.config`` importable on
            # CPU-only hosts where ``import iris`` would dlopen
            # ``libamdhip64.so``. The default-Config code path that needs
            # the XCC count already runs on a real GPU node.
            import iris  # noqa: WPS433 - intentional lazy import

            self.num_xcds = iris.hip.get_num_xcc()

        if self.chunk_size is None:
            self.chunk_size = self.swizzle_size * self.swizzle_size
            self.chunk_size = min(self.chunk_size, self.comm_sms // self.num_xcds)

        if self.block_size_m <= 0:
            raise ValueError(f"block_size_m must be positive, got {self.block_size_m}")
        if self.block_size_n <= 0:
            raise ValueError(f"block_size_n must be positive, got {self.block_size_n}")
        if self.swizzle_size <= 0:
            raise ValueError(f"swizzle_size must be positive, got {self.swizzle_size}")
        if self.comm_sms <= 0:
            raise ValueError(f"comm_sms must be positive, got {self.comm_sms}")
        if self.num_xcds <= 0:
            raise ValueError(f"num_xcds must be positive, got {self.num_xcds}")
        if self.all_gather_variant not in ["persistent", "partitioned"]:
            raise ValueError(
                f"all_gather_variant must be one of: 'persistent', 'partitioned', got {self.all_gather_variant}"
            )
        if self.all_reduce_variant not in ["atomic", "ring", "two_shot", "one_shot", "spinlock"]:
            raise ValueError(
                f"all_reduce_variant must be one of: 'atomic', 'ring', 'two_shot', 'one_shot', 'spinlock', got {self.all_reduce_variant}"
            )
        if self.all_reduce_distribution not in [0, 1]:
            raise ValueError(
                f"all_reduce_distribution must be 0 (striding) or 1 (block), got {self.all_reduce_distribution}"
            )
        if self.all_reduce_num_rings <= 0:
            raise ValueError(f"all_reduce_num_rings must be positive, got {self.all_reduce_num_rings}")
        if self.all_reduce_ring_slice_n is None:
            self.all_reduce_ring_slice_n = self.block_size_n
        if self.all_reduce_ring_slice_n <= 0:
            raise ValueError(f"all_reduce_ring_slice_n must be positive, got {self.all_reduce_ring_slice_n}")
        if self.block_size_n % self.all_reduce_ring_slice_n != 0:
            raise ValueError(
                f"all_reduce_ring_slice_n must divide block_size_n "
                f"(block_size_n={self.block_size_n}, slice={self.all_reduce_ring_slice_n})"
            )
        if self.all_reduce_ring_slice_n & (self.all_reduce_ring_slice_n - 1):
            raise ValueError(f"all_reduce_ring_slice_n must be a power of two, got {self.all_reduce_ring_slice_n}")

        # Validate reduce_scatter_variant
        if self.reduce_scatter_variant != "two_shot":
            raise ValueError(f"reduce_scatter_variant must be 'two_shot', got '{self.reduce_scatter_variant}'")

        if self.threads_per_warp not in (32, 64):
            raise ValueError(f"threads_per_warp must be 32 (NVIDIA) or 64 (AMD), got {self.threads_per_warp}")
        if self.num_warps <= 0:
            raise ValueError(f"num_warps must be positive, got {self.num_warps}")
