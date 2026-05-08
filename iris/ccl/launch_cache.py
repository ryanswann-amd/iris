# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Per-Config fused-launch fastpath cache for iris.ccl collectives.

Background
----------
K-786 sub-decomposed the small-message ``launch_us`` bucket of all four
iris.ccl collectives (all_reduce / all_gather / reduce_scatter / all_to_all)
into 5 host-side sub-phases::

    py_wrapper -> cache_lookup -> hip_enqueue -> wrapper_exit -> stream_record

K-786's R-PY-WRAPPER-FUSE (~+8 us) and R-TRITON-CACHE-FASTPATH (~+14.4 us
per call, constant across collectives) together account for the majority
of host-side overhead for sizes <=64 KB per rank.

K-820 collapsed both sub-phases for ``iris.ccl.all_reduce`` (two_shot
variant) by caching the resolved (bound kernel launcher, args, kwargs)
tuple per (M, N, dtype) on the user-supplied :class:`Config`.  This module
ports the same cache shape to the remaining three collectives so that
S-007 cross-collective small-message gap analysis can be re-baselined on
the cached fastpath.

Design
------
The cache key is intentionally restrictive so that *any* shape change,
dtype change, group-membership change, or block-size change misses the
cache and falls back to the cold path (which is the existing
:func:`iris.ccl.<coll>.launch` path).  Cache entries are invalidated
implicitly by being keyed on every observable input — including
``ctx.get_rank()`` so that a Config object accidentally shared across
processes can never replay a stale closure built for a different rank.

The cache lives on the user's ``Config`` instance (lazy attribute
``_iris_launch_cache``) — never global — so:

* Multiple Configs do not share state, matching the K-820 contract.
* Garbage-collecting the Config releases the cache.
* No locking is needed: every iris.ccl call is single-threaded per rank.

Implementation note
-------------------
The fastpath stores a *closure* that re-invokes the Triton ``JITFunction``
launcher with already-frozen scalar / constexpr / kwarg arguments.  The
only per-call work on a hit is:

* re-reading the strides from the new (input, output) tensors
* re-fetching ``ctx.get_heap_bases()`` (constant after iris init,
  but cheap)
* invoking ``kernel_fn[grid](*args, **kwargs)`` — the in-process Triton
  cache ensures this resolves the same CompiledKernel without recompile

This avoids re-running ``extract_group_info`` and the Python-side
variant dispatch on every call, which is the bulk of the
R-PY-WRAPPER-FUSE bucket from K-786.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple


def _get_cache(config) -> Dict[Tuple[Any, ...], Callable]:
    """Return the per-Config launch cache, lazily creating it.

    The attribute name is intentionally underscore-prefixed and does
    not appear in :class:`Config`'s dataclass fields so that
    ``__post_init__`` validation, equality, and repr are unaffected.
    """
    cache = getattr(config, "_iris_launch_cache", None)
    if cache is None:
        cache = {}
        # Config is a regular (non-frozen) dataclass — direct setattr is fine.
        object.__setattr__(config, "_iris_launch_cache", cache)
    return cache


def _config_signature(config) -> Tuple[Any, ...]:
    """Subset of Config fields that uniquely select a CompiledKernel.

    Anything that flows into a ``tl.constexpr`` argument or that
    chooses a different kernel_fn must appear here.  Conservative:
    we include everything Config uses to build kernel kwargs.
    """
    return (
        config.block_size_m,
        config.block_size_n,
        config.swizzle_size,
        config.comm_sms,
        config.num_xcds,
        config.chunk_size,
        config.num_stages,
        config.num_warps,
        config.threads_per_warp,
        config.waves_per_eu,
        config.use_gluon,
        config.all_gather_variant,
        config.all_reduce_variant,
        config.all_reduce_distribution,
        config.all_reduce_num_rings,
        config.all_reduce_ring_slice_n,
        config.reduce_scatter_variant,
    )


def make_key(
    collective: str,
    output_tensor,
    input_tensor,
    ctx,
    world_size: int,
    rank_global: int,
    config,
    extra: Tuple[Any, ...] = (),
) -> Tuple[Any, ...]:
    """Build a cache key for a single iris.ccl call.

    Includes every observable that could change which CompiledKernel
    or which scalar argument the launcher needs:

    * collective name (so the cache is shared across collectives without
      collision)
    * input/output shapes and dtype (drive M/N/strides + Triton spec key)
    * world_size + rank_global (rank flows in as a constexpr; world_size
      changes the constexpr args)
    * full :func:`_config_signature` (every Config field that influences
      either kernel_fn selection or constexpr args)
    * caller-supplied ``extra`` tuple for collective-specific bits
      (e.g., the active variant string for all_reduce)
    """
    return (
        collective,
        tuple(input_tensor.shape),
        tuple(output_tensor.shape),
        input_tensor.dtype,
        output_tensor.dtype,
        world_size,
        rank_global,
        _config_signature(config),
        extra,
    )


def lookup(config, key):
    """Return the cached fast-launch closure for ``key``, or ``None``."""
    return _get_cache(config).get(key)


def store(config, key, fast_launch: Callable) -> None:
    """Insert a fast-launch closure for ``key``."""
    _get_cache(config)[key] = fast_launch


def clear(config) -> None:
    """Drop every cached entry on ``config``.

    Useful in tests / benchmarks that want to re-baseline the cold path.
    """
    cache = getattr(config, "_iris_launch_cache", None)
    if cache is not None:
        cache.clear()


# ---------------------------------------------------------------------------
# Hit/miss instrumentation (test-only)
#
# The collective wrappers call ``record_hit()`` when their probe finds a
# cached closure and ``record_miss()`` when it does not.  Production code
# never reads these counters; they exist so unit tests can prove that the
# fastpath actually fires (and not silently misses) on a repeat call with
# the same Config + shape.  The counters are deliberately module-level
# (not per-Config) because the only consumer is a single-process test;
# instrumenting per-Config would push a per-call attribute write onto the
# hot path with no production benefit.
# ---------------------------------------------------------------------------
_stats = {"hits": 0, "misses": 0}


def record_hit() -> None:
    _stats["hits"] += 1


def record_miss() -> None:
    _stats["misses"] += 1


def get_stats() -> Dict[str, int]:
    """Return a copy of the hit/miss counters."""
    return dict(_stats)


def reset_stats() -> None:
    _stats["hits"] = 0
    _stats["misses"] = 0
