# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused launch descriptor cache for iris.ccl collectives.

K-871: Port of the K-820 fused-launch fastpath pattern from
``all_reduce two_shot`` to the remaining collectives:
``all_gather``, ``reduce_scatter`` (two_shot), and ``all_to_all``.

Background (K-786 v2, K-796 sub-phase decomposition on c42 8x MI300X):
    All five iris collectives sit on the same ~50us host-side launch
    envelope. K-786 v2 decomposed it as::

        py_wrapper       19.7 us   <-- iris Python dispatch (group resolve,
                                       variant if/elif, heap_bases lookup,
                                       4-deep nested function calls)
        cache_lookup     14.6 us   <-- triton.JITFunction.run binder/key/dict.get
        hip_enqueue      10.3 us   <-- HIP kernel enqueue (irreducible)
        wrapper_exit      2.0 us   <-- iris post-call bookkeeping
        stream_record     7.9 us   <-- CUDA event record (irreducible)

    The top-2 sub-phases (py_wrapper + cache_lookup ~ 34.3 us, ~63 % of
    launch) are independent of message size and resolvable once
    ``(variant, M, N, dtype, world_size, group, ...)`` is known.

Design
------
Per-Config descriptor cache keyed on ``(collective_name, M, N, dtype)``.
A single ``_LaunchDescriptor`` wraps the resolved
``(kernel_fn, grid, args_after_io, kwargs)`` tuple. Warm path becomes::

    descriptor.invoke(input_tensor, output_tensor)

i.e. one Python dict lookup + one Triton ``kernel[grid](...)`` call,
avoiding all iris-side wrapping (extract_group_info, variant if/elif,
heap_bases lookup, kwargs construction, the iris_launch tracing wrapper).

The bound launcher allocation insight from K-820 still applies: caching
the entire ``kernel_fn[grid](...)`` invocation pattern (rather than just
the kernel reference) recovers the ~14.6 us ``cache_lookup`` contribution
that comes from Triton's ``KernelInterface.__getitem__`` allocating a
fresh bound launcher object per call.

Activation: ``Config(fused_launch=True)`` or env ``IRIS_CCL_FUSED_LAUNCH=1``.

Cold-path safety: any first call falls through to the unchanged slow
path, which then records a descriptor for the warm path. The fastpath
is byte-for-byte default-off compatible.

Refactor notes (K-871 v2): a single :func:`try_fused_fastpath` driver
collapses what used to be three near-identical fastpath stanzas in the
public collective wrappers, and :func:`make_descriptor` collapses the
three near-identical ``capture_*_descriptor`` helpers in the per-collective
``triton/*.py`` files. Each public wrapper / capture site is now <15 lines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Tuple


def fused_launch_enabled() -> bool:
    """True iff the env-var fastpath toggle is set."""
    val = os.environ.get("IRIS_CCL_FUSED_LAUNCH", "").strip().lower()
    return val in ("1", "true", "yes", "on")


@dataclass
class _LaunchDescriptor:
    """Captured warm-path launch state for a single (collective, M, N, dtype) cell.

    Attributes
    ----------
    kernel_fn : triton.JITFunction
        The Triton kernel to invoke.
    grid : tuple
        Launch grid (typically ``(config.comm_sms,)``).
    args_after_io : tuple
        All positional kernel args after ``(input_ptr, output_ptr)``.
        Captured at cold-call time -- M, N, strides, heap_bases, ranks,
        constexpr block sizes, etc. -- so the warm path only rebinds
        the input/output tensors per call.
    kwargs : dict
        Keyword args passed through to the kernel invocation
        (``num_warps``, ``num_stages``, ``waves_per_eu``).

    Notes
    -----
    All iris collectives use the ``(input_ptr, output_ptr, ...)``
    positional ordering, so a single descriptor shape works for every
    collective targeted by K-871.
    """

    kernel_fn: Any
    grid: Tuple[int, ...]
    args_after_io: Tuple[Any, ...]
    kwargs: dict

    def invoke(self, input_tensor, output_tensor) -> None:
        """Warm-path replay. Equivalent to the full iris dispatch chain
        plus the trailing ``iris_launch`` call, but with all per-call
        Python overhead pre-resolved."""
        self.kernel_fn[self.grid](input_tensor, output_tensor, *self.args_after_io, **self.kwargs)


# K-820 alias kept for backwards compatibility with the K-820 branch.
TwoShotDescriptor = _LaunchDescriptor


def get_or_build_cache(config) -> dict:
    """Return the per-Config descriptor cache, lazily constructing it.

    Stored as a private attribute on the Config dataclass so that:

    - lookup is one ``getattr`` (no module-global dict),
    - cache lifetime tracks the user's Config object,
    - multiple Configs (e.g. one per workload) don't share cache pollution.

    The cache is a flat dict keyed on ``(collective_name, M, N, dtype)``
    so a single Config can host descriptors for all four collectives.
    """
    cache = getattr(config, "_fused_cache", None)
    if cache is None:
        cache = {}
        # Config is a non-frozen dataclass; setattr is fine.
        object.__setattr__(config, "_fused_cache", cache)
    return cache


def make_descriptor(
    *,
    kernel_fn: Any,
    config,
    args_after_io: Tuple[Any, ...],
) -> _LaunchDescriptor:
    """Build a standard-shape :class:`_LaunchDescriptor` for an iris collective.

    All four iris-ccl collectives launch a single Triton kernel with grid
    ``(config.comm_sms,)`` and the same three ``num_stages / num_warps /
    waves_per_eu`` kwargs. This helper centralises that boilerplate so the
    per-collective ``capture_*_descriptor`` functions only need to choose
    ``kernel_fn`` and assemble ``args_after_io``.
    """
    return _LaunchDescriptor(
        kernel_fn=kernel_fn,
        grid=(config.comm_sms,),
        args_after_io=args_after_io,
        kwargs={
            "num_stages": config.num_stages,
            "num_warps": config.num_warps,
            "waves_per_eu": config.waves_per_eu,
        },
    )


def _fastpath_active(config, *, extra_guard: bool, group) -> bool:
    """Predicate guarding entry into the fused-launch fastpath."""
    return (
        config is not None
        and (getattr(config, "fused_launch", False) or fused_launch_enabled())
        and group is None  # group != None case rarely benchmarked; falls back
        and extra_guard
    )


def try_fused_fastpath(
    *,
    collective_name: str,
    config,
    input_tensor,
    output_tensor,
    ctx,
    group,
    async_op: bool,
    slow_path: Callable[[], None],
    capture: Callable[[], _LaunchDescriptor],
    extra_guard: bool = True,
) -> bool:
    """Drive the K-871 fastpath for one collective call.

    Parameters
    ----------
    collective_name : str
        Used as the first element of the cache key.
    config : iris.ccl.Config or None
        Caller's config; gates the fastpath via ``fused_launch``.
    input_tensor, output_tensor :
        The (input, output) tensors. ``shape[0], shape[1], dtype`` form
        the rest of the cache key.
    ctx :
        Iris context (used for the trailing ``barrier()`` on the warm path).
    group, async_op :
        Pass-through; ``group`` must be ``None`` for the fastpath to engage,
        ``async_op=True`` skips the trailing barrier.
    slow_path : callable
        Zero-arg closure executing the original (slow) path. Called on the
        cold path before descriptor capture.
    capture : callable
        Zero-arg closure returning a fresh ``_LaunchDescriptor``. Called
        only once per ``(collective_name, M, N, dtype)`` cell.
    extra_guard : bool
        Per-collective extra precondition (e.g. ``not config.use_gluon``,
        ``variant == 'two_shot'``). When ``False`` the fastpath is skipped
        entirely and the caller is expected to invoke ``slow_path`` itself.

    Returns
    -------
    bool
        ``True`` iff the call was handled (either by warm-path invoke
        or cold-path slow_path + capture). The caller short-circuits.
        ``False`` means the fastpath was inapplicable; caller falls through.
    """
    if not _fastpath_active(config, extra_guard=extra_guard, group=group):
        return False

    cache = get_or_build_cache(config)
    shape = input_tensor.shape
    key = (collective_name, shape[0], shape[1], input_tensor.dtype)
    desc = cache.get(key)
    if desc is not None:
        # Warm path: one dict-get + one Triton kernel call. Skip iris dispatch.
        desc.invoke(input_tensor, output_tensor)
        if not async_op:
            ctx.barrier()
        return True

    # Cold path: full slow path AND descriptor capture for the next call.
    slow_path()
    cache[key] = capture()
    return True
