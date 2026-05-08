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
Per-Config descriptor cache, keyed on a small ``(M, N, dtype)`` tuple
(world/group/block sizes are config-bound and constant per cache).
A single ``_LaunchDescriptor`` class wraps the resolved
``(kernel_fn, grid, args_after_io, kwargs)`` tuple. The warm path
becomes::

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
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Tuple


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


# Keep K-820's name as an alias for backwards compatibility with the
# fix/K-820-fused-launch-cache branch architecture; both classes share
# the same shape (kernel_fn / grid / args_after_io / kwargs).
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
