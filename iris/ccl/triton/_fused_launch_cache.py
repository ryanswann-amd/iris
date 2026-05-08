# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused launch descriptor cache for iris.ccl all_reduce two_shot.

Background (K-786 v2, K-796 sub-phase decomposition on c42 8x MI300X):
    iris ``all_reduce`` two_shot launch_us decomposes into:

        py_wrapper       19.7 us   <-- iris Python dispatch (group resolve,
                                       variant if/elif, heap_bases lookup,
                                       4-deep nested function calls)
        cache_lookup     14.6 us   <-- triton.JITFunction.run binder/key/dict.get
        hip_enqueue      10.3 us   <-- HIP kernel enqueue (irreducible)
        wrapper_exit      2.0 us   <-- iris post-call bookkeeping
        stream_record     7.9 us   <-- CUDA event record (irreducible)
        ----------------------------
        total            54.6 us

    The top-2 sub-phases (py_wrapper + cache_lookup = 34.3 us, 63 % of launch)
    are independent of message size and fully resolvable once
    (variant, M, N, dtype, world_size, group, ...) is known.

Design
------
Instead of caching every (shape, config, ...) tuple in a module-global dict
(which costs ~3 us per call to compute the key + lookup) we attach the
descriptor cache directly to the user's ``Config`` object. The first call
with ``config.fused_launch=True`` populates ``config._fused_cache`` keyed
on a tiny tuple ``(M, N, dtype)``; subsequent calls bypass:

    iris.ccl.all_reduce wrapper            (~3 us)
    iris.ccl.triton.all_reduce.launch     (~3 us)
    iris.host.tracing.kernel_artifacts.iris_launch  (~2 us)
    variant if/elif dispatch + heap_bases  (~2 us)
    extract_group_info + validation        (~2 us)
    ----------
    total elided wrapper                  ~12 us

and go straight to ``triton_kernel[grid](...)`` with pre-bound positional
args. Triton's own ``cache_lookup`` (~14.6 us) remains because we still go
through the public ``kernel_fn[grid]`` entry point -- the fastpath is safe
across Triton versions for that reason.

Combined elided overhead: ~12 us out of ~52 us = ~23 % launch_us reduction
target; expected gain on c42 MI300X bf16 8x at <=16KB.

Activation: ``Config(fused_launch=True)`` or env var ``IRIS_CCL_FUSED_LAUNCH=1``.
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
class TwoShotDescriptor:
    """Hot-path launch state for ``persistent_all_reduce_two_shot``.

    Captured once per (M, N, dtype) cell. ``invoke`` is the warm-path entry
    and is intentionally minimal: 4 attribute loads + 1 Triton call.

    Tensor pointers (input/output) are still rebound per call (Triton
    re-derives them from the tensor objects). Everything else --
    grid, M, N, strides, ranks, block sizes, swizzle/SMs/xcds/chunk,
    distribution, heap_bases tensor reference, kernel kwargs -- is
    pre-resolved.
    """

    kernel_fn: Any
    grid: Tuple[int, ...]
    args_after_io: Tuple[Any, ...]  # everything after (input, output): M, N, strides, heap_bases, ranks, constexprs
    kwargs: dict
    # For the trailing barrier in the public wrapper:
    ctx_barrier: Any  # callable: ctx.barrier
    # async/sync flag for the trailing barrier is decided per-call.

    def invoke(self, input_tensor, output_tensor) -> None:
        """Warm-path replay. Equivalent to the full iris dispatch chain
        plus the trailing iris_launch call."""
        self.kernel_fn[self.grid](input_tensor, output_tensor, *self.args_after_io, **self.kwargs)


def get_or_build_cache(config) -> dict:
    """Return the per-Config descriptor cache, lazily constructing it.

    Stored as a private attribute on the Config dataclass so that:
      - lookup is one ``getattr`` (no module-global dict),
      - cache is naturally scoped to the user's Config lifetime,
      - multiple Configs (e.g. one per workload) don't share cache pollution.
    """
    cache = getattr(config, "_fused_cache", None)
    if cache is None:
        cache = {}
        # NB: dataclasses with frozen=False (Config is not frozen) accept
        # extra attributes via setattr.
        object.__setattr__(config, "_fused_cache", cache)
    return cache
