# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for the GPU/RMA broadcast collective operation.

Exercises both the ``direct`` and ``scatter_allgather`` variants of
``ctx.ccl.broadcast_tensor`` and verifies the ``auto`` policy switches
to the ``scatter_allgather`` variant once the payload crosses
``BROADCAST_SCATTER_ALLGATHER_THRESHOLD_BYTES`` (1 MiB).

Includes a non-aligned shape (``M`` not divisible by ``world_size``) to
prove the per-tile mask handles the trailing short shard correctly.

The reference is ``torch.distributed.broadcast`` over NCCL.
"""

import gc

import pytest
import torch
import torch.distributed as dist

import iris
from iris.ccl import Config
from iris.ccl.broadcast import (
    BROADCAST_SCATTER_ALLGATHER_THRESHOLD_BYTES,
    BROADCAST_TREE_THRESHOLD_BYTES,
    _resolve_variant,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cleanup(shmem):
    """Barrier + drop the iris context — match the pattern in the other CCL tests."""
    shmem.barrier()
    del shmem
    gc.collect()


def _run_broadcast(M, N, dtype, variant, src):
    """Shared body: compare ``ctx.ccl.broadcast_tensor`` against NCCL broadcast."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8 GiB — large enough for the >=1 MiB tree-variant cases.
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    if not (0 <= src < world_size):
        pytest.skip(f"src={src} out of range for world_size={world_size}")

    try:
        # Mirror the bring-up sequence used by tests/ccl/test_all_gather.py
        # exactly: regular CUDA tensors first, ``shmem.barrier()``, NCCL
        # collective for the reference, ``torch.cuda.synchronize``, *then* the
        # symmetric-heap allocations.  Reversing this order has been observed
        # to deadlock the symmetric-heap peer-access fd-exchange under
        # pytest on torch+ROCm 7.2.
        if rank == src:
            pytorch_input = torch.arange(M * N, dtype=dtype, device=f"cuda:{rank}").reshape(M, N)
        else:
            pytorch_input = torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}")
        pytorch_output = torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}")

        shmem.barrier()
        # NCCL all_gather computes the broadcast reference: every rank sends
        # its current ``pytorch_input`` and receives a stacked tensor, then
        # we extract the source-rank slice.  We use ``all_gather`` instead of
        # ``dist.broadcast`` because the latter has been observed to interact
        # poorly with the iris peer-access bring-up on this stack (and
        # all-gather is the same NCCL operation pattern that
        # ``test_all_gather.py`` uses successfully).
        gathered = torch.zeros(world_size * M, N, dtype=dtype, device=f"cuda:{rank}")
        dist.all_gather_into_tensor(gathered, pytorch_input)
        torch.cuda.synchronize()
        # Reference == the source rank's slice of the all_gather output.
        pytorch_output.copy_(gathered[src * M : (src + 1) * M, :])
        del gathered

        iris_input = shmem.zeros((M, N), dtype=dtype)
        if rank == src:
            iris_input.copy_(pytorch_input)
        iris_output = shmem.zeros((M, N), dtype=dtype)

        shmem.barrier()
        config = Config(block_size_m=32, block_size_n=64, broadcast_variant=variant)
        shmem.ccl.broadcast_tensor(iris_output, iris_input, src=src, config=config)
        torch.cuda.synchronize()

        atol = 1e-3 if dtype == torch.float16 else 1e-5
        max_diff = torch.abs(iris_output - pytorch_output).max().item()
        assert torch.allclose(iris_output, pytorch_output, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris broadcast (variant={variant!r}, src={src}) "
            "doesn't match the NCCL reference"
        )
    finally:
        _cleanup(shmem)


# ---------------------------------------------------------------------------
# Variant correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["direct", "scatter_allgather", "auto"])
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32, torch.bfloat16],
)
@pytest.mark.parametrize(
    "M, N",
    [
        (128, 64),     # ~16 KiB (fp16) — sub-threshold; auto picks direct.
        (1024, 256),   # ~512 KiB (fp16) — sub-threshold.
        (1024, 1024),  # 2 MiB (fp16) / 4 MiB (fp32) — over-threshold; auto picks scatter_allgather.
        (4096, 4096),  # 32 MiB (fp16) / 64 MiB (fp32) — large, scatter_allgather wins.
    ],
)
def test_broadcast_tensor(variant, dtype, M, N):
    """Both variants (and auto) must produce bit-equivalent output to NCCL broadcast."""
    _run_broadcast(M=M, N=N, dtype=dtype, variant=variant, src=0)


@pytest.mark.parametrize("variant", ["direct", "scatter_allgather"])
def test_broadcast_tensor_nonzero_src(variant):
    """Source rank != 0 must work for both variants."""
    _run_broadcast(M=1024, N=1024, dtype=torch.float32, variant=variant, src=1)


@pytest.mark.parametrize("variant", ["direct", "scatter_allgather", "auto"])
@pytest.mark.parametrize(
    "M, N, dtype",
    [
        # M not divisible by world_size (8). 1025*128*fp16 = 256.25 KiB — sub-threshold,
        # exercises the masked-tail path of the scatter+allgather kernels even when
        # auto would otherwise pick `direct`.
        (1025, 128, torch.float16),
        # 1 MiB + 1 element: ((1<<19)+1)*fp16 = 1 MiB + 2 B.  At 8 ranks,
        # rows_per_shard=cdiv(M, 8) leaves the last shard with a partial row.
        # This is the precise case the Skeptic flagged.
        ((1 << 19) + 1, 1, torch.float16),
        # 1 MiB-ish (fp32) at a prime number of rows so EVERY shard except
        # the last is full and the last shard is short by a non-trivial amount.
        (257, 1024, torch.float32),
    ],
)
def test_broadcast_tensor_non_aligned_shape(variant, M, N, dtype):
    """``M`` not divisible by ``world_size`` must not corrupt the output.

    The scatter and all-gather kernels both clamp the per-tile mask to
    ``shard_row_end = min(start + rows_per_shard, M)`` so the trailing
    short or empty shard never writes past ``M``.  This test explicitly
    exercises that path.
    """
    _run_broadcast(M=M, N=N, dtype=dtype, variant=variant, src=0)


# ---------------------------------------------------------------------------
# Auto policy
# ---------------------------------------------------------------------------


def test_auto_threshold_is_one_mib():
    """The auto-policy threshold is the documented 1 MiB."""
    assert BROADCAST_SCATTER_ALLGATHER_THRESHOLD_BYTES == 1 << 20
    # Backwards-compat alias still resolves to the same bytes value.
    assert BROADCAST_TREE_THRESHOLD_BYTES == BROADCAST_SCATTER_ALLGATHER_THRESHOLD_BYTES


@pytest.mark.parametrize(
    "M, N, dtype, expected",
    [
        (128, 64, torch.float16, "direct"),                # 16 KiB
        (256, 256, torch.float16, "direct"),               # 128 KiB
        (1024, 512, torch.float16, "scatter_allgather"),   # 1 MiB exactly.
        (1024, 1024, torch.float16, "scatter_allgather"),  # 2 MiB.
        (1024, 1024, torch.float32, "scatter_allgather"),  # 4 MiB.
    ],
)
def test_auto_resolves_to_expected_variant(M, N, dtype, expected):
    """The auto policy must pick scatter_allgather at exactly the 1 MiB threshold."""
    out = torch.empty((M, N), dtype=dtype)
    assert _resolve_variant("auto", out) == expected


def test_explicit_variant_is_respected():
    """An explicit variant string must pass through ``_resolve_variant`` unchanged."""
    out = torch.empty((4096, 4096), dtype=torch.float32)
    assert _resolve_variant("direct", out) == "direct"
    assert _resolve_variant("scatter_allgather", out) == "scatter_allgather"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_broadcast_variant_raises():
    """The Config must reject unknown variants."""
    with pytest.raises(ValueError, match="broadcast_variant"):
        Config(broadcast_variant="bogus")


def test_legacy_tree_variant_string_raises():
    """The old ``"tree"`` name was renamed to ``"scatter_allgather"`` because the
    phase-2 step is structurally an all-gather (O(N) sends/rank), not a
    log-N tree.  Make sure we surface that to callers explicitly rather
    than silently accepting the misleading old name.
    """
    with pytest.raises(ValueError, match="broadcast_variant"):
        Config(broadcast_variant="tree")
