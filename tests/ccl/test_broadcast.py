# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for the GPU/RMA broadcast collective operation.

Exercises both the ``direct`` and ``tree`` variants of
``ctx.ccl.broadcast_tensor`` and verifies the ``auto`` policy switches
to the ``tree`` variant once the payload crosses
``BROADCAST_TREE_THRESHOLD_BYTES`` (1 MiB).

The reference is ``torch.distributed.broadcast`` over NCCL.
"""

import gc

import pytest
import torch
import torch.distributed as dist

import iris
from iris.ccl import Config
from iris.ccl.broadcast import BROADCAST_TREE_THRESHOLD_BYTES, _resolve_variant


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
        # Reference: PyTorch / NCCL broadcast.
        if rank == src:
            pytorch_input = torch.arange(M * N, dtype=dtype, device=f"cuda:{rank}").reshape(M, N)
        else:
            pytorch_input = torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}")
        shmem.barrier()
        dist.broadcast(pytorch_input, src=src)
        torch.cuda.synchronize()

        # Iris path. ``shmem.zeros`` is a collective — every rank must allocate
        # both buffers in lock-step. Only the source rank populates ``iris_input``;
        # other ranks pass the (zero-filled) buffer purely as a same-shape stub.
        iris_output = shmem.zeros((M, N), dtype=dtype)
        iris_input = shmem.zeros((M, N), dtype=dtype)
        if rank == src:
            iris_input.copy_(pytorch_input)

        shmem.barrier()
        config = Config(block_size_m=32, block_size_n=64, broadcast_variant=variant)
        shmem.ccl.broadcast_tensor(iris_output, iris_input, src=src, config=config)
        torch.cuda.synchronize()

        atol = 1e-3 if dtype == torch.float16 else 1e-5
        max_diff = torch.abs(iris_output - pytorch_input).max().item()
        assert torch.allclose(iris_output, pytorch_input, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris broadcast (variant={variant!r}, src={src}) "
            "doesn't match torch.distributed.broadcast"
        )
    finally:
        _cleanup(shmem)


# ---------------------------------------------------------------------------
# Variant correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["direct", "tree", "auto"])
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32, torch.bfloat16],
)
@pytest.mark.parametrize(
    "M, N",
    [
        (128, 64),  # ~16 KiB (fp16) — sub-threshold; auto picks direct.
        (1024, 256),  # ~512 KiB (fp16) — sub-threshold.
        (1024, 1024),  # 2 MiB (fp16) / 4 MiB (fp32) — over-threshold; auto picks tree.
        (4096, 4096),  # 32 MiB (fp16) / 64 MiB (fp32) — large, tree is the win.
    ],
)
def test_broadcast_tensor(variant, dtype, M, N):
    """Both variants (and auto) must produce bit-equivalent output to NCCL broadcast."""
    _run_broadcast(M=M, N=N, dtype=dtype, variant=variant, src=0)


@pytest.mark.parametrize("variant", ["direct", "tree"])
def test_broadcast_tensor_nonzero_src(variant):
    """Source rank != 0 must work for both variants."""
    _run_broadcast(M=1024, N=1024, dtype=torch.float32, variant=variant, src=1)


# ---------------------------------------------------------------------------
# Auto policy
# ---------------------------------------------------------------------------


def test_auto_threshold_is_one_mib():
    """The auto-policy threshold is the documented 1 MiB."""
    assert BROADCAST_TREE_THRESHOLD_BYTES == 1 << 20


@pytest.mark.parametrize(
    "M, N, dtype, expected",
    [
        (128, 64, torch.float16, "direct"),  # 16 KiB
        (256, 256, torch.float16, "direct"),  # 128 KiB
        (1024, 512, torch.float16, "tree"),  # 1 MiB exactly — tree.
        (1024, 1024, torch.float16, "tree"),  # 2 MiB — tree.
        (1024, 1024, torch.float32, "tree"),  # 4 MiB — tree.
    ],
)
def test_auto_resolves_to_expected_variant(M, N, dtype, expected):
    """The auto policy must pick tree at exactly the 1 MiB threshold."""
    out = torch.empty((M, N), dtype=dtype)
    assert _resolve_variant("auto", out) == expected


def test_explicit_variant_is_respected():
    """An explicit variant string must pass through ``_resolve_variant`` unchanged."""
    out = torch.empty((4096, 4096), dtype=torch.float32)
    assert _resolve_variant("direct", out) == "direct"
    assert _resolve_variant("tree", out) == "tree"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_broadcast_variant_raises():
    """The Config must reject unknown variants."""
    with pytest.raises(ValueError, match="broadcast_variant"):
        Config(broadcast_variant="bogus")
