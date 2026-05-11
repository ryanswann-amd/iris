# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Broadcast collective operation — public API.

A single-source broadcast where rank ``src`` distributes ``input_tensor`` to
``output_tensor`` on every rank in the group.

Two variants are available (selected via ``Config.broadcast_variant``):

- ``direct`` — the source rank pushes the entire tensor to every peer over
  its own egress link.  Best for small payloads where setup cost dominates.

- ``tree`` — staged scatter + all-gather.  Phase 1 has the source push a
  ``1/world_size`` shard to every peer; phase 2 has every peer push its
  shard to every other peer.  Phase 2 saturates all egress links in
  parallel, closing the kernel-time gap observed at >=1 MiB sizes
  (see K-156 / K-357).  Auto-selected for payloads >=
  ``BROADCAST_TREE_THRESHOLD_BYTES`` (default 1 MiB) when
  ``broadcast_variant="auto"``.
"""

from iris.ccl.utils import extract_group_info


# Bytes threshold above which the auto policy chooses the ``tree`` variant.
# Sourced from K-357 RC-3 / K-156 §key finding 4.
BROADCAST_TREE_THRESHOLD_BYTES = 1 << 20  # 1 MiB


def _tensor_nbytes(tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _resolve_variant(variant: str, output_tensor) -> str:
    """Resolve ``"auto"`` to either ``"direct"`` or ``"tree"`` based on size."""
    if variant != "auto":
        return variant
    return "tree" if _tensor_nbytes(output_tensor) >= BROADCAST_TREE_THRESHOLD_BYTES else "direct"


def broadcast(
    output_tensor,
    input_tensor,
    ctx,
    src=0,
    group=None,
    async_op=False,
    config=None,
):
    """
    Broadcast: rank ``src`` distributes ``input_tensor`` to ``output_tensor``
    on every rank.

    Args:
        output_tensor: Shape (M, N) — receive buffer on every rank.
        input_tensor:  Shape (M, N) — only the contents on rank ``src`` are read.
                       Non-source ranks may pass any tensor with the same shape
                       (typically the same buffer as ``output_tensor``).
        ctx: Iris instance.
        src: Source rank (within the group). Defaults to 0.
        group: ProcessGroup or None. If None, uses all ranks in ``ctx``.
        async_op: If True, skip the trailing ``ctx.barrier()``.
        config: Config with kernel parameters.

    Notes:
        - The variant is selected by ``config.broadcast_variant``
          (``"direct"``, ``"tree"``, or ``"auto"``).
        - ``"auto"`` chooses ``"tree"`` for payloads >= 1 MiB and
          ``"direct"`` otherwise.
    """
    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=64)

    if config.use_gluon:
        # Gluon backend not yet implemented; fall back to triton kernels.
        # No silent semantic change — just route to the triton path.
        pass

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    if not (0 <= src < world_size):
        raise ValueError(f"src={src} out of range for world_size={world_size}")

    M, N = output_tensor.shape[:2]
    if input_tensor.shape[:2] != (M, N):
        raise ValueError(
            f"Input tensor shape {input_tensor.shape[:2]} does not match output shape {(M, N)}. "
            "Broadcast requires identically shaped input and output tensors."
        )
    if input_tensor.dtype != output_tensor.dtype:
        raise ValueError(
            f"Input dtype {input_tensor.dtype} does not match output dtype {output_tensor.dtype}."
        )

    # Resolve auto variant before dispatch so the selected variant is recorded
    # by tracing under a stable name.
    resolved_variant = _resolve_variant(config.broadcast_variant, output_tensor)
    if resolved_variant != config.broadcast_variant:
        # Mutate a shallow copy so the caller's config is untouched.
        from dataclasses import replace

        config = replace(config, broadcast_variant=resolved_variant)

    from iris.ccl.triton.broadcast import launch

    launch(
        input_tensor,
        output_tensor,
        ctx,
        src,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        config,
    )

    if not async_op:
        ctx.barrier()
