# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tile-level gather primitive for Iris.

This is a simpler, lower-level primitive than all_gather:
- gather: Load a tile from a SPECIFIC rank and return it (no store)
- all_gather: Load tiles from ALL ranks and store them to a buffer

Use `gather` when you want to consume the tile immediately without materializing it.
"""

import triton
import triton.language as tl
from iris.iris import DeviceContext
from .core import Tile, TensorView


@triton.jit()
def gather(
    tile: Tile,
    src_view: TensorView,
    source_rank: tl.constexpr,
    ctx: DeviceContext,
):
    """
    Tile-level gather from a specific rank.

    Loads a tile from source_rank's memory and returns it directly.
    Unlike all_gather, this does NOT store the tile - it's meant to be
    immediately consumed by the caller (e.g., in a GEMM dot product).

    Args:
        tile: Tile object with position and dimensions.
        src_view: TensorView for source tensor on source_rank.
        source_rank: Specific rank to load from (constexpr).
        ctx: DeviceContext with rank, world_size, and heap_bases.

    Returns:
        Loaded tile data as a tensor.

    Example usage in fused GEMM:
        for source_rank in range(world_size):
            a = gather(tile, src_view, source_rank, ctx)
            b = tl.load(...)
            acc += tl.dot(a, b)  # Consume immediately, no materialization
    """
    # Get tile pointer and mask
    src_tile_ptr, mask = src_view.tile_ptr(tile)

    if source_rank == ctx.rank:
        # Local load - can use vectorization hints since alignment is guaranteed
        local_ptr = tl.multiple_of(src_tile_ptr, (1, tile.block_n))
        local_ptr = tl.max_contiguous(local_ptr, (1, tile.block_n))
        tile_data = tl.load(local_ptr, mask=mask)
    else:
        # Remote load using RMA - inline translation and apply hints AFTER translation
        # Hints must be applied to the translated pointer because pointer arithmetic
        # (cast to uint64, subtract, add, cast back) destroys hint metadata.
        # Alignment IS preserved because symmetric heaps are all page-aligned.
        from_base = tl.load(ctx.heap_bases + ctx.rank)
        to_base = tl.load(ctx.heap_bases + source_rank)
        ptr_int = tl.cast(src_tile_ptr, tl.uint64)
        offset = ptr_int - from_base
        to_base_byte = tl.cast(to_base, tl.pointer_type(tl.int8))
        translated_ptr_byte = to_base_byte + offset
        translated_ptr = tl.cast(translated_ptr_byte, src_tile_ptr.dtype)
        # Apply vectorization hints AFTER translation
        translated_ptr = tl.multiple_of(translated_ptr, (1, tile.block_n))
        translated_ptr = tl.max_contiguous(translated_ptr, (1, tile.block_n))
        tile_data = tl.load(translated_ptr, mask=mask)

    return tile_data
