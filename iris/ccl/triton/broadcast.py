# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernels for broadcast collective communication.

Two variants:

- ``direct``  : the source rank pushes the entire tensor to every other rank.
                Best for small payloads (<1 MiB) where setup cost dominates.

- ``tree``    : staged scatter + all-gather (a.k.a. relay/Rabenseifner-style
                broadcast).  Phase 1 partitions the payload into ``world_size``
                row-wise shards; the source pushes shard ``i`` to rank ``i`` only.
                Phase 2 has every rank simultaneously push its shard to every
                other rank.  Phase 2 keeps all 8 GPU egress links saturated
                instead of just the source's single link, closing the kernel-time
                gap observed at >=1 MiB sizes (see K-156, K-357).
"""

import triton
import triton.language as tl

import iris
from iris.host.tracing.kernel_artifacts import iris_launch


# ---------------------------------------------------------------------------
# Direct broadcast (source pushes to all peers)
# ---------------------------------------------------------------------------


@triton.jit()
def persistent_broadcast_direct(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    src_rank_in_group: tl.constexpr,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,  # unused, kept for signature uniformity
    CHUNK_SIZE: tl.constexpr,  # unused, kept for signature uniformity
):
    """One-shot broadcast: only the src rank actively pushes.

    Non-source ranks do nothing — they wait for the trailing ``ctx.barrier()``
    and read the populated ``output_tensor`` that the source wrote into their
    symmetric heap via ``iris.store``.
    """
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    tl.assume(total_tiles > 0)

    # Only the source rank performs the push.
    if group_rank != src_rank_in_group:
        return

    for tile_id in range(pid, total_tiles, COMM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        gid = tile_id // num_pid_in_group
        first_pid_m = gid * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        mask = (rm[:, None] < M) & (rn[None, :] < N)

        in_off = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        out_off = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        in_p = input_ptr + in_off
        out_p = output_ptr + out_off
        in_p = tl.multiple_of(in_p, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        out_p = tl.multiple_of(out_p, (BLOCK_SIZE_M, BLOCK_SIZE_N))

        data = tl.load(in_p, mask=mask, other=0.0)

        # Local copy first.
        tl.store(out_p, data, mask=mask, cache_modifier=".wt")

        # Push to all remote peers.
        for i in tl.static_range(world_size):
            if i != src_rank_in_group:
                target_rank = rank_start + i * rank_stride
                iris.store(
                    out_p,
                    data,
                    iris_rank,
                    target_rank,
                    heap_bases,
                    mask=mask,
                    hint=(1, BLOCK_SIZE_N),
                )


# ---------------------------------------------------------------------------
# Tree / staged-relay broadcast — phase 1: scatter
# ---------------------------------------------------------------------------


@triton.jit()
def persistent_broadcast_tree_scatter(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    src_rank_in_group: tl.constexpr,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,  # unused
    CHUNK_SIZE: tl.constexpr,  # unused
):
    """Tree broadcast — phase 1.

    The source rank partitions the M dimension into ``world_size`` shards.
    Shard ``i`` is pushed to rank ``i`` only (including a local copy for the
    source's own shard).  After this kernel + a barrier, every rank holds its
    own shard in ``output_tensor`` and can participate in phase 2.
    """
    pid = tl.program_id(0)

    if group_rank != src_rank_in_group:
        return

    rows_per_shard = tl.cdiv(M, world_size)
    num_pid_m = tl.cdiv(rows_per_shard, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    tiles_per_shard = num_pid_m * num_pid_n
    total_tiles = tiles_per_shard * world_size
    tl.assume(total_tiles > 0)

    for tile_id in range(pid, total_tiles, COMM_SMS):
        shard_idx = tile_id // tiles_per_shard
        local_tile = tile_id % tiles_per_shard

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        gid = local_tile // num_pid_in_group
        first_pid_m = gid * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((local_tile % num_pid_in_group) % group_size_m)
        pid_n = (local_tile % num_pid_in_group) // group_size_m

        shard_row_start = shard_idx * rows_per_shard
        rm_local = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = shard_row_start + rm_local
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        shard_row_end = tl.minimum(shard_row_start + rows_per_shard, M)
        mask = (rm[:, None] < shard_row_end) & (rn[None, :] < N) & (rm[:, None] < M)

        in_off = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        out_off = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        in_p = input_ptr + in_off
        out_p = output_ptr + out_off

        data = tl.load(in_p, mask=mask, other=0.0)

        target_rank = rank_start + shard_idx * rank_stride
        if shard_idx == src_rank_in_group:
            # Local copy for the source's own shard.
            tl.store(out_p, data, mask=mask, cache_modifier=".wt")
        else:
            iris.store(
                out_p,
                data,
                iris_rank,
                target_rank,
                heap_bases,
                mask=mask,
                hint=(1, BLOCK_SIZE_N),
            )


# ---------------------------------------------------------------------------
# Tree / staged-relay broadcast — phase 2: all-gather
# ---------------------------------------------------------------------------


@triton.jit()
def persistent_broadcast_tree_gather(
    output_ptr,
    M,
    N,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    src_rank_in_group: tl.constexpr,  # unused, kept for signature uniformity
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,  # unused
    CHUNK_SIZE: tl.constexpr,  # unused
):
    """Tree broadcast — phase 2.

    Every rank already holds its own shard (rows ``group_rank * R .. (group_rank+1)*R``)
    in ``output_tensor`` from phase 1.  Each rank now reads its shard and
    pushes it to every other rank in parallel — saturating all
    ``world_size`` egress links simultaneously.
    """
    pid = tl.program_id(0)

    rows_per_shard = tl.cdiv(M, world_size)
    shard_row_start = group_rank * rows_per_shard
    shard_row_end = tl.minimum(shard_row_start + rows_per_shard, M)

    num_pid_m = tl.cdiv(rows_per_shard, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    tl.assume(total_tiles > 0)

    for tile_id in range(pid, total_tiles, COMM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        gid = tile_id // num_pid_in_group
        first_pid_m = gid * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm_local = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = shard_row_start + rm_local
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        mask = (rm[:, None] < shard_row_end) & (rn[None, :] < N) & (rm[:, None] < M)

        out_off = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        out_p = output_ptr + out_off

        data = tl.load(out_p, mask=mask, other=0.0)

        # Push our shard to every peer except ourselves; we already have it.
        # Stagger the start rank by pid to avoid all PIDs hitting the same
        # peer first — improves egress link utilization.
        start_rank_idx = pid % world_size
        for i in tl.static_range(world_size):
            target_idx = (start_rank_idx + i) % world_size
            if target_idx != group_rank:
                target_rank = rank_start + target_idx * rank_stride
                iris.store(
                    out_p,
                    data,
                    iris_rank,
                    target_rank,
                    heap_bases,
                    mask=mask,
                    hint=(1, BLOCK_SIZE_N),
                )


# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------


def launch(
    input_tensor,
    output_tensor,
    ctx,
    src_rank_in_group,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    config,
):
    """Dispatch to the chosen broadcast variant.

    For the ``tree`` variant the launch is two kernels separated by a
    ``ctx.barrier()`` — phase 1 (scatter) populates each rank's shard, and
    phase 2 (all-gather) replicates shards to every peer.
    """
    M, N = output_tensor.shape[:2]
    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    heap_bases = ctx.get_heap_bases()
    variant = config.broadcast_variant

    common_kwargs = dict(
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm="broadcast",
        rank=rank_global,
        dtype=output_tensor.dtype,
    )

    if variant == "direct":
        iris_launch(
            persistent_broadcast_direct,
            (config.comm_sms,),
            input_tensor,
            output_tensor,
            M,
            N,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
            heap_bases,
            src_rank_in_group,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            **common_kwargs,
        )
        return

    if variant == "tree":
        # Phase 1: source scatters shards to every rank.
        iris_launch(
            persistent_broadcast_tree_scatter,
            (config.comm_sms,),
            input_tensor,
            output_tensor,
            M,
            N,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
            heap_bases,
            src_rank_in_group,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            **common_kwargs,
        )
        # Synchronize before phase 2 — every rank must own its shard before
        # peers can read+push it.
        ctx.barrier()
        # Phase 2: every rank pushes its shard to every other rank.
        iris_launch(
            persistent_broadcast_tree_gather,
            (config.comm_sms,),
            output_tensor,
            M,
            N,
            stride_out_m,
            stride_out_n,
            heap_bases,
            src_rank_in_group,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            **common_kwargs,
        )
        return

    raise ValueError(f"Unknown broadcast_variant: {variant!r}")
