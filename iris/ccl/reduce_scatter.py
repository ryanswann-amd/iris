# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce-scatter collective communication primitive for Iris.
Uses the two-shot approach: reduce assigned tiles and store only to own rank.
"""

import triton
import triton.language as tl
import iris
from .config import Config
from .utils import chiplet_transform_chunked, ReduceOp, extract_group_info
from .all_reduce import persistent_all_reduce_ts_push_scatter


@triton.jit()
def persistent_reduce_scatter_v2_reduce(
    scratch_ptr,
    output_ptr,
    input_ptr,
    M,
    N,
    stride_out_m,
    stride_out_n,
    stride_in_m,
    stride_in_n,
    group_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    COMM_SMS: tl.constexpr,
):
    """Write-based reduce-scatter — local reduce of owned row-block.

    Owner ``group_rank`` owns global rows [group_rank*Mb:(group_rank+1)*Mb]. Its
    scratch holds the world_size-1 REMOTE contributions (slot s = rows
    [s*Mb:(s+1)*Mb]); the own contribution is read straight from ``input``. Sum
    them into output's owned rows. Local only (no xGMI).
    """
    pid = tl.program_id(0)
    Mb = M // world_size
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pids_per_owner = Mb // BLOCK_SIZE_M
    owned_tiles = pids_per_owner * num_pid_n
    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    for tile in range(pid, owned_tiles, COMM_SMS):
        local_pid_m = tile // num_pid_n
        pid_n = tile % num_pid_n
        global_pid_m = group_rank * pids_per_owner + local_pid_m

        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        local_rows = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        gm = global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        gm = tl.max_contiguous(tl.multiple_of(gm, BLOCK_SIZE_M), BLOCK_SIZE_M)

        acc = tl.load(input_ptr + gm[:, None] * stride_in_m + rn[None, :] * stride_in_n).to(acc_dtype)
        for s in tl.static_range(world_size):
            if s != group_rank:
                sm = (s * Mb) + local_rows
                sm = tl.max_contiguous(tl.multiple_of(sm, BLOCK_SIZE_M), BLOCK_SIZE_M)
                acc += tl.load(scratch_ptr + sm[:, None] * N + rn[None, :]).to(acc_dtype)

        out_off = gm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        tl.store(output_ptr + out_off, acc.to(output_ptr.type.element_ty))


@triton.jit()
def persistent_reduce_scatter_two_shot(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
):
    """
    Reduce-scatter using two-shot approach.

    Each rank reduces its assigned tiles from all ranks and stores the result
    only to its own output (no broadcast to other ranks).
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    tiles_per_rank = tl.cdiv(total_tiles, world_size)
    if DISTRIBUTION == 0:
        start_tile = group_rank
        stride = world_size
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.cdiv(remaining, stride)
    else:
        start_tile = group_rank * tiles_per_rank
        stride = 1
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.minimum(tiles_per_rank, remaining)

    for tile_offset in range(pid, max_tile_offset, COMM_SMS):
        tile_id = start_tile + tile_offset * stride

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N

        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

        # Build indices (used by both paths)
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)

        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        base_ptr = input_ptr + input_offset
        out_ptr = output_ptr + output_offset

        # Fast path: NO MASKS (full tiles)
        # The masking is problem size dependent, and the compiler does not recognize it can have two paths
        # (one with masks and one without). Separate unmasked paths allow the compiler to generate
        # more efficient vectorized instructions.
        if is_full:
            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(base_ptr, iris_rank, start_rank_global, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)

            reduced = acc.to(output_ptr.type.element_ty)

            # Store only to own rank (no broadcast)
            tl.store(out_ptr, reduced, cache_modifier=".wt")

        # Slow path: MASKED (only boundary tiles land here)
        # This path handles tiles at tensor boundaries where not all elements are valid.
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(base_ptr, iris_rank, start_rank_global, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(
                acc_dtype
            )
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(
                    acc_dtype
                )

            reduced = acc.to(output_ptr.type.element_ty)

            # Store only to own rank (no broadcast)
            tl.store(out_ptr, reduced, mask=mask, cache_modifier=".wt")


@triton.jit()
def persistent_reduce_scatter_push_scatter(
    input_ptr,
    scratch_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
):
    """
    Push (PUT) reduce-scatter — scatter phase.

    Direction-flipped twin of the two_shot (pull) kernel. Each rank WRITES every
    one of its input tiles to the *owner* rank's scratch (a remote WRITE), into
    per-source slot ``group_rank``. The owner later sums the world_size slots for
    the tiles it owns. Owner of a tile is computed identically to the two_shot
    pull kernel's tile assignment so the comparison is apples-to-apples.

    scratch is a contiguous (world_size*M, N) buffer; slot s lives at rows
    [s*M:(s+1)*M]. Only the owner-owned tile positions of each slot are written,
    so per-rank remote egress is (world_size-1)/world_size * M*N — matching the
    reduce-scatter bus volume.
    """
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    tiles_per_rank = tl.cdiv(total_tiles, world_size)

    for tile_id in range(pid, total_tiles, COMM_SMS):
        # Owner of this tile (mirrors two_shot assignment).
        if DISTRIBUTION == 0:
            owner = tile_id % world_size
        else:
            owner = tile_id // tiles_per_rank
            owner = min(owner, world_size - 1)
        dest_rank = rank_start + owner * rank_stride

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        data = tl.load(input_ptr + input_offset, mask=mask, other=0.0)

        # Write into the owner's scratch, slot = this rank (group_rank).
        rm_scratch = rm + group_rank * M
        scratch_offset = rm_scratch[:, None] * N + rn[None, :]
        scratch_local = scratch_ptr + scratch_offset

        if owner == group_rank:
            tl.store(scratch_local, data, mask=mask, cache_modifier=".wt")
        else:
            iris.store(
                scratch_local,
                data,
                iris_rank,
                dest_rank,
                heap_bases,
                mask=mask,
                hint=(1, BLOCK_SIZE_N),
            )


@triton.jit()
def persistent_reduce_scatter_push_reduce(
    scratch_ptr,
    output_ptr,
    M,
    N,
    stride_out_m,
    stride_out_n,
    group_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
):
    """
    Push (PUT) reduce-scatter — local reduce phase.

    This rank sums the world_size local scratch slots for the tiles it owns and
    writes them to output. Entirely local (no xGMI). Tile assignment matches the
    two_shot pull kernel.
    """
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    tiles_per_rank = tl.cdiv(total_tiles, world_size)

    acc_dtype = tl.float32 if output_ptr.type.element_ty != tl.int8 else tl.int32

    if DISTRIBUTION == 0:
        start_tile = group_rank
        stride = world_size
        remaining = tl.maximum(total_tiles - start_tile, 0)
        max_tile_offset = tl.cdiv(remaining, stride)
    else:
        start_tile = group_rank * tiles_per_rank
        stride = 1
        remaining = tl.maximum(total_tiles - start_tile, 0)
        max_tile_offset = tl.minimum(tiles_per_rank, remaining)

    for tile_offset in range(pid, max_tile_offset, COMM_SMS):
        tile_id = start_tile + tile_offset * stride

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for s in tl.static_range(world_size):
            rm_scratch = rm + s * M
            scratch_offset = rm_scratch[:, None] * N + rn[None, :]
            partial = tl.load(scratch_ptr + scratch_offset, mask=mask, other=0.0)
            acc += partial.to(acc_dtype)

        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        tl.store(output_ptr + output_offset, acc.to(output_ptr.type.element_ty), mask=mask)


# Cache for push-variant scratch buffers, keyed by (id(shmem), M, N, dtype, W).
_RS_SCRATCH_CACHE: dict = {}

# Byte threshold (full-buffer M*N*elem) at/above which the write-based
# two_shot_push beats the read-based two_shot for reduce_scatter. Measured on
# 8x MI300X (bf16, comm_sms=64, device barrier): read wins through 8 MiB
# (1.26x at 8 MiB), push wins from 32 MiB up (1.03x). Crossover set at 16 MiB.
_RS_AUTO_PUSH_BYTES = 16 << 20


def _get_rs_scratch(shmem, M, N, dtype, world_size):
    key = (id(shmem), M, N, dtype, world_size)
    buf = _RS_SCRATCH_CACHE.get(key)
    if buf is None or buf.shape != (world_size * M, N) or buf.dtype != dtype:
        buf = shmem.zeros((world_size * M, N), dtype=dtype)
        _RS_SCRATCH_CACHE[key] = buf
    return buf


def reduce_scatter(
    output_tensor,
    input_tensor,
    shmem,
    op=ReduceOp.SUM,
    group=None,
    async_op=False,
    config=None,
):
    """
    Internal reduce-scatter collective operation implementation.

    This function is called internally by shmem.ccl.reduce_scatter().
    Users should use the Iris instance method instead:
        >>> shmem.ccl.reduce_scatter(output_tensor, input_tensor)

    Each rank reduces its assigned tiles from all ranks' inputs and stores
    the result only to its own output tensor. This is similar to all-reduce
    but without broadcasting the result to all ranks.

    Args:
        output_tensor: Output tensor of shape (M, N) - will contain reduced tiles for this rank
        input_tensor: Input tensor of shape (M, N) - local rank's partial data
        shmem: Iris shmem context
        op: Reduction operation to apply. Currently only ReduceOp.SUM is supported.
            Default: ReduceOp.SUM.
        group: ProcessGroup or None. If None, uses all ranks in shmem context.
               Default: None.
        async_op: If False, performs a barrier at the end. If True, returns immediately.
                  Default: False.
        config: Config instance with kernel parameters (default: None).
                If None, uses default Config values.
                Only supports reduce_scatter_variant="two_shot".

    Example:
        >>> shmem = iris.iris()
        >>> shmem.ccl.reduce_scatter(output_tensor, input_tensor)

        >>> # Custom configuration
        >>> from iris.ccl import Config
        >>> config = Config(reduce_scatter_variant="two_shot", all_reduce_distribution=1)
        >>> shmem.ccl.reduce_scatter(output_tensor, input_tensor, config=config)
    """
    # Validate op parameter
    if op != ReduceOp.SUM:
        raise ValueError(
            f"Only ReduceOp.SUM is currently supported, got {op}. "
            "Support for other operations (PRODUCT, MAX, MIN, etc.) will be added in a future release."
        )
    if config is None:
        config = Config(block_size_m=32, block_size_n=64, all_reduce_distribution=1)

    # Check for unsupported options
    if config.use_gluon:
        raise ValueError(
            "reduce_scatter does not support use_gluon=True. "
            "Gluon implementation is not available for reduce_scatter. "
            "Use default config (use_gluon=False)."
        )

    # Validate variant
    variant = getattr(config, "reduce_scatter_variant", "two_shot")
    if variant not in ("two_shot", "push", "two_shot_push", "auto"):
        raise ValueError(
            f"reduce_scatter only supports variant='two_shot', 'push', 'two_shot_push', or 'auto', got '{variant}'."
        )

    # Extract group information
    # rank_in_group: position within the group (0, 1, 2, ...) - used for tile assignment
    # rank_global: global rank in iris context - passed as iris_rank to kernel for RMA operations
    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, shmem)
    M, N = input_tensor.shape[:2]

    # Size- and world-size-based variant selection (mirrors all_reduce 'auto').
    # The write-based two_shot_push is a 2-kernel + 2-barrier path that wins on
    # bandwidth at scale; the read-based two_shot is a single kernel + single
    # barrier that wins in the latency regime. Critically, push only amortizes
    # its fixed overhead when the read path's remote fan-in (world_size-1 remote
    # reads per tile) is large enough to create LFIFO contention. Measured on
    # 8x MI300X (bf16): at world_size<=2 (a single remote) push NEVER beats read,
    # even at 128 MiB; at world_size>=4 the full-buffer crossover is ~16 MiB.
    if variant == "auto":
        nbytes = M * N * input_tensor.element_size()
        push_ok = (
            (M % world_size == 0) and ((M // world_size) % config.block_size_m == 0) and (N % config.block_size_n == 0)
        )
        if push_ok and world_size >= 4 and nbytes >= _RS_AUTO_PUSH_BYTES:
            variant = "two_shot_push"
        else:
            variant = "two_shot"

    # Validate output shape matches input shape
    if output_tensor.shape[:2] != (M, N):
        raise ValueError(
            f"Output tensor shape {output_tensor.shape[:2]} does not match input shape {(M, N)}. "
            f"For reduce-scatter, output should have the same shape as input."
        )

    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    heap_bases = shmem.get_heap_bases()

    # Use all_reduce_distribution for tile distribution
    distribution = config.all_reduce_distribution

    if variant == "two_shot_push":
        if M % world_size != 0:
            raise ValueError(f"two_shot_push requires M ({M}) divisible by world_size ({world_size})")
        Mb = M // world_size
        if Mb % config.block_size_m != 0:
            raise ValueError(f"two_shot_push requires M/world_size ({Mb}) divisible by block_size_m")
        if N % config.block_size_n != 0:
            raise ValueError(f"two_shot_push requires N ({N}) divisible by block_size_n")
        # (M, N) symmetric scratch. Barrier once on fresh allocation so all ranks
        # have registered the symmetric slot before any remote store targets it.
        _key = (id(shmem), M, N, input_tensor.dtype, 1)
        _newly = _key not in _RS_SCRATCH_CACHE
        scratch = _get_rs_scratch(shmem, M, N, input_tensor.dtype, 1)
        if _newly:
            shmem.barrier()
        nw = getattr(config, "all_reduce_num_warps", 8)
        # Phase 1: owner-rotated remote WRITES of each row-block into owners' scratch.
        persistent_all_reduce_ts_push_scatter[(config.comm_sms,)](
            input_tensor,
            scratch,
            M,
            N,
            stride_in_m,
            stride_in_n,
            heap_bases,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            config.block_size_m,
            config.block_size_n,
            config.comm_sms,
            num_warps=nw,
        )
        if getattr(config, "barrier_mode", "host") == "device":
            shmem.device_barrier(group=group)
        else:
            shmem.barrier()
        # Phase 2: local reduce of owned block (own from input, rest from scratch).
        persistent_reduce_scatter_v2_reduce[(config.comm_sms,)](
            scratch,
            output_tensor,
            input_tensor,
            M,
            N,
            stride_out_m,
            stride_out_n,
            stride_in_m,
            stride_in_n,
            rank_in_group,
            world_size,
            config.block_size_m,
            config.block_size_n,
            config.comm_sms,
            num_warps=nw,
        )
        if not async_op:
            if getattr(config, "barrier_mode", "host") == "device":
                shmem.device_barrier(group=group)
            else:
                shmem.barrier()
        return

    if variant == "push":
        scratch = _get_rs_scratch(shmem, M, N, input_tensor.dtype, world_size)
        scratch.zero_()
        shmem.barrier()
        # Phase 1: scatter every input tile to its owner's scratch (remote WRITE).
        persistent_reduce_scatter_push_scatter[(config.comm_sms,)](
            input_tensor,
            scratch,
            M,
            N,
            stride_in_m,
            stride_in_n,
            heap_bases,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            distribution,
            num_stages=config.num_stages,
            num_warps=config.num_warps,
            waves_per_eu=config.waves_per_eu,
        )
        shmem.barrier()
        # Phase 2: owner reduces its world_size local slots (local only).
        persistent_reduce_scatter_push_reduce[(config.comm_sms,)](
            scratch,
            output_tensor,
            M,
            N,
            stride_out_m,
            stride_out_n,
            rank_in_group,
            world_size,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            distribution,
            num_stages=config.num_stages,
            num_warps=config.num_warps,
            waves_per_eu=config.waves_per_eu,
        )
        if not async_op:
            if getattr(config, "barrier_mode", "host") == "device":
                shmem.device_barrier(group=group)
            else:
                shmem.barrier()
        return

    # Correctness guard for the m-grouping swizzle: with DISTRIBUTION=1 each rank
    # claims a CONTIGUOUS tile_id range and writes the reduced tiles back to those
    # same (pid_m, pid_n) positions in its own output. The group swizzle reorders
    # tiles within blocks of (GROUP_SIZE_M * num_pid_n) tile_ids. If a group spans
    # more row-tiles than a rank owns (GROUP_SIZE_M > num_pid_m // world_size) the
    # swizzle crosses rank boundaries and a rank writes rows it does not own,
    # leaving its real output rows unwritten (silent corruption, observed at
    # M=512/768 on ws=8). Clamp the effective group to the largest divisor of the
    # per-rank row-tile count that does not exceed swizzle_size; this preserves
    # the L2-locality swizzle when it fits and degrades to row-major (1) otherwise.
    num_pid_m = (M + config.block_size_m - 1) // config.block_size_m
    eff_swizzle = config.swizzle_size
    if num_pid_m % world_size == 0:
        rows_per_rank = num_pid_m // world_size
        eff_swizzle = 1
        for d in range(min(config.swizzle_size, rows_per_rank), 0, -1):
            if rows_per_rank % d == 0:
                eff_swizzle = d
                break
    else:
        # Per-rank row ownership is not integral; row-major is the only safe order.
        eff_swizzle = 1

    persistent_reduce_scatter_two_shot[(config.comm_sms,)](
        input_tensor,
        output_tensor,
        M,
        N,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        heap_bases,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        config.block_size_m,
        config.block_size_n,
        eff_swizzle,
        config.comm_sms,
        config.num_xcds,
        config.chunk_size,
        distribution,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
    )

    if not async_op:
        # Cross-rank completion barrier. "device" mode uses an on-GPU,
        # stream-ordered atomic barrier (no ~370us host round-trip); "host"
        # preserves torch.cuda.synchronize() + distributed.barrier().
        if getattr(config, "barrier_mode", "host") == "device":
            shmem.device_barrier(group=group)
        else:
            shmem.barrier()
