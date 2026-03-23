# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused All-Gather + GEMM for the column-parallel pattern (M-sharded
activation, N-sharded weight).

Column-parallel layout:
  - A_local[M/ws, K] per GPU  -> gather along M -> staged_a[M, K]
  - B_local[K, N/ws] per GPU  (no gather needed)
  - C_local[M, N/ws] output   (partial, distributed)

Supports two modes:
  1. split_kernels=True: Two independent kernels on separate CUDA streams
  2. split_kernels=False (default): Single fused kernel with interleaved PID layout

Producer-consumer synchronization via per-(m_tile, k_flag_group) integer flags
in HBM, using .cg stores + release/acquire atomics.
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris

from iris.device_utils import read_realtime
from iris.tracing.events import TraceEvent
from .config import FusedConfig
from .workspace import FusedWorkspace


# =========================================================================
# Kernel 1: Fetch-only (producer) — for split-kernel mode
# =========================================================================

@triton.jit
def _col_parallel_fetch_kernel(
    A_sharded, staged_a, flags_ptr,
    M, K, M_local,
    stride_am, stride_ak, stride_sa_m, stride_sa_k,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr, world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    FETCH_SMS: tl.constexpr,
    NUM_M_TILES: tl.constexpr, NUM_K_BLOCKS: tl.constexpr,
    NUM_M_TILES_LOCAL: tl.constexpr,
    K_PER_FLAG: tl.constexpr, NUM_FLAG_GROUPS_K: tl.constexpr,
    TOTAL_FLAG_GROUPS: tl.constexpr,
):
    """Persistent PUSH fetch kernel: reads local A, writes to all ranks' staged_a.
    Uses hoisted heap_bases + raw tl.store to pipeline stores across ranks.
    All heap_base loads are hoisted to kernel start to eliminate vmcnt(0) serialization."""
    pid = tl.program_id(0)
    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size, tracing=False)
    heap_bases = ctx.heap_bases

    # Hoist ALL heap_base loads to kernel start — load once, reuse forever.
    # This eliminates the per-rank tl.load(heap_bases + target_rank) that
    # causes s_waitcnt vmcnt(0) between ranks in the inner loop.
    from_base = tl.load(heap_bases + cur_rank)
    # Precompute base difference for each rank (to_base - from_base)
    # so inner loop only needs: remote_ptr = staged_ptrs + base_diff[rank]
    # Using tl.static_range to ensure full unrolling at compile time.
    base_diff_0 = tl.load(heap_bases + 0).to(tl.int64) - from_base.to(tl.int64)
    base_diff_1 = tl.load(heap_bases + 1).to(tl.int64) - from_base.to(tl.int64)
    base_diff_2 = tl.load(heap_bases + 2).to(tl.int64) - from_base.to(tl.int64)
    base_diff_3 = tl.load(heap_bases + 3).to(tl.int64) - from_base.to(tl.int64)
    base_diff_4 = tl.load(heap_bases + 4).to(tl.int64) - from_base.to(tl.int64)
    base_diff_5 = tl.load(heap_bases + 5).to(tl.int64) - from_base.to(tl.int64)
    base_diff_6 = tl.load(heap_bases + 6).to(tl.int64) - from_base.to(tl.int64)
    base_diff_7 = tl.load(heap_bases + 7).to(tl.int64) - from_base.to(tl.int64)

    for fg_idx in range(pid, TOTAL_FLAG_GROUPS, FETCH_SMS):
        m_tile_local = fg_idx // NUM_FLAG_GROUPS_K
        k_flag_group = fg_idx % NUM_FLAG_GROUPS_K
        m_tile_local = min(m_tile_local, NUM_M_TILES_LOCAL - 1)
        k_block_start = k_flag_group * K_PER_FLAG

        # Global m_tile index for this rank's shard in staged_a
        m_tile_global = cur_rank * NUM_M_TILES_LOCAL + m_tile_local

        # Local A_sharded row indices
        rm_local = m_tile_local * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm_local = tl.max_contiguous(tl.multiple_of(rm_local, BLOCK_SIZE_M), BLOCK_SIZE_M)

        # Global staged_a row indices
        rm_global = m_tile_global * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm_global = tl.max_contiguous(tl.multiple_of(rm_global, BLOCK_SIZE_M), BLOCK_SIZE_M)

        for k_off in range(K_PER_FLAG):
            k_block = k_block_start + k_off
            rk = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)

            # Load from local A_sharded
            a_ptrs = A_sharded + rm_local.to(tl.int64)[:, None] * stride_am + rk[None, :] * stride_ak
            data = tl.load(a_ptrs)

            # Destination pointers in staged_a (using global m_tile)
            staged_ptrs = staged_a + rm_global.to(tl.int64)[:, None] * stride_sa_m + rk[None, :] * stride_sa_k

            # Local store (fast, no XGMI)
            tl.store(staged_ptrs, data, cache_modifier=".cg")

            # Remote stores: add precomputed base_diff to local pointers.
            # No heap_base loads in the inner loop = no vmcnt(0) between ranks.
            ptr_int = tl.cast(staged_ptrs, tl.uint64)
            for const_r in tl.static_range(world_size):
                target_rank = (pid + const_r) % world_size
                if target_rank != cur_rank:
                    # Select precomputed base_diff for this rank
                    if target_rank == 0:
                        diff = base_diff_0
                    elif target_rank == 1:
                        diff = base_diff_1
                    elif target_rank == 2:
                        diff = base_diff_2
                    elif target_rank == 3:
                        diff = base_diff_3
                    elif target_rank == 4:
                        diff = base_diff_4
                    elif target_rank == 5:
                        diff = base_diff_5
                    elif target_rank == 6:
                        diff = base_diff_6
                    else:
                        diff = base_diff_7
                    remote_int = (ptr_int.to(tl.int64) + diff).to(tl.uint64)
                    remote_ptr = tl.cast(remote_int, staged_ptrs.dtype)
                    remote_ptr = tl.max_contiguous(tl.multiple_of(remote_ptr, (1, BLOCK_SIZE_K)), (1, BLOCK_SIZE_K))
                    tl.store(remote_ptr, data)

        # Signal flags on all ranks — same per-WG rotation
        flag_idx = m_tile_global * NUM_FLAG_GROUPS_K + k_flag_group
        tl.atomic_xchg(flags_ptr + flag_idx, 1, sem="release", scope="gpu")
        for i in range(world_size):
            target_rank = (pid + i) % world_size
            if target_rank != cur_rank:
                ctx.atomic_xchg(flags_ptr + flag_idx, 1, to_rank=target_rank, sem="release", scope="gpu")


# =========================================================================
# Kernel 2: GEMM-only (consumer) — for split-kernel mode
# =========================================================================

@triton.jit
def _col_parallel_gemm_kernel(
    staged_a, B, C, bias_ptr, flags_ptr,
    M, N_LOCAL, K,
    stride_sa_m, stride_sa_k, stride_bk, stride_bn,
    stride_cm, stride_cn, stride_bias,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    GEMM_SMS: tl.constexpr,
    NUM_M_TILES: tl.constexpr, NUM_TILES_N: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    K_PER_FLAG: tl.constexpr, NUM_FLAG_GROUPS_K: tl.constexpr,
    BIAS: tl.constexpr, ALLOW_TF32: tl.constexpr,
):
    """Persistent GEMM kernel: polls flags, loads gathered A, computes C tiles."""
    pid = tl.program_id(0)
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32
    total_tiles = NUM_M_TILES * NUM_TILES_N

    for tile_id in range(pid, total_tiles, GEMM_SMS):
        num_pid_in_group = GROUP_SIZE_M * NUM_TILES_N
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        first_pid_m = min(first_pid_m, NUM_M_TILES - 1)
        group_sz = min(NUM_M_TILES - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_sz)
        pid_n = (tile_id % num_pid_in_group) // group_sz
        pid_m = min(pid_m, NUM_M_TILES - 1)

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N_LOCAL, BLOCK_SIZE_N), BLOCK_SIZE_N)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for k_fg in range(NUM_FLAG_GROUPS_K):
            flag_idx = pid_m * NUM_FLAG_GROUPS_K + k_fg
            while tl.atomic_add(flags_ptr + flag_idx, 0, sem="acquire", scope="gpu") == 0:
                pass

            k_block_base = k_fg * K_PER_FLAG
            for k_off in range(K_PER_FLAG):
                k_block = k_block_base + k_off
                rk = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)

                a_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + rk[None, :] * stride_sa_k
                a = tl.load(a_ptrs)
                B_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
                b = tl.load(B_ptrs)

                if ALLOW_TF32:
                    acc = tl.dot(a, b, acc, allow_tf32=True)
                else:
                    acc += tl.dot(a, b, allow_tf32=False)

        if BIAS:
            bias_val = tl.load(bias_ptr + rm * stride_bias, mask=rm < M, other=0.0)
            acc = acc + bias_val[:, None]

        c = acc.to(C.type.element_ty)
        C_ptrs = C + rm.to(tl.int64)[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_ptrs, c)


# =========================================================================
# Fused kernel — interleaved fetch+GEMM WGs (concurrent execution)
# =========================================================================

@triton.jit
def _col_parallel_all_gather_matmul_kernel(
    A_sharded, B, C, bias_ptr, staged_a, flags_ptr,
    M, N_LOCAL, K, M_local,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn, stride_sa_m, stride_sa_k, stride_bias,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr, world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    NUM_FETCH_SMS: tl.constexpr,
    NUM_M_TILES: tl.constexpr, NUM_TILES_N: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr, NUM_M_TILES_LOCAL: tl.constexpr,
    K_PER_FLAG: tl.constexpr, NUM_FLAG_GROUPS_K: tl.constexpr,
    TOTAL_GATHER_TILES: tl.constexpr,
    BIAS: tl.constexpr, ALLOW_TF32: tl.constexpr,
    NUM_FETCH_STAGES: tl.constexpr, GEMM_TILES_PER_STAGE: tl.constexpr,
    FIRST_STAGE_FETCH_SMS: tl.constexpr,
    FETCH_PIPE_DEPTH: tl.constexpr,
    GEMM_WGS: tl.constexpr,
    TOTAL_GEMM_TILES: tl.constexpr,
    TRACE: tl.constexpr,
):
    """
    Interleaved col-parallel AG+GEMM kernel with dedicated fetch and GEMM WGs.

    Grid layout (interleaved by stage):
      [fetch0 (P)] [gemm0 (G)] [fetch1 (F)] [gemm1 (G)] ...
    P = FIRST_STAGE_FETCH_SMS, F = NUM_FETCH_SMS, G = GEMM_TILES_PER_STAGE

    Each stage owns a contiguous range of M-tiles. Fetch WGs gather tiles
    for their stage; GEMM WGs poll flags and compute as tiles arrive.
    This enables concurrent fetch and GEMM across stages.
    """
    pid = tl.program_id(0)
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32
    zero = tl.program_id(0) * 0

    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size, tracing=TRACE)

    # Interleaved layout with asymmetric first stage:
    #   [fetch0 (P)] [gemm0 (G)] [fetch1 (F)] [gemm1 (G)] ...
    # P = FIRST_STAGE_FETCH_SMS, F = NUM_FETCH_SMS, G = GEMM_TILES_PER_STAGE
    FIRST_STAGE_SIZE: tl.constexpr = FIRST_STAGE_FETCH_SMS + GEMM_TILES_PER_STAGE
    REST_STAGE_SIZE: tl.constexpr = NUM_FETCH_SMS + GEMM_TILES_PER_STAGE
    M_PER_STAGE: tl.constexpr = (NUM_M_TILES + NUM_FETCH_STAGES - 1) // NUM_FETCH_STAGES

    # Two-phase decode: stage 0 has a different size than subsequent stages
    if pid < FIRST_STAGE_SIZE:
        my_stage = zero
        local_pid = pid
        fetch_threshold = zero + FIRST_STAGE_FETCH_SMS
    else:
        adjusted = pid - FIRST_STAGE_SIZE
        my_stage = 1 + adjusted // REST_STAGE_SIZE
        local_pid = adjusted % REST_STAGE_SIZE
        fetch_threshold = zero + NUM_FETCH_SMS

    if local_pid < fetch_threshold:
        # ==============================================================
        # FETCHER — PUSH: read local A, write to all ranks' staged_a
        # ==============================================================
        stage_pid = local_pid

        # Hoist ALL heap_base loads — precompute base_diff for each rank.
        # Eliminates per-rank tl.load(heap_bases) in inner loop, avoiding
        # s_waitcnt vmcnt(0) serialization between ranks.
        heap_bases = ctx.heap_bases
        from_base = tl.load(heap_bases + cur_rank)
        base_diff_0 = tl.load(heap_bases + 0).to(tl.int64) - from_base.to(tl.int64)
        base_diff_1 = tl.load(heap_bases + 1).to(tl.int64) - from_base.to(tl.int64)
        base_diff_2 = tl.load(heap_bases + 2).to(tl.int64) - from_base.to(tl.int64)
        base_diff_3 = tl.load(heap_bases + 3).to(tl.int64) - from_base.to(tl.int64)
        base_diff_4 = tl.load(heap_bases + 4).to(tl.int64) - from_base.to(tl.int64)
        base_diff_5 = tl.load(heap_bases + 5).to(tl.int64) - from_base.to(tl.int64)
        base_diff_6 = tl.load(heap_bases + 6).to(tl.int64) - from_base.to(tl.int64)
        base_diff_7 = tl.load(heap_bases + 7).to(tl.int64) - from_base.to(tl.int64)

        if TRACE:
            _trace_handle = ctx.tracing.record_event_start(
                event_id=TraceEvent().wg_fetch, target_rank=cur_rank,
                address=flags_ptr + tl.arange(0, 1), pid_m=pid, pid_n=my_stage,
            )

        # PUSH model: all fetchers share a pool of LOCAL flag groups
        TOTAL_FG_LOCAL: tl.constexpr = NUM_M_TILES_LOCAL * NUM_FLAG_GROUPS_K

        for const_stage in range(NUM_FETCH_STAGES):
            if my_stage == const_stage:
                stage_fetch_sms = FIRST_STAGE_FETCH_SMS if const_stage == 0 else NUM_FETCH_SMS

                # In PUSH, stage boundaries map to local m-tile ranges
                LOCAL_M_PER_STAGE: tl.constexpr = (NUM_M_TILES_LOCAL + NUM_FETCH_STAGES - 1) // NUM_FETCH_STAGES
                stage_local_m_start = const_stage * LOCAL_M_PER_STAGE
                stage_local_m_count = min(LOCAL_M_PER_STAGE, NUM_M_TILES_LOCAL - stage_local_m_start)
                total_fg_stage = NUM_FLAG_GROUPS_K * stage_local_m_count

                for fg_idx in range(stage_pid, total_fg_stage, stage_fetch_sms):
                    # K-major ordering: cycle through k-groups first, m-tiles second.
                    k_flag_group = fg_idx // stage_local_m_count
                    m_in_stage = fg_idx % stage_local_m_count

                    m_tile_local = stage_local_m_start + m_in_stage
                    m_tile_local = min(m_tile_local, NUM_M_TILES_LOCAL - 1)

                    # Global m_tile for this rank's shard
                    m_tile_global = cur_rank * NUM_M_TILES_LOCAL + m_tile_local
                    k_block_start = k_flag_group * K_PER_FLAG

                    # Local A_sharded row indices
                    rm_local = m_tile_local * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    rm_local = tl.max_contiguous(tl.multiple_of(rm_local, BLOCK_SIZE_M), BLOCK_SIZE_M)

                    # Global staged_a row indices
                    rm_global = m_tile_global * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    rm_global = tl.max_contiguous(tl.multiple_of(rm_global, BLOCK_SIZE_M), BLOCK_SIZE_M)

                    for k_off in range(K_PER_FLAG):
                        k_block = k_block_start + k_off
                        rk = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                        rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)

                        # Load from local A_sharded
                        a_ptrs = A_sharded + rm_local.to(tl.int64)[:, None] * stride_am + rk[None, :] * stride_ak
                        data = tl.load(a_ptrs)

                        # Destination pointers in staged_a (using global m_tile)
                        staged_ptrs = staged_a + rm_global.to(tl.int64)[:, None] * stride_sa_m + rk[None, :] * stride_sa_k

                        # Local store (fast, no XGMI)
                        tl.store(staged_ptrs, data, cache_modifier=".cs")

                        # Remote stores with precomputed base_diff — no heap_base
                        # loads in inner loop, no vmcnt(0) between ranks.
                        ptr_int = tl.cast(staged_ptrs, tl.uint64)
                        for const_r in tl.static_range(world_size):
                            target_rank = (stage_pid + const_r) % world_size
                            if target_rank != cur_rank:
                                if target_rank == 0:
                                    diff = base_diff_0
                                elif target_rank == 1:
                                    diff = base_diff_1
                                elif target_rank == 2:
                                    diff = base_diff_2
                                elif target_rank == 3:
                                    diff = base_diff_3
                                elif target_rank == 4:
                                    diff = base_diff_4
                                elif target_rank == 5:
                                    diff = base_diff_5
                                elif target_rank == 6:
                                    diff = base_diff_6
                                else:
                                    diff = base_diff_7
                                remote_int = (ptr_int.to(tl.int64) + diff).to(tl.uint64)
                                remote_ptr = tl.cast(remote_int, staged_ptrs.dtype)
                                remote_ptr = tl.max_contiguous(tl.multiple_of(remote_ptr, (1, BLOCK_SIZE_K)), (1, BLOCK_SIZE_K))
                                tl.store(remote_ptr, data, cache_modifier=".cs")

                    # Signal flags on all ranks — same per-WG rotation
                    flag_idx = m_tile_global * NUM_FLAG_GROUPS_K + k_flag_group
                    tl.atomic_xchg(flags_ptr + flag_idx, 1, sem="release", scope="gpu")
                    for i in range(world_size):
                        target_rank = (stage_pid + i) % world_size
                        if target_rank != cur_rank:
                            ctx.atomic_xchg(flags_ptr + flag_idx, 1, to_rank=target_rank, sem="release", scope="gpu")

        if TRACE:
            ctx.tracing.record_event_end(_trace_handle)

    else:
        # ==============================================================
        # GEMM — compute output tiles matching this stage's fetch range
        # ==============================================================
        # Stage tile mapping: each fetch stage pushes LOCAL_M_PER_STAGE
        # local m-tiles from EACH rank. So the global m-tiles available
        # after stage s are:
        #   {r * NUM_M_TILES_LOCAL + s * LOCAL_M_PER_STAGE + offset}
        # for r in [0, world_size) and offset in [0, LOCAL_M_PER_STAGE).
        # We index linearly within this set (M_PER_STAGE tiles total)
        # and convert to the correct global pid_m.
        gemm_local_id = local_pid - fetch_threshold
        LOCAL_M_PER_STAGE: tl.constexpr = (NUM_M_TILES_LOCAL + NUM_FETCH_STAGES - 1) // NUM_FETCH_STAGES

        # nfs auto-correction ensures NUM_M_TILES_LOCAL % NUM_FETCH_STAGES == 0
        # so LOCAL_M_PER_STAGE is exact (no remainder stage).
        stage_local_m_start = my_stage * LOCAL_M_PER_STAGE
        M_PER_STAGE: tl.constexpr = LOCAL_M_PER_STAGE * world_size

        num_pid_in_group = GROUP_SIZE_M * NUM_TILES_N
        group_id = gemm_local_id // num_pid_in_group
        first_linear_m = group_id * GROUP_SIZE_M
        first_linear_m = min(first_linear_m, M_PER_STAGE - 1)
        group_sz = min(M_PER_STAGE - first_linear_m, GROUP_SIZE_M)
        linear_m = first_linear_m + ((gemm_local_id % num_pid_in_group) % group_sz)
        pid_n = (gemm_local_id % num_pid_in_group) // group_sz
        linear_m = min(linear_m, M_PER_STAGE - 1)

        # Convert linear stage index to global m-tile index
        stage_rank = linear_m // LOCAL_M_PER_STAGE
        stage_offset = linear_m % LOCAL_M_PER_STAGE
        pid_m = stage_rank * NUM_M_TILES_LOCAL + stage_local_m_start + stage_offset
        pid_m = min(pid_m, NUM_M_TILES - 1)

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N_LOCAL, BLOCK_SIZE_N), BLOCK_SIZE_N)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        if TRACE:
            _trace_handle = ctx.tracing.record_event_start(
                event_id=TraceEvent().wg_gemm, target_rank=cur_rank,
                address=flags_ptr + tl.arange(0, 1), pid_m=pid, pid_n=my_stage,
            )
            _wt = zero.to(tl.int64)

        for k_fg in range(NUM_FLAG_GROUPS_K):
            if TRACE:
                _ws = read_realtime()

            flag_idx = pid_m * NUM_FLAG_GROUPS_K + k_fg
            while tl.atomic_add(flags_ptr + flag_idx, 0, sem="acquire", scope="gpu") == 0:
                pass

            if TRACE:
                _wt = _wt + (read_realtime() - _ws)

            k_block_base = k_fg * K_PER_FLAG
            for k_off in range(K_PER_FLAG):
                k_block = k_block_base + k_off
                rk = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)

                a_ptrs = staged_a + rm.to(tl.int64)[:, None] * stride_sa_m + rk[None, :] * stride_sa_k
                a = tl.load(a_ptrs)
                B_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
                b = tl.load(B_ptrs)

                if ALLOW_TF32:
                    acc = tl.dot(a, b, acc, allow_tf32=True)
                else:
                    acc += tl.dot(a, b, allow_tf32=False)

        if BIAS:
            bias_val = tl.load(bias_ptr + rm * stride_bias, mask=rm < M, other=0.0)
            acc = acc + bias_val[:, None]

        c = acc.to(C.type.element_ty)
        C_ptrs = C + rm.to(tl.int64)[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_ptrs, c)

        if TRACE:
            ctx.tracing.record_event_end(_trace_handle)
            ctx.tracing.record_event_start(
                event_id=TraceEvent().wg_gemm_wait, target_rank=cur_rank,
                address=flags_ptr + tl.arange(0, 1), pid_m=pid, pid_n=_wt.to(tl.int32),
            )


# ==========================================================================
# Python API
# ==========================================================================


def all_gather_matmul_col_parallel_preamble(
    shmem, A_sharded: torch.Tensor, B_local: torch.Tensor,
    config: Optional[FusedConfig] = None, k_per_flag: int = 1,
    staged_a_layout: str = "k_contiguous",
) -> FusedWorkspace:
    if config is None:
        config = FusedConfig()

    M_local, K = A_sharded.shape
    K_b, N_local = B_local.shape
    world_size = shmem.get_num_ranks()
    M = M_local * world_size

    assert K_b == K, f"A K dim ({K}) != B K dim ({K_b})"
    assert K % config.block_size_k == 0
    assert M % config.block_size_m == 0
    assert M_local % config.block_size_m == 0

    num_m_tiles = M // config.block_size_m
    num_k_blocks = K // config.block_size_k
    assert num_k_blocks % k_per_flag == 0
    num_flag_groups_k = num_k_blocks // k_per_flag

    ws = FusedWorkspace(
        operation="all_gather_matmul_col_parallel",
        shape=(M, N_local, K), dtype=A_sharded.dtype,
        world_size=world_size, variant=f"col_parallel_{staged_a_layout}",
        prepared=True,
    )

    if staged_a_layout == "m_contiguous":
        storage = shmem.zeros((K, M), dtype=A_sharded.dtype)
        ws.aux_buffer = storage.T
    else:
        ws.aux_buffer = shmem.zeros((M, K), dtype=A_sharded.dtype)

    ws.locks = shmem.zeros((num_m_tiles * num_flag_groups_k,), dtype=torch.int32)

    buffer_mb = M * K * A_sharded.element_size() / (1024**2)
    sa_stride_m, sa_stride_k = ws.aux_buffer.stride()
    shmem.info(
        f"Col-parallel HBM buffer: staged_a=({M},{K}) [{buffer_mb:.1f} MB] "
        f"layout={staged_a_layout} strides=({sa_stride_m},{sa_stride_k}), "
        f"flags={num_m_tiles}x{num_flag_groups_k}, k_per_flag={k_per_flag}"
    )

    shmem.barrier()
    return ws


_WG_FETCH = 14
_WG_GEMM = 15
_WG_GEMM_WAIT = 16


def _extract_wg_trace(shmem, grid_size, **metadata):
    import numpy as np
    bufs = shmem.tracing.trace_buffers
    n = min(shmem.tracing.trace_counter.item(), shmem.tracing.max_events)

    event_ids = bufs["event_id"][:n].cpu().numpy()
    pids = bufs["pid"][:n].cpu().numpy()
    timestamps = bufs["timestamp"][:n].cpu().numpy().astype(np.int64)
    end_ts = bufs["duration_cycles"][:n].cpu().numpy().astype(np.int64)
    xcc_ids = bufs["xcc_id"][:n].cpu().numpy().astype(np.int32)
    pid_ns = bufs["pid_n"][:n].cpu().numpy()

    starts = torch.zeros(grid_size, dtype=torch.int64)
    ends = torch.zeros(grid_size, dtype=torch.int64)
    waits = torch.zeros(grid_size, dtype=torch.int64)
    xcds = torch.zeros(grid_size, dtype=torch.int32)

    for i in range(n):
        eid = int(event_ids[i])
        wg = int(pids[i])
        if wg >= grid_size:
            continue
        if eid == _WG_FETCH or eid == _WG_GEMM:
            starts[wg] = int(timestamps[i])
            ends[wg] = int(end_ts[i])
            xcds[wg] = int(xcc_ids[i])
        elif eid == _WG_GEMM_WAIT:
            waits[wg] = int(pid_ns[i])

    return {"start": starts, "end": ends, "wait": waits, "xcd": xcds, "grid_size": grid_size, **metadata}


def all_gather_matmul_col_parallel(
    shmem, output_tensor: torch.Tensor, A_sharded: torch.Tensor,
    B_local: torch.Tensor, bias: Optional[torch.Tensor] = None,
    async_op: bool = False, config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
    num_fetch_sms: Optional[int] = None, k_per_flag: int = 1,
    staged_a_layout: str = "k_contiguous",
    num_warps: Optional[int] = None, num_stages: Optional[int] = None,
    num_fetch_stages: int = 8,
    first_stage_fetch_sms: Optional[int] = None,
    fetch_pipe_depth: int = 4, trace: bool = False,
    split_kernels: bool = False, gemm_sms: Optional[int] = None,
    gemm_wgs: Optional[int] = None,
    pure_fetch_first_stage: Optional[bool] = None,
) -> FusedWorkspace:
    if config is None:
        config = FusedConfig()

    M_local, K = A_sharded.shape
    K_b, N_local = B_local.shape
    world_size = shmem.get_num_ranks()
    rank = shmem.get_rank()
    M = M_local * world_size

    assert K_b == K
    assert output_tensor.shape == (M, N_local)
    assert M % config.block_size_m == 0
    assert K % config.block_size_k == 0
    assert M_local % config.block_size_m == 0
    assert N_local % config.block_size_n == 0, \
        f"N_local ({N_local}) must be divisible by block_size_n ({config.block_size_n})"

    num_k_blocks = K // config.block_size_k
    assert num_k_blocks % k_per_flag == 0

    if workspace is None:
        workspace = all_gather_matmul_col_parallel_preamble(
            shmem, A_sharded, B_local, config, k_per_flag, staged_a_layout
        )

    workspace.locks.zero_()

    stride_am, stride_ak = A_sharded.stride()
    stride_bk, stride_bn = B_local.stride()
    stride_cm, stride_cn = output_tensor.stride()
    stride_sa_m, stride_sa_k = workspace.aux_buffer.stride()

    if bias is not None:
        assert bias.shape[0] == M
        bias_ptr = bias
        stride_bias = bias.stride()[0] if bias.dim() > 0 else 1
        use_bias = True
    else:
        bias_ptr = output_tensor
        stride_bias = 1
        use_bias = False

    device = A_sharded.device
    total_sms = config.num_sms
    if total_sms is None:
        props = torch.cuda.get_device_properties(device)
        total_sms = props.multi_processor_count

    num_m_tiles = M // config.block_size_m
    num_m_tiles_local = M_local // config.block_size_m
    num_tiles_n = (N_local + config.block_size_n - 1) // config.block_size_n
    num_flag_groups_k = num_k_blocks // k_per_flag
    # PUSH model: each GPU pushes only its own local tiles to all ranks
    total_flag_groups = num_m_tiles_local * num_flag_groups_k

    if split_kernels:
        # ============================================================
        # SPLIT KERNEL PATH
        # ============================================================
        if num_fetch_sms is None:
            num_fetch_sms = 200
        if gemm_sms is None:
            gemm_sms = total_sms - num_fetch_sms
            if gemm_sms <= 0:
                gemm_sms = total_sms // 3
                num_fetch_sms = total_sms - gemm_sms

        fetch_stream = torch.cuda.Stream(device=device)
        gemm_stream = torch.cuda.Stream(device=device)

        with torch.cuda.stream(fetch_stream):
            _col_parallel_fetch_kernel[(num_fetch_sms,)](
                A_sharded, workspace.aux_buffer, workspace.locks,
                M, K, M_local, stride_am, stride_ak, stride_sa_m, stride_sa_k,
                shmem.get_device_context(), rank, world_size,
                config.block_size_m, config.block_size_k,
                num_fetch_sms, num_m_tiles, num_k_blocks,
                num_m_tiles_local, k_per_flag, num_flag_groups_k,
                total_flag_groups, num_warps=4,
            )

        gemm_launch_kwargs = {"matrix_instr_nonkdim": 16}
        if num_warps is not None:
            gemm_launch_kwargs["num_warps"] = num_warps
        if num_stages is not None:
            gemm_launch_kwargs["num_stages"] = num_stages

        with torch.cuda.stream(gemm_stream):
            _col_parallel_gemm_kernel[(gemm_sms,)](
                workspace.aux_buffer, B_local, output_tensor, bias_ptr,
                workspace.locks, M, N_local, K,
                stride_sa_m, stride_sa_k, stride_bk, stride_bn,
                stride_cm, stride_cn, stride_bias,
                config.block_size_m, config.block_size_n,
                config.block_size_k, config.group_size_m,
                gemm_sms, num_m_tiles, num_tiles_n, num_k_blocks,
                k_per_flag, num_flag_groups_k, use_bias, config.allow_tf32,
                **gemm_launch_kwargs,
            )

        if not async_op:
            shmem.barrier()

    else:
        # ============================================================
        # FUSED KERNEL PATH — interleaved fetch+GEMM WGs
        # ============================================================
        # Interleaved PID layout: [fetch0][gemm0][fetch1][gemm1]...
        # Dedicated fetch and GEMM WGs enable concurrent execution:
        # while fetchers gather M-tiles for stage N+1, GEMM WGs
        # compute output tiles for stage N.

        if num_fetch_sms is None:
            num_fetch_sms = max(1, total_sms // 10)

        if first_stage_fetch_sms is None:
            first_stage_fetch_sms = num_fetch_sms

        assert num_fetch_stages >= 1
        num_m_tiles_local = M_local // config.block_size_m
        if num_m_tiles_local % num_fetch_stages != 0:
            valid = [s for s in range(1, num_m_tiles_local+1) if num_m_tiles_local % s == 0 and s <= 16]
            # Auto-select closest valid nfs
            best = max((s for s in valid if s <= num_fetch_stages), default=valid[-1])
            shmem.info(f"nfs={num_fetch_stages} invalid for {num_m_tiles_local} M-tiles, "
                       f"auto-correcting to nfs={best} (valid: {valid})")
            num_fetch_stages = best

        total_gemm_tiles = num_m_tiles * num_tiles_n

        # Interleaved layout: [fetch0 (P)] [gemm0 (G)] [fetch1 (F)] [gemm1 (G)] ...
        m_per_stage = (num_m_tiles + num_fetch_stages - 1) // num_fetch_stages
        gemm_tiles_per_stage = m_per_stage * num_tiles_n

        first_stage_size = first_stage_fetch_sms + gemm_tiles_per_stage
        rest_stage_size = num_fetch_sms + gemm_tiles_per_stage
        total_fetch_wgs = first_stage_fetch_sms + num_fetch_sms * max(0, num_fetch_stages - 1)
        grid_size = first_stage_size + rest_stage_size * max(0, num_fetch_stages - 1)

        # gemm_wgs is not used in the interleaved layout (kept for API compat)
        if gemm_wgs is None:
            gemm_wgs = gemm_tiles_per_stage

        total_gather_tiles = num_m_tiles * num_k_blocks

        if trace:
            max_trace_events = grid_size * 4
            if not shmem.tracing.enabled:
                shmem.tracing.enable(max_events=max_trace_events)
            else:
                shmem.tracing.reset()

        launch_kwargs = {"matrix_instr_nonkdim": 16}
        if num_warps is not None:
            launch_kwargs["num_warps"] = num_warps
        if num_stages is not None:
            launch_kwargs["num_stages"] = num_stages

        _col_parallel_all_gather_matmul_kernel[(grid_size,)](
            A_sharded, B_local, output_tensor, bias_ptr,
            workspace.aux_buffer, workspace.locks,
            M, N_local, K, M_local,
            stride_am, stride_ak, stride_bk, stride_bn,
            stride_cm, stride_cn, stride_sa_m, stride_sa_k, stride_bias,
            shmem.get_device_context(), rank, world_size,
            config.block_size_m, config.block_size_n,
            config.block_size_k, config.group_size_m,
            num_fetch_sms, num_m_tiles, num_tiles_n, num_k_blocks,
            num_m_tiles_local, k_per_flag, num_flag_groups_k,
            total_gather_tiles, use_bias, config.allow_tf32,
            num_fetch_stages, gemm_tiles_per_stage,
            first_stage_fetch_sms, fetch_pipe_depth,
            gemm_wgs, total_gemm_tiles, trace,
            **launch_kwargs,
        )

        if not async_op:
            shmem.barrier()

        if trace:
            torch.cuda.synchronize()
            workspace.trace_data = _extract_wg_trace(
                shmem, grid_size,
                num_fetch_sms=num_fetch_sms,
                num_fetch_stages=num_fetch_stages,
                total_fetch_wgs=total_fetch_wgs,
                num_m_tiles=num_m_tiles,
                num_tiles_n=num_tiles_n,
                first_stage_fetch_sms=first_stage_fetch_sms,
                first_stage_size=first_stage_size,
                rest_stage_size=rest_stage_size,
                gemm_tiles_per_stage=gemm_tiles_per_stage,
            )

    return workspace
