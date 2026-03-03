# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from iris import DeviceContext, TraceEvent
from iris.device_utils import read_realtime


@triton.jit()
def persistent_gemm_all_scatter(
    A,
    B,
    C,
    c_global,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_cm_global,
    stride_cn_global,
    stride_bias,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    TRACING: tl.constexpr = False,
    COLLECT_TIMESTAMPS: tl.constexpr = False,
    mm_begin_timestamp_ptr: tl.tensor = None,
    mm_end_timestamp_ptr: tl.tensor = None,
):
    # Initialize DeviceContext with tracing
    ctx = DeviceContext.initialize(context_tensor, cur_rank, world_size, tracing=TRACING)

    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    for tile_id in range(pid, total_tiles, NUM_SMS):
        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_min(mm_begin_timestamp_ptr + tile_id, timestamp)

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        # Accumulator registers with C results
        c = acc.to(C.type.element_ty)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        # Add compiler hints
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Define the C-mask (BLOCK_SIZE_M, 1) x (1, BLOCK_SIZE_N)
        sub_mask = (rm[:, None] < M) & (rn[None, :] < N)

        # Calculate the "global" offset of C based on the rank.
        # Note how the N-dimension is being multiplied by current rank.
        # This is because each rank is computing a portion of the N-dimension
        # locally and then scattering it to all other ranks to complete
        # the global N-dimension.
        global_offset = rm[:, None] * stride_cm_global + (rn[None, :] + cur_rank * N) * stride_cn_global

        # Timestamp for GEMM before store
        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(mm_end_timestamp_ptr + tile_id, timestamp)

        # Store local result to C (needed by callers that consume the rank-local output).
        C_ptr = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_ptr, c, mask=sub_mask)

        # Scatter accumulator directly from registers to c_global on every rank.
        # Using ctx.store(pointer, value, to_rank) instead of ctx.put(from_ptr, to_ptr, to_rank)
        # avoids the unnecessary HBM roundtrip that ctx.put incurs:
        #   ctx.put  = tl.load(C_ptr)   ← HBM read (BLK_M*BLK_N fp16 elements per rank)
        #              + tl.store(remote)
        #   ctx.store = tl.store(remote, c)   ← scatter directly from accumulator registers
        # This eliminates 7 × BLK_M × BLK_N × 2 bytes of HBM reads per output tile.
        c_global_ptr = c_global + global_offset
        for remote_rank in range(world_size):
            if remote_rank == cur_rank:
                # For the current rank, apply alignment hint for the global C pointer so the
                # compiler can emit wider vector stores (same benefit as ctx.store hint below).
                c_global_hinted = tl.max_contiguous(tl.multiple_of(c_global_ptr, (1, BLOCK_SIZE_N)), (1, BLOCK_SIZE_N))
                tl.store(c_global_hinted, c, mask=sub_mask)
            else:
                # Record duration event around remote store (compiles away if tracing=False)
                handle = ctx.tracing.record_event_start(
                    event_id=TraceEvent().put,
                    target_rank=remote_rank,
                    address=c_global_ptr,
                    pid_m=pid_m,
                    pid_n=pid_n,
                )

                # Scatter accumulator registers directly to remote c_global.
                # hint=(1, BLOCK_SIZE_N) enables 128-bit vectorised global_store_dwordx4.
                ctx.store(c_global_ptr, c, to_rank=remote_rank, mask=sub_mask, hint=(1, BLOCK_SIZE_N))

                # End duration event
                ctx.tracing.record_event_end(handle)
