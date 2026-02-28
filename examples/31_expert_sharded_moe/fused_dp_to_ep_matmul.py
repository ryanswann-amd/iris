# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
"""
Fused DP->EP dispatch + expert matmul.

This module fuses:
  convert_dp_to_ep(...)
  + grouped_matmul(y_ep_local, w_ep_local, b_ep_local, ...)

into a single Triton kernel that:
  1) resolves, for each expert-sorted row, which rank owns the source token
  2) gathers the activation row from the owning rank via iris.load (prologue)
  3) computes a tiled GEMM (BLOCK_M x BLOCK_N via tl.dot)
  4) stores the output locally in expert-sorted order (epilogue)

Grid: (n_n_tiles * n_local_experts,)  -- same tiling as grouped_matmul.
Each program loops over M-tiles for one (expert, N-tile) pair.  For each
M-tile, it uses combine_indx (col_sorted_indx) to map expert-sorted
positions back to global tokens, determines the owning rank, and pulls
the activation data from that rank's iris heap via per-rank masked 2D
iris.load.

Prerequisites:
  x_dp_local must be copied to the iris heap before launch so that remote
  ranks can access it.  All ranks allocate the same shape at the same heap
  offset (symmetric allocation), making pointer translation correct.
"""

import torch
import triton
import triton.language as tl
import iris

from ragged_metadata import RaggedTensorMetadata


@triton.jit
def _fused_dp_to_ep_matmul_kernel(
    y_ptr,
    y_stride_m,
    y_stride_n,
    x_shmem_ptr,
    x_stride_m,
    x_stride_k,
    w_ptr,
    w_stride_e,
    w_stride_k,
    w_stride_n,
    b_ptr,
    b_stride_e,
    b_stride_n,
    slice_offs_ptr,
    slice_sizes_ptr,
    combine_indx_ptr,
    n_local_experts,
    n_tokens_local,
    n_expts_act,
    K,
    N,
    heap_bases,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CUR_RANK: tl.constexpr,
    N_RANKS: tl.constexpr,
):
    pid = tl.program_id(0)
    n_n_tiles = tl.cdiv(N, BLOCK_N)

    local_expert_id = pid // n_n_tiles
    pid_n = pid % n_n_tiles

    if local_expert_id >= n_local_experts:
        return

    local_expert_id_64 = local_expert_id.to(tl.int64)
    slice_off = tl.load(slice_offs_ptr + local_expert_id_64).to(tl.int64)
    slice_size = tl.load(slice_sizes_ptr + local_expert_id_64)
    if slice_size == 0:
        return

    n_m_tiles = tl.cdiv(slice_size, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    mask_n = offs_n < N

    for pid_m in range(0, n_m_tiles):
        offs_m_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = slice_off + offs_m_local
        mask_m = offs_m_local < slice_size

        # -- Prologue: resolve source rank and local row for each row. --
        flat_idxs = tl.load(combine_indx_ptr + offs_m, mask=mask_m, other=-1)
        row_valid = mask_m & (flat_idxs >= 0)

        safe_flat = tl.where(row_valid, flat_idxs, tl.zeros_like(flat_idxs))
        token_ids = safe_flat // n_expts_act
        src_ranks = token_ids // n_tokens_local
        local_rows = token_ids % n_tokens_local

        # -- Body: tiled GEMM with per-rank remote gather. --
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for start_k in range(0, K, BLOCK_K):
            offs_k = start_k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            # Build X tile by gathering from each rank's x_dp_local on heap.
            x_ptrs = x_shmem_ptr + local_rows[:, None] * x_stride_m + offs_k[None, :] * x_stride_k
            x_tile = tl.zeros([BLOCK_M, BLOCK_K], dtype=x_shmem_ptr.dtype.element_ty)
            for r in tl.static_range(N_RANKS):
                rank_mask = row_valid & (src_ranks == r)
                load_mask = rank_mask[:, None] & mask_k[None, :]
                if r == CUR_RANK:
                    loaded = tl.load(x_ptrs, mask=load_mask, other=0.0)
                else:
                    loaded = iris.load(x_ptrs, CUR_RANK, r, heap_bases, mask=load_mask, hint=(1, 16))
                x_tile = tl.where(load_mask, loaded, x_tile)

            w_ptrs = (
                w_ptr + local_expert_id_64 * w_stride_e + offs_k[:, None] * w_stride_k + offs_n[None, :] * w_stride_n
            )
            w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

            acc += tl.dot(x_tile, w)

        if HAS_BIAS:
            b_ptrs = b_ptr + local_expert_id_64 * b_stride_e + offs_n * b_stride_n
            bias = tl.load(b_ptrs, mask=mask_n, other=0.0)
            acc += bias[None, :]

        # -- Epilogue: store output locally (expert-sorted order). --
        y_ptrs = y_ptr + offs_m[:, None] * y_stride_m + offs_n[None, :] * y_stride_n
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def fused_dp_to_ep_matmul(
    x_dp_local: torch.Tensor,
    w_ep_local: torch.Tensor,
    b_ep_local: torch.Tensor | None,
    combine_indx: torch.Tensor,
    n_expts_act: int,
    shmem,
    ragged_metadata: RaggedTensorMetadata,
) -> torch.Tensor:
    """Gather tokens from remote ranks and compute expert matmul in one kernel.

    Replaces the standalone convert_dp_to_ep + grouped_matmul sequence.
    Each GEMM tile's input rows are pulled directly from the owning rank's
    iris heap via per-rank masked 2D iris.load.

    Args:
        x_dp_local: (n_tokens_local, d_model) local token activations.
        w_ep_local: (n_local_experts, K, N) local expert weights.
        b_ep_local: (n_local_experts, N) local expert biases or None.
        combine_indx: (n_total_slots,) col_sorted_indx mapping expert-sorted
            positions back to global flat token*k indices.
        n_expts_act: k (experts per token).
        shmem: iris.Iris instance.
        ragged_metadata: local-expert-view ragged metadata (slice_offs, slice_sizes).

    Returns:
        (n_total_slots, N) output in expert-sorted order (same as grouped_matmul).
    """
    n_tokens_local, d_model = x_dp_local.shape
    n_local_experts = w_ep_local.shape[0]
    n_total_slots = combine_indx.shape[0]
    K = d_model
    N = d_model

    # Place x_dp_local on the iris heap so remote ranks can read it.
    x_shmem = shmem.zeros((n_tokens_local, d_model), dtype=x_dp_local.dtype)
    x_shmem.copy_(x_dp_local)
    shmem.barrier()

    y = torch.zeros((n_total_slots, N), dtype=x_dp_local.dtype, device=x_dp_local.device)

    BLOCK_M = 128
    BLOCK_N = min(triton.next_power_of_2(N), 128)
    BLOCK_K = min(triton.next_power_of_2(K), 64)

    n_n_tiles = triton.cdiv(N, BLOCK_N)
    grid = (n_n_tiles * n_local_experts,)

    _fused_dp_to_ep_matmul_kernel[grid](
        y,
        y.stride(0),
        y.stride(1),
        x_shmem,
        x_shmem.stride(0),
        x_shmem.stride(1),
        w_ep_local,
        w_ep_local.stride(0),
        w_ep_local.stride(1),
        w_ep_local.stride(2),
        b_ep_local if b_ep_local is not None else x_dp_local,
        b_ep_local.stride(0) if b_ep_local is not None else 0,
        b_ep_local.stride(1) if b_ep_local is not None else 0,
        ragged_metadata.slice_offs,
        ragged_metadata.slice_sizes,
        combine_indx,
        n_local_experts,
        n_tokens_local,
        n_expts_act,
        K,
        N,
        shmem.get_heap_bases(),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        HAS_BIAS=(b_ep_local is not None),
        CUR_RANK=shmem.get_rank(),
        N_RANKS=shmem.get_num_ranks(),
        num_warps=8,
        num_stages=2,
        matrix_instr_nonkdim=16,
        kpack=1,
    )

    return y
