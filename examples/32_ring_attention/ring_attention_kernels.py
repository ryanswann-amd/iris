################################################################################
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
#
# Ring Attention implementation based on:
#   "Ring Attention with Blockwise Transformers for Near-Infinite Context"
#   Liu et al., 2023 (https://arxiv.org/pdf/2310.01889)
#
################################################################################

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _ring_attn_fwd_kernel(
    Q,
    K,
    V,
    O,
    M,
    L,
    # strides for Q, K, V, O: [seq, num_heads, head_dim]
    stride_qs,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_os,
    stride_oh,
    stride_od,
    # strides for M, L: [num_heads, seq]
    stride_mh,
    stride_ms,
    stride_lh,
    stride_ls,
    # sizes
    seq_q,
    seq_kv,
    # global offsets for causal masking
    q_rank_start,
    kv_rank_start,
    scale,
    # compile-time constants
    CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Flash attention kernel for one ring step.

    Each program instance handles one attention head and one block of Q tokens.
    Iterates over all KV blocks and accumulates using online softmax.

    Accumulates into O (unnormalized), M (running log-sum-exp), L (running sum).
    The final output is O / L, computed after all ring steps complete.
    """
    h = tl.program_id(0)
    q_blk = tl.program_id(1)

    q_off = q_blk * BLOCK_Q
    q_idx = q_off + tl.arange(0, BLOCK_Q)
    q_mask = q_idx < seq_q

    # Load Q block: [BLOCK_Q, HEAD_DIM]
    q_ptrs = Q + h * stride_qh + q_idx[:, None] * stride_qs + tl.arange(0, HEAD_DIM)[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    # Load running statistics for this head and Q block
    m_ptrs = M + h * stride_mh + q_idx * stride_ms
    l_ptrs = L + h * stride_lh + q_idx * stride_ls
    o_ptrs = O + h * stride_oh + q_idx[:, None] * stride_os + tl.arange(0, HEAD_DIM)[None, :] * stride_od

    m = tl.load(m_ptrs, mask=q_mask, other=-float("inf"))
    l = tl.load(l_ptrs, mask=q_mask, other=0.0)
    o = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    # Global Q positions for causal masking.
    # Triton loads q_rank_start (a Python int) and q_idx (int32 arange) as int32.
    # The maximum value is world_size * seq_q which fits comfortably in int32.
    q_global = q_rank_start + q_idx

    # Iterate over all KV blocks
    d_idx = tl.arange(0, HEAD_DIM)
    for kv_off in range(0, seq_kv, BLOCK_KV):
        kv_idx = kv_off + tl.arange(0, BLOCK_KV)
        kv_mask = kv_idx < seq_kv

        # Load K in transposed layout [HEAD_DIM, BLOCK_KV] for efficient dot product
        k_ptrs = K + h * stride_kh + d_idx[:, None] * stride_kd + kv_idx[None, :] * stride_ks
        v_ptrs = V + h * stride_vh + kv_idx[:, None] * stride_vs + d_idx[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=kv_mask[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

        # Attention scores: [BLOCK_Q, BLOCK_KV] = Q [BLOCK_Q, HEAD_DIM] @ K^T [HEAD_DIM, BLOCK_KV]
        qk = tl.dot(q, k) * scale

        # Apply padding mask (validity) and optional causal mask
        if CAUSAL:
            kv_global = kv_rank_start + kv_idx
            # Causal: token at kv_pos can only be attended to if kv_pos <= q_pos
            causal_mask = kv_global[None, :] <= q_global[:, None]
            qk = tl.where(causal_mask & kv_mask[None, :], qk, -float("inf"))
        else:
            qk = tl.where(kv_mask[None, :], qk, -float("inf"))

        # Online softmax accumulation
        # m_new = max(m, row_max(qk))
        m_new = tl.maximum(m, tl.max(qk, axis=1))

        # Scale factor for previous running values
        alpha = libdevice.fast_expf(m - m_new)

        # Unnormalized attention probabilities
        p = libdevice.fast_expf(qk - m_new[:, None])

        # Update running sum
        l = alpha * l + tl.sum(p, axis=1)

        # Update running output (unnormalized weighted value sum)
        o = alpha[:, None] * o + tl.dot(p, v)

        # Update running max
        m = m_new

    # Write back updated statistics and output
    tl.store(m_ptrs, m, mask=q_mask)
    tl.store(l_ptrs, l, mask=q_mask)
    tl.store(o_ptrs, o, mask=q_mask[:, None])


def ring_attn_fwd(q, k, v, shmem, causal=True, scale=None):
    """
    Ring Attention forward pass.

    Each device holds a contiguous chunk of the sequence (Q, K, V). K and V
    are rotated around the ring of devices using torch.distributed send/recv,
    while Q remains local. At each step the local flash-attention kernel
    accumulates partial results into O, M, L using online softmax.

    After all ``world_size`` steps, O is normalised by L to produce the output.

    Args:
        q (torch.Tensor): Query tensor, shape ``[seq_q, num_heads, head_dim]``.
            Lives on the local device's CUDA memory.
        k (torch.Tensor): Key tensor, same shape as ``q``.
        v (torch.Tensor): Value tensor, same shape as ``q``.
        shmem: Iris shmem context (provides ``get_rank()`` / ``get_num_ranks()``
            and ``barrier()``).
        causal (bool): If ``True``, apply a causal (lower-triangular) mask so
            that position ``i`` only attends to positions ``j <= i``.
        scale (float | None): Softmax scale factor. Defaults to
            ``head_dim ** -0.5``.

    Returns:
        torch.Tensor: Attention output, shape ``[seq_q, num_heads, head_dim]``,
            same dtype as ``q``.
    """
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    seq_q, num_heads, head_dim = q.shape
    seq_kv = k.shape[0]

    assert (head_dim & (head_dim - 1)) == 0, f"head_dim must be a power of 2, got {head_dim}"
    assert seq_q % 64 == 0, f"seq_q ({seq_q}) must be divisible by BLOCK_Q (64)"
    assert seq_kv % 64 == 0, f"seq_kv ({seq_kv}) must be divisible by BLOCK_KV (64)"

    if scale is None:
        scale = head_dim**-0.5

    input_dtype = q.dtype

    # Running accumulators in float32 for numerical stability
    # O is the *unnormalized* weighted value sum
    O = torch.zeros(seq_q, num_heads, head_dim, dtype=torch.float32, device=q.device)
    # M: running row-max (log domain), L: running normalisation denominator
    M = torch.full((num_heads, seq_q), fill_value=-float("inf"), dtype=torch.float32, device=q.device)
    L = torch.zeros(num_heads, seq_q, dtype=torch.float32, device=q.device)

    # Choose block sizes; keep them as powers of 2
    BLOCK_Q = 64
    BLOCK_KV = 64
    HEAD_DIM = head_dim  # already validated as power of 2

    # We work with a rotating KV pair; start from the local K, V
    k_cur = k.contiguous()
    v_cur = v.contiguous()

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size

    for step in range(world_size):
        # The KV chunk we currently hold comes from rank kv_rank
        kv_rank = (rank - step) % world_size

        # Determine masking strategy for this step
        if causal:
            if kv_rank > rank:
                # All KV positions are strictly AFTER our Q positions → skip
                if step < world_size - 1:
                    k_cur, v_cur = _rotate_kv(k_cur, v_cur, next_rank, prev_rank)
                continue
            elif kv_rank == rank:
                # Same block → apply diagonal causal mask
                apply_causal = True
            else:
                # KV positions are all BEFORE our Q positions → full attention
                apply_causal = False
        else:
            apply_causal = False

        q_rank_start = rank * seq_q
        kv_rank_start = kv_rank * seq_kv

        grid = (num_heads, triton.cdiv(seq_q, BLOCK_Q))
        _ring_attn_fwd_kernel[grid](
            q,
            k_cur,
            v_cur,
            O,
            M,
            L,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cur.stride(0),
            k_cur.stride(1),
            k_cur.stride(2),
            v_cur.stride(0),
            v_cur.stride(1),
            v_cur.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            M.stride(0),
            M.stride(1),
            L.stride(0),
            L.stride(1),
            seq_q,
            seq_kv,
            q_rank_start,
            kv_rank_start,
            scale,
            CAUSAL=apply_causal,
            BLOCK_Q=BLOCK_Q,
            BLOCK_KV=BLOCK_KV,
            HEAD_DIM=HEAD_DIM,
            num_warps=4,
            num_stages=2,
        )

        # Rotate K, V to the next step (not needed after the last step)
        if step < world_size - 1:
            k_cur, v_cur = _rotate_kv(k_cur, v_cur, next_rank, prev_rank)

    # Normalize: output = O / L, where L is the softmax denominator
    # L: [num_heads, seq_q] → [seq_q, num_heads, 1] for broadcasting
    L_expanded = L.permute(1, 0).unsqueeze(-1)  # [seq_q, num_heads, 1]
    output = O / L_expanded

    return output.to(input_dtype)


def _rotate_kv(k, v, next_rank, prev_rank):
    """
    Perform one step of ring KV rotation using point-to-point communication.

    Sends the current ``k`` and ``v`` tensors to ``next_rank`` and receives
    new ``k`` and ``v`` from ``prev_rank``.

    The send and receive are posted concurrently to avoid deadlocks.
    """
    k_recv = torch.empty_like(k)
    v_recv = torch.empty_like(v)

    reqs = []
    reqs.append(dist.isend(k.contiguous(), dst=next_rank))
    reqs.append(dist.irecv(k_recv, src=prev_rank))
    reqs.append(dist.isend(v.contiguous(), dst=next_rank))
    reqs.append(dist.irecv(v_recv, src=prev_rank))

    for r in reqs:
        r.wait()

    return k_recv, v_recv
