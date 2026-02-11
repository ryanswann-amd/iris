# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused All-Gather + GEMM using a local HBM staging buffer with per-tile flags.

Each rank has a column-sharded input A_sharded (M x K_local).
This operation computes C = all_gather(A_sharded) @ B by:
  1. All CUs cooperate to gather A into a local HBM buffer, setting a ready
     flag for each (m_tile, k_block) as it lands.
  2. Each CU then runs GEMM from the local buffer. Before consuming a tile,
     it checks the ready flag; if not yet set, it spins until the gathering
     CU writes it.

No global barriers are needed. The per-tile flags provide fine-grained
producer-consumer synchronization: a CU that finishes gathering early can
start GEMM immediately, consuming any tile whose flag is already set.
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris
import iris.x

from .config import FusedConfig
from .workspace import FusedWorkspace


# ==========================================================================
# Kernel
# ==========================================================================


@triton.jit
def _hbm_buffer_all_gather_matmul_kernel(
    A_sharded,
    B,
    C,
    bias_ptr,
    staged_a,  # Local HBM buffer: (M, K) fp16
    flags_ptr,  # int32[NUM_M_TILES * NUM_K_BLOCKS] per-tile ready flags
    M,
    N,
    K,
    K_local,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,  # K // BLOCK_SIZE_K (global)
    NUM_K_BLOCKS_LOCAL: tl.constexpr,  # K_local // BLOCK_SIZE_K
    BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    HBM-buffered all-gather + GEMM with per-tile ready flags.

    Each CU executes two phases back-to-back (no global barrier):

    Phase 1 (gather): The CU is assigned a slice of the (m_tile, src_rank,
    k_block_local) gather work. For each assigned tile it pulls from remote
    via iris.x.gather, writes to staged_a, and atomically sets the ready
    flag. Local rank tiles are copied via a fast local load.

    Phase 2 (GEMM): The CU iterates over its assigned output tiles
    (pid_m, pid_n). For each K-block in the accumulation loop it checks the
    ready flag; if not yet set, it spins until the producing CU posts it.
    A tiles are loaded from staged_a (local HBM) and B tiles from B.
    """
    pid = tl.program_id(0)

    # XCD-aware PID remapping
    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)

    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32

    # DeviceContext and TensorView for gather
    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
    src_view = iris.x.make_tensor_view(A_sharded, M, K_local, stride_am, stride_ak)

    # ==================================================================
    # Phase 1: Cooperative gather into staged_a, set per-tile flags
    # ==================================================================
    # Total gather work = NUM_M_TILES * world_size * NUM_K_BLOCKS_LOCAL
    # Each tile is BLOCK_SIZE_M x BLOCK_SIZE_K elements.
    total_gather_tiles = NUM_M_TILES * world_size * NUM_K_BLOCKS_LOCAL

    for gather_idx in range(pid, total_gather_tiles, NUM_SMS):
        # Decompose flat index -> (m_tile, src_rank_idx, k_block_local)
        m_tile = gather_idx // (world_size * NUM_K_BLOCKS_LOCAL)
        remainder = gather_idx % (world_size * NUM_K_BLOCKS_LOCAL)
        src_rank_idx = remainder // NUM_K_BLOCKS_LOCAL
        k_block_local = remainder % NUM_K_BLOCKS_LOCAL

        # Global k-block index in the full K dimension
        k_block_global = src_rank_idx * NUM_K_BLOCKS_LOCAL + k_block_local

        # Gather the tile from the source rank, store to buffer, set flag.
        # source_rank must be constexpr for iris.x.gather, so we iterate
        # over all ranks at compile time and select at runtime.
        # The store and flag-set are inside the branch so that a_tile is
        # always defined when used.
        zero = tl.program_id(0) * 0
        pid_m_t = zero + m_tile
        tile_k_t = zero + k_block_local
        k_tile = iris.x.TileView(pid_m_t, tile_k_t, BLOCK_SIZE_M, BLOCK_SIZE_K)

        rm = m_tile * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rk = k_block_global * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        staged_ptrs = staged_a + rm[:, None] * K + rk[None, :]
        flag_idx = m_tile * NUM_K_BLOCKS + k_block_global

        for compile_rank in range(world_size):
            if src_rank_idx == compile_rank:
                a_tile = iris.x.gather(k_tile, src_view, compile_rank, ctx)
                tl.store(staged_ptrs, a_tile)
                tl.atomic_xchg(flags_ptr + flag_idx, 1, sem="release", scope="gpu")

    # ==================================================================
    # Phase 2: GEMM from staged_a (local) x B, checking flags
    # ==================================================================
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_gemm_tiles = NUM_M_TILES * num_tiles_n

    for gemm_tile_id in range(pid, total_gemm_tiles, NUM_SMS):
        # Tile scheduling with swizzle (GROUP_SIZE_M grouping)
        num_pid_in_group = GROUP_SIZE_M * num_tiles_n
        group_id = gemm_tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_sz = min(NUM_M_TILES - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((gemm_tile_id % num_pid_in_group) % group_sz)
        pid_n = (gemm_tile_id % num_pid_in_group) // group_sz

        # Row / column indices
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        # K-reduction loop
        for k_block in range(NUM_K_BLOCKS):
            # Wait for the (pid_m, k_block) tile to be ready.
            # acquire semantics ensure subsequent loads see the stored data.
            flag_idx = pid_m * NUM_K_BLOCKS + k_block
            while tl.atomic_add(flags_ptr + flag_idx, 0, sem="acquire", scope="gpu") == 0:
                pass

            # Load A from staged_a (purely local HBM)
            rk = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_SIZE_K), BLOCK_SIZE_K)
            a_ptrs = staged_a + rm[:, None] * K + rk[None, :]
            a = tl.load(a_ptrs)

            # Load B
            B_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            b = tl.load(B_ptrs)

            # Accumulate
            if ALLOW_TF32:
                acc = tl.dot(a, b, acc, allow_tf32=True)
            else:
                acc += tl.dot(a, b, allow_tf32=False)

        # Add bias if provided
        if BIAS:
            bias_val = tl.load(bias_ptr + rm * stride_bias, mask=rm < M, other=0.0)
            acc = acc + bias_val[:, None]

        # Convert to output dtype and store
        c = acc.to(C.type.element_ty)
        C_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(C_ptrs, c, mask=mask)


# ==========================================================================
# Python API
# ==========================================================================


def all_gather_matmul_hbm_buffer_preamble(
    shmem,
    A_sharded: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
) -> FusedWorkspace:
    """
    Allocate workspace for the HBM-buffered all_gather_matmul.

    Allocates:
      - staged_a: (M, K) local HBM buffer for the gathered A matrix.
      - flags: int32[num_m_tiles * num_k_blocks] per-tile ready flags.
    """
    if config is None:
        config = FusedConfig()

    M, K_local = A_sharded.shape
    K, N = B.shape
    world_size = shmem.get_num_ranks()

    expected_K = world_size * K_local
    assert K == expected_K, f"K ({K}) must equal world_size ({world_size}) * K_local ({K_local})"
    assert K_local % config.block_size_k == 0, (
        f"K_local ({K_local}) must be divisible by block_size_k ({config.block_size_k})"
    )
    assert K % config.block_size_k == 0, f"K ({K}) must be divisible by block_size_k ({config.block_size_k})"
    assert M % config.block_size_m == 0, f"M ({M}) must be divisible by block_size_m ({config.block_size_m})"

    num_m_tiles = M // config.block_size_m
    num_k_blocks = K // config.block_size_k

    ws = FusedWorkspace(
        operation="all_gather_matmul_hbm_buffer",
        shape=(M, N, K),
        dtype=A_sharded.dtype,
        world_size=world_size,
        variant="hbm_buffer",
        prepared=True,
    )

    # (M, K) staging buffer in local HBM
    ws.aux_buffer = shmem.zeros((M, K), dtype=A_sharded.dtype)
    # Per-tile ready flags
    ws.locks = shmem.zeros((num_m_tiles * num_k_blocks,), dtype=torch.int32)

    buffer_mb = M * K * A_sharded.element_size() / (1024**2)
    shmem.info(
        f"HBM buffer workspace: staged_a=({M},{K}) [{buffer_mb:.1f} MB], "
        f"flags=[{num_m_tiles}x{num_k_blocks}={num_m_tiles * num_k_blocks}]"
    )

    shmem.barrier()
    return ws


def all_gather_matmul_hbm_buffer(
    shmem,
    output_tensor: torch.Tensor,
    A_sharded: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
) -> FusedWorkspace:
    """
    All-gather + matmul using a local HBM staging buffer with per-tile flags.

    Computes C = all_gather(A_sharded) @ B + bias.

    Each CU first gathers its assigned slice of A tiles into the local buffer
    (setting per-tile ready flags), then runs GEMM from the buffer, spinning
    on flags for any tile not yet available.
    """
    if config is None:
        config = FusedConfig()

    M, K_local = A_sharded.shape
    K, N = B.shape
    world_size = shmem.get_num_ranks()
    rank = shmem.get_rank()

    expected_K = world_size * K_local
    assert K == expected_K, f"K ({K}) must equal world_size ({world_size}) * K_local ({K_local})"
    assert output_tensor.shape == (M, N), f"Output must be ({M}, {N}), got {output_tensor.shape}"
    assert M % config.block_size_m == 0, f"M ({M}) must be divisible by block_size_m ({config.block_size_m})"
    assert K % config.block_size_k == 0, f"K ({K}) must be divisible by block_size_k ({config.block_size_k})"
    assert K_local % config.block_size_k == 0, (
        f"K_local ({K_local}) must be divisible by block_size_k ({config.block_size_k})"
    )

    if workspace is None:
        workspace = all_gather_matmul_hbm_buffer_preamble(shmem, A_sharded, B, config)

    # Reset flags to 0 before each launch
    workspace.locks.zero_()

    stride_am, stride_ak = A_sharded.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = output_tensor.stride()

    if bias is not None:
        assert bias.shape[0] == M
        bias_ptr = bias
        stride_bias = bias.stride()[0] if bias.dim() > 0 else 1
        use_bias = True
    else:
        bias_ptr = output_tensor  # dummy, won't be read
        stride_bias = 1
        use_bias = False

    device = A_sharded.device
    num_sms = config.num_sms
    if num_sms is None:
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count

    num_m_tiles = M // config.block_size_m
    num_k_blocks = K // config.block_size_k
    num_k_blocks_local = K_local // config.block_size_k

    grid = (num_sms,)
    _hbm_buffer_all_gather_matmul_kernel[grid](
        A_sharded,
        B,
        output_tensor,
        bias_ptr,
        workspace.aux_buffer,  # staged_a
        workspace.locks,  # flags
        M,
        N,
        K,
        K_local,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bias,
        shmem.get_device_context(),
        rank,
        world_size,
        config.block_size_m,
        config.block_size_n,
        config.block_size_k,
        config.group_size_m,
        num_sms,
        config.num_xcds,
        num_m_tiles,
        num_k_blocks,
        num_k_blocks_local,
        use_bias,
        config.allow_tf32,
        matrix_instr_nonkdim=16,
    )

    if not async_op:
        shmem.barrier()

    return workspace
