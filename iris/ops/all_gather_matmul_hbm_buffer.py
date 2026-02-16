# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused All-Gather + GEMM using a local HBM staging buffer with dedicated
fetcher and GEMM workgroups, launched data-parallel.

Supports configurable staged_a buffer layout (M-contiguous or K-contiguous)
and B layout to match optimal tritonblas conventions (TN, TT, NT, NN).
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris
import iris.x

from .config import FusedConfig
from .workspace import FusedWorkspace


@triton.jit
def _hbm_buffer_all_gather_matmul_kernel(
    A_sharded,
    B,
    C,
    bias_ptr,
    staged_a,
    flags_ptr,
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
    stride_sa_m,    # staged_a stride in M dim
    stride_sa_k,    # staged_a stride in K dim
    stride_bias,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_FETCH_SMS: tl.constexpr,
    NUM_M_TILES: tl.constexpr,
    NUM_TILES_N: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    NUM_K_BLOCKS_LOCAL: tl.constexpr,
    K_PER_FLAG: tl.constexpr,
    NUM_FLAG_GROUPS_K: tl.constexpr,
    TOTAL_GATHER_TILES: tl.constexpr,
    BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid = tl.program_id(0)
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32
    zero = tl.program_id(0) * 0

    if pid < NUM_FETCH_SMS:
        # ==============================================================
        # FETCHER
        # ==============================================================
        ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
        src_view = iris.x.make_tensor_view(A_sharded, M, K_local, stride_am, stride_ak)

        num_m_groups = (NUM_M_TILES + GROUP_SIZE_M - 1) // GROUP_SIZE_M
        tiles_per_m_group = NUM_FLAG_GROUPS_K * GROUP_SIZE_M
        total_flag_groups = NUM_FLAG_GROUPS_K * NUM_M_TILES

        for fg_idx in range(pid, total_flag_groups, NUM_FETCH_SMS):
            m_group = fg_idx // tiles_per_m_group
            within_group = fg_idx % tiles_per_m_group
            k_flag_group = within_group // GROUP_SIZE_M
            m_in_group = within_group % GROUP_SIZE_M
            m_tile = m_group * GROUP_SIZE_M + m_in_group
            m_tile = min(m_tile, NUM_M_TILES - 1)
            k_block_start = k_flag_group * K_PER_FLAG

            rm = m_tile * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

            for k_off in range(K_PER_FLAG):
                k_block_global = k_block_start + k_off

                src_rank_idx = k_block_global // NUM_K_BLOCKS_LOCAL
                k_block_local = k_block_global % NUM_K_BLOCKS_LOCAL

                pid_m_t = zero + m_tile
                tile_k_t = zero + k_block_local
                k_tile = iris.x.TileView(pid_m_t, tile_k_t, BLOCK_SIZE_M, BLOCK_SIZE_K)

                rk = k_block_global * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                # Use parameterized strides for staged_a
                staged_ptrs = staged_a + rm[:, None] * stride_sa_m + rk[None, :] * stride_sa_k

                for compile_rank in range(world_size):
                    if src_rank_idx == compile_rank:
                        a_tile = iris.x.gather(k_tile, src_view, compile_rank, ctx)
                        tl.store(staged_ptrs, a_tile,cache_modifier=".wt")   

            flag_idx = m_tile * NUM_FLAG_GROUPS_K + k_flag_group
            #tl.atomic_xchg(flags_ptr + flag_idx, 1, sem="release", scope="gpu")
            tl.store(flags_ptr + flag_idx, 1)

    else:
        # ==============================================================
        # GEMM
        # ==============================================================
        gemm_tile_id = pid - NUM_FETCH_SMS

        num_pid_in_group = GROUP_SIZE_M * NUM_TILES_N
        group_id = gemm_tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_sz = min(NUM_M_TILES - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((gemm_tile_id % num_pid_in_group) % group_sz)
        pid_n = (gemm_tile_id % num_pid_in_group) // group_sz

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

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

                # Use parameterized strides for staged_a
                a_ptrs = staged_a + rm[:, None] * stride_sa_m + rk[None, :] * stride_sa_k
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
        C_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(C_ptrs, c, mask=c_mask)


# ==========================================================================
# Python API
# ==========================================================================


def all_gather_matmul_hbm_buffer_preamble(
    shmem,
    A_sharded: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
    k_per_flag: int = 1,
    staged_a_layout: str = "k_contiguous",
) -> FusedWorkspace:
    """
    Allocate workspace.

    Args:
        staged_a_layout: "k_contiguous" (default, row-major (M,K)) or
                         "m_contiguous" (col-major, stored as (K,M) transposed).
    """
    if config is None:
        config = FusedConfig()

    M, K_local = A_sharded.shape
    K, N = B.shape
    world_size = shmem.get_num_ranks()

    assert world_size * K_local == K
    assert K_local % config.block_size_k == 0
    assert K % config.block_size_k == 0
    assert M % config.block_size_m == 0

    num_m_tiles = M // config.block_size_m
    num_k_blocks = K // config.block_size_k
    assert num_k_blocks % k_per_flag == 0
    num_flag_groups_k = num_k_blocks // k_per_flag

    ws = FusedWorkspace(
        operation="all_gather_matmul_hbm_buffer",
        shape=(M, N, K),
        dtype=A_sharded.dtype,
        world_size=world_size,
        variant=f"hbm_buffer_{staged_a_layout}",
        prepared=True,
    )

    if staged_a_layout == "m_contiguous":
        # Allocate (K, M) row-major, .T gives (M, K) with stride_m=1, stride_k=M
        storage = shmem.zeros((K, M), dtype=A_sharded.dtype)
        ws.aux_buffer = storage.T  # (M, K) view, M-contiguous
    else:
        # Default: (M, K) row-major, stride_m=K, stride_k=1
        ws.aux_buffer = shmem.zeros((M, K), dtype=A_sharded.dtype)

    ws.locks = shmem.zeros((num_m_tiles * num_flag_groups_k,), dtype=torch.int32)

    buffer_mb = M * K * A_sharded.element_size() / (1024 ** 2)
    sa_stride_m, sa_stride_k = ws.aux_buffer.stride()
    shmem.info(
        f"HBM buffer: staged_a=({M},{K}) [{buffer_mb:.1f} MB] "
        f"layout={staged_a_layout} strides=({sa_stride_m},{sa_stride_k}), "
        f"flags={num_m_tiles}x{num_flag_groups_k}, k_per_flag={k_per_flag}"
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
    num_fetch_sms: Optional[int] = None,
    k_per_flag: int = 1,
    fetch_block_m: Optional[int] = None,
    fetch_block_k: Optional[int] = None,
    staged_a_layout: str = "k_contiguous",
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> FusedWorkspace:
    """
    All-gather + matmul with dedicated fetcher/GEMM workgroups.

    Args:
        staged_a_layout: Buffer layout for gathered A.
            "k_contiguous" — (M,K) row-major, K is fast dim. Matches NN convention.
            "m_contiguous" — (M,K) with M as fast dim. Matches TN convention (best for tritonblas).
    """
    if config is None:
        config = FusedConfig()

    M, K_local = A_sharded.shape
    K, N = B.shape
    world_size = shmem.get_num_ranks()
    rank = shmem.get_rank()

    assert world_size * K_local == K
    assert output_tensor.shape == (M, N)
    assert M % config.block_size_m == 0
    assert K % config.block_size_k == 0
    assert K_local % config.block_size_k == 0

    if fetch_block_m is None:
        fetch_block_m = config.block_size_m
    if fetch_block_k is None:
        fetch_block_k = config.block_size_k

    num_k_blocks = K // config.block_size_k
    assert num_k_blocks % k_per_flag == 0

    if workspace is None:
        workspace = all_gather_matmul_hbm_buffer_preamble(
            shmem, A_sharded, B, config, k_per_flag, staged_a_layout
        )

    workspace.locks.zero_()

    stride_am, stride_ak = A_sharded.stride()
    stride_bk, stride_bn = B.stride()
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
    num_sms = config.num_sms
    if num_sms is None:
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count

    num_m_tiles = M // config.block_size_m
    num_tiles_n = (N + config.block_size_n - 1) // config.block_size_n
    total_gemm_tiles = num_m_tiles * num_tiles_n
    num_k_blocks_local = K_local // config.block_size_k
    num_flag_groups_k = num_k_blocks // k_per_flag
    total_gather_tiles = num_m_tiles * num_k_blocks

    if num_fetch_sms is None:
        num_fetch_sms = max(1, num_sms // 10)
    assert 0 < num_fetch_sms

    grid_size = num_fetch_sms + total_gemm_tiles

    launch_kwargs = {"matrix_instr_nonkdim": 16}
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps
    if num_stages is not None:
        launch_kwargs["num_stages"] = num_stages

    _hbm_buffer_all_gather_matmul_kernel[(grid_size,)](
        A_sharded, B, output_tensor, bias_ptr,
        workspace.aux_buffer, workspace.locks,
        M, N, K, K_local,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_sa_m, stride_sa_k,
        stride_bias,
        shmem.get_device_context(),
        rank, world_size,
        config.block_size_m,
        config.block_size_n,
        config.block_size_k,
        config.group_size_m,
        num_fetch_sms,
        num_m_tiles,
        num_tiles_n,
        num_k_blocks,
        num_k_blocks_local,
        k_per_flag,
        num_flag_groups_k,
        total_gather_tiles,
        use_bias,
        config.allow_tf32,
        **launch_kwargs,
    )

    if not async_op:
        shmem.barrier()

    return workspace
