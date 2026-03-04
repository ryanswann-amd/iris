# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused GEMM + All-Scatter operation using scatter pattern.

Each rank has a column-sharded input B_local (K x N_local) and full input A (M x K).
Each rank computes C_local = A @ B_local, then scatters C_local tiles to all ranks
so that each rank ends up with the full C (M x N) where N = world_size * N_local.

This is useful for tensor-parallel workloads where weights are column-sharded and
outputs need to be gathered along the column dimension.
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris
import iris.x

from tritonblas.kernels.stages import GemmContext, ScheduleContext, make_tensor_view

from .config import FusedConfig
from .workspace import FusedWorkspace


@triton.jit()
def _fused_matmul_all_scatter_kernel(
    A,  # (M, K) - replicated across ranks
    B_local,  # (K, N_local) - each rank's column shard
    C_gathered,  # (M, N) - gathered output (N = N_local * world_size)
    bias_ptr,
    M,
    N,
    K,
    N_local,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm_gathered,
    stride_cn_gathered,
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
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Fused GEMM + all-scatter kernel using scatter pattern.

    Computes local GEMM tile and immediately scatters to all ranks along the N dimension.
    No intermediate buffer needed - direct from registers to remote memory.
    """
    # ═══════════════════════════════════════════════════════════════════════
    # Create tritonblas views, context, and scheduler for GEMM
    # ═══════════════════════════════════════════════════════════════════════
    tensorA = make_tensor_view(A, M, K, stride_am, stride_ak)
    tensorB = make_tensor_view(B_local, K, N_local, stride_bk, stride_bn)
    gemm_ctx = GemmContext(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        num_sms=NUM_SMS,
        num_xcds=NUM_XCDS,
        group_size_m=GROUP_SIZE_M,
        even_k=EVEN_K,
        allow_tf32=ALLOW_TF32,
    )
    sched = ScheduleContext(M, N_local, K, gemm_ctx)

    # Persistent loop over local tiles using scheduler
    start, total, stride = sched.persistent_tile_range()
    for tile_id in range(start, total, stride):
        # Get tile coordinates with swizzling from scheduler
        out_tile = sched.get_tile_from_idx(tile_id)

        # GEMM using tritonblas stages
        acc = gemm_ctx.reduce_axis(tensorA, tensorB, out_tile)

        # Add bias if provided
        if BIAS:
            rm, _ = out_tile.indices()
            bias_vector = tl.load(bias_ptr + rm * stride_bias, mask=rm < M, other=0.0)
            acc = acc + bias_vector[:, None]

        # Convert to output dtype
        c = acc.to(C_gathered.type.element_ty)

        # Create DeviceContext and destination TensorView for all-scatter
        ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
        dst_view = iris.x.make_tensor_view(C_gathered, M, N, stride_cm_gathered, stride_cn_gathered)
        tile_obj = iris.x.Tile(out_tile.pid_m, out_tile.pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, c)

        # Broadcast this rank's tile to all ranks using iris.x.all_gather with dim=1.
        # dim=1 places the tile at the current rank's column offset in the global output,
        # so every rank receives each rank's column-shard (all-scatter along N dimension).
        iris.x.all_gather(tile_obj, dst_view, dim=1, ctx=ctx)


def matmul_all_scatter_preamble(
    shmem,
    A: torch.Tensor,
    B_local: torch.Tensor,
    config: Optional[FusedConfig] = None,
) -> FusedWorkspace:
    """Allocate workspace for matmul_all_scatter (none needed for scatter pattern)."""
    if config is None:
        config = FusedConfig()

    M, K = A.shape
    K2, N_local = B_local.shape
    world_size = shmem.get_num_ranks()

    assert K == K2, f"Inner dimensions must match: A has K={K}, B_local has K={K2}"

    N = N_local * world_size

    return FusedWorkspace(
        operation="matmul_all_scatter",
        shape=(M, N, K),
        dtype=A.dtype,
        world_size=world_size,
        prepared=True,
    )


def matmul_all_scatter(
    shmem,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B_local: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
) -> FusedWorkspace:
    """
    Fused matrix multiplication and all-scatter using scatter pattern.

    Computes: output = all_scatter(A @ B_local) along N dimension

    Each rank has B_local of shape (K, N_local) where N_local = N / world_size.
    The operation computes C_local = A @ B_local on each rank and immediately
    broadcasts each rank's column-shard tiles to all ranks via iris.x.all_gather
    (dim=1), so that every rank ends up with the full C (M, N).

    This is a single-kernel implementation - no intermediate buffer needed.
    Internally this uses iris.x.all_gather(dim=1) to broadcast each rank's
    computed column tiles to all other ranks at the correct N offset.

    Args:
        shmem: Iris shmem context
        output_tensor: Output tensor C of shape (M, N) where N = N_local * world_size
        A: Input matrix A of shape (M, K) - replicated across ranks
        B_local: Column-sharded input matrix B of shape (K, N_local)
        bias: Optional bias vector (M,)
        async_op: If False, performs barrier at end
        config: Optional FusedConfig for tuning
        workspace: Optional pre-allocated workspace

    Returns:
        FusedWorkspace object

    Example:
        >>> N_local = N // world_size
        >>> B_local = shmem.randn((K, N_local), dtype=torch.float16)
        >>> output = shmem.zeros((M, N), dtype=torch.float16)
        >>> shmem.ops.matmul_all_scatter(output, A, B_local)
    """
    if config is None:
        config = FusedConfig()

    M, K = A.shape
    K2, N_local = B_local.shape
    world_size = shmem.get_num_ranks()
    rank = shmem.get_rank()

    assert K == K2, f"Inner dimensions must match: A has K={K}, B_local has K={K2}"

    N = N_local * world_size
    assert output_tensor.shape == (M, N), f"Output must be ({M}, {N}), got {output_tensor.shape}"

    # Validate problem size against block sizes
    assert M >= config.block_size_m, (
        f"M ({M}) must be >= block_size_m ({config.block_size_m}). Use smaller block sizes for small problems."
    )
    assert K >= config.block_size_k, (
        f"K ({K}) must be >= block_size_k ({config.block_size_k}). Use smaller block sizes for small problems."
    )
    assert N_local >= config.block_size_n, (
        f"N_local ({N_local}) must be >= block_size_n ({config.block_size_n}). "
        f"Use smaller block sizes for small problems."
    )

    # Allocate workspace if not provided
    if workspace is None:
        workspace = matmul_all_scatter_preamble(shmem, A, B_local, config)

    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B_local.stride()
    stride_cm_gathered, stride_cn_gathered = output_tensor.stride()

    if bias is not None:
        assert bias.shape[0] == M
        bias_ptr = bias
        stride_bias = bias.stride()[0] if bias.dim() > 0 else 1
        use_bias = True
    else:
        bias_ptr = output_tensor
        stride_bias = 1
        use_bias = False

    device = A.device
    num_sms = config.num_sms
    if num_sms is None:
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count

    even_k = K % config.block_size_k == 0

    # Launch single fused kernel
    grid = (num_sms,)
    _fused_matmul_all_scatter_kernel[grid](
        A,
        B_local,
        output_tensor,
        bias_ptr,
        M,
        N,
        K,
        N_local,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm_gathered,
        stride_cn_gathered,
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
        use_bias,
        even_k,
        config.allow_tf32,
    )

    if not async_op:
        shmem.barrier()

    return workspace
