import triton
import triton.language as tl
import iris
import torch
from typing import Optional
import warnings

from fused_helpers import _get_activation_from_str, pid_grid, remap_xcd, _get_config

@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _ff_a16w16_fused_ungated_iris(
    x_ptr,
    w1_ptr,
    w2_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_w1k,
    stride_w1n,
    stride_w2n,
    stride_w2k,
    stride_ym,
    stride_yk,
    # Iris args
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    heap_bases_ptr,
    # End iris args
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
    activation: tl.constexpr,
    use_activation: tl.constexpr,
):

    tl.assume(stride_xm > 0)
    tl.assume(stride_xk > 0)
    tl.assume(stride_w1k > 0)
    tl.assume(stride_w1n > 0)
    tl.assume(stride_w2k > 0)
    tl.assume(stride_w2n > 0)
    tl.assume(stride_ym > 0)
    tl.assume(stride_yk > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    remap_xcd(pid, GRID_MN)

    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Create pointers for first block of x and w1 input matrices
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_xm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    x_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)

    acc_dtype = tl.float32 if y_ptr.type.element_ty != tl.int8 else tl.int32

    offs_w1n = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_w1n[None, :] * stride_w1n)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            x = tl.load(x_ptrs, mask=offs_xm[:, None] < M)
            w1 = tl.load(w1_ptrs, mask=offs_w1n[None, :] < N, cache_modifier=cache_modifier)
        else:
            x = tl.load(x_ptrs, mask=(offs_xm[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
            w1 = tl.load(w1_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_w1n[None, :] < N), other=0.0, cache_modifier=cache_modifier)
        acc += tl.dot(x, w1, input_precision="ieee")
        # Advance the ptrs to the next K block.
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w1_ptrs += BLOCK_SIZE_K * stride_w1k

    if use_activation:
        acc = activation(acc)
    acc = acc.to(w2_ptr.type.element_ty)

    offs_w2n = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    w2_ptrs = w2_ptr + (offs_w2n[:, None] * stride_w2n + offs_k[None, :] * stride_w2k)
    offs_ym = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    y_ptrs = y_ptr + (offs_ym[:, None] * stride_ym + offs_k[None, :] * stride_yk)

    # Stagger k-loop start position based on N block index (to minimize contention)
    k_cyclic_offset = pid_n % tl.cdiv(K, BLOCK_SIZE_K)
    w2_ptrs += k_cyclic_offset * stride_w2k * BLOCK_SIZE_K
    y_ptrs += k_cyclic_offset * stride_yk * BLOCK_SIZE_K

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            w2 = tl.load(w2_ptrs, mask=offs_w2n[:, None] < N)
        else:
            w2 = tl.load(w2_ptrs, mask=(offs_w2n[:, None] < N) & ((offs_k[None, :] + k_cyclic_offset * BLOCK_SIZE_K) < K), other=0.0)
        
        partial_sum_y = tl.dot(acc, w2)
        
        # tl.device_print("w2:", w2)
        # tl.device_print("partial y:", partial_sum_y)
        y_mask = (offs_ym[:, None] < M) & ((offs_k[None, :] + BLOCK_SIZE_K * k) < K)

        # --- IRIS FUSED ALLREDUCE ---
        # Replaces the original tl.atomic_add with a loop over all GPUs.
        for dest_rank_id in range(0, world_size):
            iris.atomic_add(y_ptrs, partial_sum_y, my_rank, dest_rank_id,
                            heap_bases_ptr, mask=y_mask, sem="relaxed", scope="sys")
        # --- END IRIS MODIFICATION ---

        k_cyclic_offset += 1
        if k_cyclic_offset >= tl.cdiv(K, BLOCK_SIZE_K):
            k_cyclic_offset = 0
            w2_ptrs -= BLOCK_SIZE_K * stride_w2k * (tl.cdiv(K, BLOCK_SIZE_K) - 1)
            y_ptrs -= BLOCK_SIZE_K * stride_yk * (tl.cdiv(K, BLOCK_SIZE_K) - 1)
        else:
            w2_ptrs += BLOCK_SIZE_K * stride_w2k
            y_ptrs += BLOCK_SIZE_K * stride_yk

def ff_a16w16_fused_ungated_iris(
    x,
    w_up,
    w_down,
    iris_instance, # iris context
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    activation: Optional[str] = None,
):
    """
    Computes a full feed-forward operation with activation and a fused AllReduce
    for multi-GPU tensor parallelism.

    Key parameters:
    - X: Matrix X with shape (M, K), replicated on all ranks.
    - w_up: Up-projection W shard with shape (N/TP, K).
    - w_down: Down-projection W shard with shape (N/TP, K).
    - iris_instance: The Iris object for communication.
    - Y: Output Matrix Y buffer with shape (M, K), must be in the symmetric heap.
    - activation: Optional activation function to apply.
      One of ("gelu", "gelu_tanh", "silu", "silu_exp2", "relu", None)

    Returns:
    - Y: The output matrix with shape (M, K).
    """

    # Shape checks
    assert (
        x.shape[1] == w_up.shape[1] == w_down.shape[1]
    ), f"Incompatible matrix shapes: x:{x.shape}, w_up:{w_up.shape}, w_down:{w_down.shape}"
    assert (
        w_up.shape[0] == w_down.shape[0]
    ), f"Incompatible matrix shapes: w_up:{w_up.shape}, w_down:{w_down.shape}"
    
    # K (hidden_dim) is consistent, N (intermediate_dim) is sharded.
    N_shard, K = w_up.shape
    M = x.shape[0]
    
    if M > 64:
        warnings.warn(
            "The fused FF kernel is slower than the unfused equivalent for large batch sizes (>64)."
        )

    w_up = w_up.T

    if y is None:
        raise ValueError("Output tensor 'y' must be pre-allocated for the iris kernel.")

    if config is None:
        config = _get_config(M, N_shard, K)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N_shard, META["BLOCK_SIZE_N"]),
    )

    _ff_a16w16_fused_ungated_iris[grid](
        x,
        w_up,
        w_down,
        y,
        M,
        N_shard, # Use the sharded N dimension
        K,
        x.stride(0), x.stride(1),
        w_up.stride(0), w_up.stride(1),
        w_down.stride(0), w_down.stride(1),
        y.stride(0), y.stride(1),
        # Pass iris arguments to the kernel
        my_rank=iris_instance.get_rank(),
        world_size=iris_instance.get_num_ranks(),
        heap_bases_ptr=iris_instance.get_heap_bases(),
        activation=_get_activation_from_str(activation) if activation else "",
        use_activation=activation is not None,
        **config,
    )

    return y