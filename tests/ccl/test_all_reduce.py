# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for all-reduce collective operation.
"""

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris
from iris.ccl import Config
from iris.host.tracing.kernel_artifacts import iris_launch


# ---------------------------------------------------------------------------
# K-377 LEGACY SNAPSHOT — self-contained bit-identity reference
# ---------------------------------------------------------------------------
# Verbatim copy of the persistent_all_reduce_two_shot kernel as it existed
# *before* the K-377 NUM_CHANNELS knob was added (Apple-to-apple legacy formula:
# `start_rank_idx = pid % world_size`). This kernel is the trusted reference for
# ``test_all_reduce_two_shot_nch1_bit_identical_to_legacy`` — running NUM_CHANNELS=1
# through the production kernel and the *legacy* kernel below must produce
# byte-equal tensors (`torch.equal`). Drift between the two would surface here.
# DO NOT change this body.
# ---------------------------------------------------------------------------
@triton.jit
def _legacy_persistent_all_reduce_two_shot(
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
    pid = tl.program_id(0)

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

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N

        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        base_ptr = input_ptr + input_offset
        out_ptr = output_ptr + output_offset

        if is_full:
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            # Legacy formula: start_rank_idx = pid % world_size
            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(base_ptr, iris_rank, start_rank_global, heap_bases).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases).to(acc_dtype)

            reduced = acc.to(output_ptr.type.element_ty)
            tl.store(out_ptr, reduced, cache_modifier=".wt")

            for i in tl.static_range(0, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                if remote_rank_idx != group_rank:
                    iris.store(out_ptr, reduced, iris_rank, remote_rank, heap_bases, hint=(1, BLOCK_SIZE_N))
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            start_rank_idx = pid % world_size
            start_rank_global = rank_start + start_rank_idx * rank_stride
            acc = iris.load(base_ptr, iris_rank, start_rank_global, heap_bases, mask=mask).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                acc += iris.load(base_ptr, iris_rank, remote_rank, heap_bases, mask=mask).to(acc_dtype)

            reduced = acc.to(output_ptr.type.element_ty)
            tl.store(out_ptr, reduced, mask=mask, cache_modifier=".wt")

            for i in tl.static_range(0, world_size):
                remote_rank_idx = (start_rank_idx + i) % world_size
                remote_rank = rank_start + remote_rank_idx * rank_stride
                if remote_rank_idx != group_rank:
                    iris.store(
                        out_ptr,
                        reduced,
                        iris_rank,
                        remote_rank,
                        heap_bases,
                        mask=mask,
                        hint=(1, BLOCK_SIZE_N),
                    )


def _launch_legacy_two_shot(out, inp, ctx, config):
    """Launch the K-377 legacy snapshot kernel above. Mirrors the production
    launch() flow for VARIANT_TWO_SHOT but invokes the legacy kernel."""
    M, N = inp.shape[:2]
    sim, sin = inp.stride(0), inp.stride(1)
    som, son = out.stride(0), out.stride(1)
    rank_in_group = ctx.get_rank()
    rank_global = rank_in_group
    world_size = ctx.get_num_ranks()
    iris_launch(
        _legacy_persistent_all_reduce_two_shot,
        (config.comm_sms,),
        inp,
        out,
        M,
        N,
        sim,
        sin,
        som,
        son,
        ctx.get_heap_bases(),
        rank_in_group,
        rank_global,
        world_size,
        0,  # rank_start
        1,  # rank_stride
        config.block_size_m,
        config.block_size_n,
        config.swizzle_size,
        config.comm_sms,
        config.num_xcds,
        config.chunk_size,
        config.all_reduce_distribution,
        num_warps=8,
        num_stages=1,
        waves_per_eu=1,
        algorithm="all_reduce_legacy_reference",
        rank=rank_global,
        dtype=inp.dtype,
    )


@pytest.mark.parametrize(
    "variant",
    [
        "atomic",
        # "ring",
        "two_shot",
        "one_shot",
        # TODO enable these tests when support for cache-modifiers is in place.
        # "spinlock",
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "M, N, block_size_m, block_size_n",
    [
        (128, 64, 32, 64),  # Small
        (128, 128, 32, 32),  # BLOCK_N < N/world_size (partial-width, multi-block per rank)
        (256, 128, 32, 16),  # Minimum BLOCK_N=16 (16-bit vectorization path)
        (1024, 256, 32, 64),  # Medium
        (8192, 8192, 32, 64),  # Large
    ],
)
def test_all_reduce(variant, dtype, M, N, block_size_m, block_size_n):
    """Test all-reduce functionality by comparing against PyTorch's implementation."""
    # Ensure torch.distributed is initialized (should be done by test runner)
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()

    # PyTorch's all_reduce format: each rank has M x N data
    # All ranks compute the sum of all tensors
    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    # Fill with deterministic values for easier debugging
    pytorch_input_tensor.fill_(float(rank + 1))

    # Run PyTorch's all_reduce to get reference output
    pytorch_output_tensor = pytorch_input_tensor.clone()
    shmem.barrier()
    dist.all_reduce(pytorch_output_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Now set up Iris all_reduce format
    # Iris format: same as PyTorch - input and output are both (M, N)
    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)

    iris_output_tensor = shmem.zeros((M, N), dtype=dtype)

    # Run Iris all_reduce with specified variant
    shmem.barrier()
    config = Config(all_reduce_variant=variant, block_size_m=block_size_m, block_size_n=block_size_n)
    if variant == "two_shot":
        # Test both distribution modes for two_shot
        config.all_reduce_distribution = 0  # striding
    if variant == "ring":
        config.all_reduce_num_rings = min(2, config.comm_sms)

    # Explicitly call preamble to ensure proper initialization and synchronization
    # This helps with test isolation when tests run sequentially
    workspace = shmem.ccl.all_reduce_preamble(iris_output_tensor, iris_input_tensor, config=config)
    shmem.barrier()  # Ensure all ranks have completed preamble before starting kernel

    # Now call all_reduce with the prepared workspace
    shmem.ccl.all_reduce(iris_output_tensor, iris_input_tensor, config=config, workspace=workspace)
    torch.cuda.synchronize()

    # Compare results
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    max_diff = torch.abs(iris_output_tensor - pytorch_output_tensor).max().item()

    try:
        assert torch.allclose(iris_output_tensor, pytorch_output_tensor, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris output doesn't match PyTorch's all_reduce (variant={variant})"
        )
    finally:
        # Final barrier to ensure all ranks complete before test cleanup
        # This helps with test isolation when running multiple tests
        # Note: shmem.barrier() already does cuda.synchronize()
        shmem.barrier()
        # Explicitly delete the shmem instance to trigger cleanup
        del shmem
        # Force garbage collection to ensure IPC handles are cleaned up
        import gc

        gc.collect()


@pytest.mark.parametrize(
    "distribution",
    [
        0,  # striding
        1,  # block
    ],
)
def test_all_reduce_two_shot_distribution(distribution, dtype=torch.float32, M=1024, N=256):
    """Test two-shot all-reduce with different distribution modes."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()

    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_input_tensor.fill_(float(rank + 1))

    pytorch_output_tensor = pytorch_input_tensor.clone()
    shmem.barrier()
    dist.all_reduce(pytorch_output_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)

    iris_output_tensor = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()
    config = Config(all_reduce_variant="two_shot", all_reduce_distribution=distribution)

    # Explicitly call preamble to ensure proper initialization and synchronization
    workspace = shmem.ccl.all_reduce_preamble(iris_output_tensor, iris_input_tensor, config=config)
    shmem.barrier()  # Ensure all ranks have completed preamble before starting kernel

    # Now call all_reduce with the prepared workspace
    shmem.ccl.all_reduce(iris_output_tensor, iris_input_tensor, config=config, workspace=workspace)
    torch.cuda.synchronize()

    atol = 1e-5
    max_diff = torch.abs(iris_output_tensor - pytorch_output_tensor).max().item()

    try:
        assert torch.allclose(iris_output_tensor, pytorch_output_tensor, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris two-shot output doesn't match PyTorch (distribution={distribution})"
        )
    finally:
        # Final barrier to ensure all ranks complete before test cleanup
        # This helps with test isolation when running multiple tests
        # Note: shmem.barrier() already does cuda.synchronize()
        shmem.barrier()
        # Explicitly delete the shmem instance to trigger cleanup
        del shmem
        # Force garbage collection to ensure IPC handles are cleaned up
        import gc

        gc.collect()


@pytest.mark.parametrize("num_channels", [1, 2, 4, 8])
@pytest.mark.parametrize("distribution", [0, 1])
def test_all_reduce_two_shot_num_channels(num_channels, distribution, dtype=torch.float32, M=2048, N=512):
    """K-377: numerical correctness for NUM_CHANNELS ∈ {1,2,4,8} at non-trivial size.

    Uses *per-element randomized* fp32 inputs (distinct seed per rank), so any
    bug in the per-channel ring start formula or in tile-ownership would produce
    a non-equal tensor. Compares against ``dist.all_reduce(SUM)`` which is the
    canonical reference. fp32 sums are order-independent at this scale.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    if num_channels > world_size:
        pytest.skip(
            f"NUM_CHANNELS={num_channels} > world_size={world_size}: launch() clamps "
            f"to world_size, so this case collapses to the world_size variant."
        )

    # Per-element randomized fp32 — distinct values per (rank, element) so a
    # wrong ring traversal cannot accidentally produce the right answer (as a
    # constant-per-rank fill would). Per-rank seed makes inputs reproducible.
    torch.manual_seed(0xBEEF + rank)
    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")

    pytorch_output_tensor = pytorch_input_tensor.clone()
    shmem.barrier()
    dist.all_reduce(pytorch_output_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)
    iris_output_tensor = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()
    config = Config(
        all_reduce_variant="two_shot",
        all_reduce_distribution=distribution,
        all_reduce_num_channels=num_channels,
    )
    workspace = shmem.ccl.all_reduce_preamble(iris_output_tensor, iris_input_tensor, config=config)
    shmem.barrier()
    shmem.ccl.all_reduce(iris_output_tensor, iris_input_tensor, config=config, workspace=workspace)
    torch.cuda.synchronize()

    atol = 1e-4  # fp32 sum of W=8 randn elements: max round-off ≈ 8 * eps_fp32
    max_diff = torch.abs(iris_output_tensor - pytorch_output_tensor).max().item()
    try:
        assert torch.allclose(iris_output_tensor, pytorch_output_tensor, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: K-377 multi-channel two-shot mismatch "
            f"(num_channels={num_channels}, distribution={distribution})"
        )
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()


@pytest.mark.parametrize("distribution", [0, 1])
def test_all_reduce_two_shot_nch1_bit_identical_to_legacy(distribution, dtype=torch.float32, M=1024, N=256):
    """K-377: prove NUM_CHANNELS=1 is BIT-IDENTICAL to the *legacy* (pre-K-377) kernel.

    The K-377 channel partition adds a ``channel_id * (world_size // NUM_CHANNELS)``
    offset on top of the legacy ``start_rank_idx = pid % world_size`` formula. At
    NUM_CHANNELS=1 the offset collapses algebraically to zero. This test verifies
    that collapse at *runtime* by comparing the K-377 production kernel (with
    NUM_CHANNELS=1) against ``_legacy_persistent_all_reduce_two_shot`` — a frozen
    snapshot of the kernel as it existed before this PR, defined at the top of
    this file. ``torch.equal`` (byte-equal) is required, not ``allclose``.

    Inputs are non-trivial randomized fp32 (per-rank distinct seed), so partial
    sums are dependent on the exact ring traversal order; a real algorithmic drift
    would surface as a non-equal tensor. We additionally cross-check both kernels
    against ``dist.all_reduce(SUM)`` for an independent reference.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Non-trivial randomized inputs (per-rank seed). fp32 keeps reduction exact.
    torch.manual_seed(0xC0FFEE + rank)
    pytorch_input = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")

    # Independent reference via dist.all_reduce.
    pytorch_reference = pytorch_input.clone()
    shmem.barrier()
    dist.all_reduce(pytorch_reference, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    iris_input = shmem.zeros((M, N), dtype=dtype)
    iris_input.copy_(pytorch_input)
    shmem.barrier()

    # Run #1 — K-377 production kernel at NUM_CHANNELS=1.
    config_nch1 = Config(
        all_reduce_variant="two_shot",
        all_reduce_distribution=distribution,
        all_reduce_num_channels=1,
    )
    out_nch1 = shmem.zeros((M, N), dtype=dtype)
    ws = shmem.ccl.all_reduce_preamble(out_nch1, iris_input, config=config_nch1)
    shmem.barrier()
    shmem.ccl.all_reduce(out_nch1, iris_input, config=config_nch1, workspace=ws)
    torch.cuda.synchronize()
    shmem.barrier()

    # Run #2 — frozen legacy snapshot kernel (no NUM_CHANNELS knob at all).
    config_legacy = Config(
        all_reduce_variant="two_shot",
        all_reduce_distribution=distribution,
    )
    out_legacy = shmem.zeros((M, N), dtype=dtype)
    _launch_legacy_two_shot(out_legacy, iris_input, shmem, config_legacy)
    torch.cuda.synchronize()
    shmem.barrier()

    try:
        # Default Config must keep NCH=1 (legacy preservation contract).
        assert config_legacy.all_reduce_num_channels == 1, (
            f"Default Config.all_reduce_num_channels must be 1 to preserve legacy, "
            f"got {config_legacy.all_reduce_num_channels}"
        )
        # Primary reviewer-requested check: K-377 NCH=1 ↔ true legacy snapshot.
        assert torch.equal(out_nch1, out_legacy), (
            f"Rank {rank}: K-377 NCH=1 vs frozen LEGACY snapshot NOT bit-identical "
            f"(distribution={distribution}). max |Δ| = "
            f"{torch.abs(out_nch1 - out_legacy).max().item()}"
        )
        # Independent cross-check against dist.all_reduce reference. Iris and
        # RCCL use different reduction orders so the lowest fp32 bits may differ;
        # the ulp-bounded atol below is still a tight numerical guard for an
        # 8-element sum of randn fp32 values (max round-off ≈ W * eps_fp32).
        atol = 1e-4
        max_diff_nch1 = torch.abs(out_nch1 - pytorch_reference).max().item()
        max_diff_legacy = torch.abs(out_legacy - pytorch_reference).max().item()
        assert torch.allclose(out_nch1, pytorch_reference, atol=atol), (
            f"Rank {rank}: K-377 NCH=1 vs dist.all_reduce mismatch "
            f"(distribution={distribution}). max |Δ| = {max_diff_nch1}"
        )
        assert torch.allclose(out_legacy, pytorch_reference, atol=atol), (
            f"Rank {rank}: legacy snapshot vs dist.all_reduce mismatch "
            f"(distribution={distribution}). max |Δ| = {max_diff_legacy}"
        )
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()


def test_all_reduce_two_shot_num_channels_non_pow2_rejected():
    """K-377: Config rejects non-power-of-two num_channels values.

    The kernel's channel_offset = channel_id * (world_size // NUM_CHANNELS) only
    yields a uniform fan-out when NUM_CHANNELS divides world_size evenly. The
    canonical world_size on MI300X is 8, so we restrict the knob to powers of
    two. This is the user-facing error case the reviewer asked for.
    """
    # Each invalid value must raise from Config.__post_init__ (no GPU needed).
    for bad in (0, -1, -2, 3, 5, 6, 7, 9, 10, 12, 100):
        with pytest.raises(ValueError):
            Config(all_reduce_variant="two_shot", all_reduce_num_channels=bad)

    # Spot-check that the valid power-of-two grid is accepted.
    for good in (1, 2, 4, 8, 16, 32):
        cfg = Config(all_reduce_variant="two_shot", all_reduce_num_channels=good)
        assert cfg.all_reduce_num_channels == good


def test_all_reduce_two_shot_num_channels_clamps_to_world_size(dtype=torch.float32, M=1024, N=256):
    """K-377: launch() clamps NUM_CHANNELS to world_size at runtime.

    A user can ask for NUM_CHANNELS=16 on an 8-rank job. Without clamping, the
    kernel computes `world_size // NUM_CHANNELS == 0` and every channel
    silently degenerates back onto the single-channel ring. The launch wrapper
    must clamp before forwarding the constexpr, and the reduced result must
    still be correct.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Pick a NUM_CHANNELS strictly larger than world_size to force a clamp.
    # Must be a power of two so Config validation accepts it.
    nc_request = 1
    while nc_request <= world_size:
        nc_request *= 2  # smallest pow2 > world_size

    iris_input = shmem.zeros((M, N), dtype=dtype)
    iris_input.fill_(float(rank + 1))
    iris_output = shmem.zeros((M, N), dtype=dtype)

    expected = torch.full((M, N), float(world_size * (world_size + 1) // 2), dtype=dtype, device=f"cuda:{rank}")

    shmem.barrier()
    config = Config(
        all_reduce_variant="two_shot",
        all_reduce_distribution=1,
        all_reduce_num_channels=nc_request,
    )
    workspace = shmem.ccl.all_reduce_preamble(iris_output, iris_input, config=config)
    shmem.barrier()
    # If clamp is missing, this either crashes (div-by-zero in offset) or silently
    # produces wrong output (every channel on rank 0). Both fail the assertion below.
    shmem.ccl.all_reduce(iris_output, iris_input, config=config, workspace=workspace)
    torch.cuda.synchronize()

    try:
        assert torch.equal(iris_output, expected), (
            f"Rank {rank}: NUM_CHANNELS={nc_request} (>world_size={world_size}) "
            f"output mismatched expected SUM. "
            f"got {iris_output.flatten()[0].item()} expected {expected.flatten()[0].item()}"
        )
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()


def test_all_reduce_spinlock_lock_too_small():
    """Test that ValueError is raised when the spinlock lock array is too small for current tile count.

    Scenario: workspace is prepared with larger block sizes (fewer tiles), then all_reduce
    is called with smaller block sizes (more tiles). workspace.matches() skips the preamble,
    and the undersized lock array is detected.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)

    M, N = 512, 512

    iris_input = shmem.zeros((M, N), dtype=torch.float32)
    iris_output = shmem.zeros((M, N), dtype=torch.float32)

    shmem.barrier()

    # Step 1: run preamble with larger block sizes → allocates a smaller lock array
    config_large = Config(all_reduce_variant="spinlock", block_size_m=128, block_size_n=128)
    workspace = shmem.ccl.all_reduce_preamble(iris_output, iris_input, config=config_large)

    # Step 2: call all_reduce with smaller block sizes that need more tiles —
    # workspace.matches() returns True (same shape/dtype/variant), preamble is skipped,
    # and the undersized lock array is detected.
    config_small = Config(all_reduce_variant="spinlock", block_size_m=64, block_size_n=64)
    with pytest.raises(ValueError, match="Lock array too small"):
        shmem.ccl.all_reduce(iris_output, iris_input, config=config_small, workspace=workspace)

    shmem.barrier()
    del shmem
    import gc

    gc.collect()


def test_all_reduce_ring_flags_too_small():
    """Test that ValueError is raised when the ring flags array is too small for current tile count.

    Scenario: workspace is prepared with larger block sizes (fewer tiles), then all_reduce
    is called with smaller block sizes (more tiles). workspace.matches() skips the preamble,
    and the undersized flags array is detected.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    world_size = shmem.get_num_ranks()

    M, N = 512, 512

    # Choose block_size_n values divisible by world_size for both configs
    # Use 128 and 64 which are divisible by typical world sizes (1, 2, 4, 8)
    block_size_n_large = (128 // world_size) * world_size
    block_size_n_small = (64 // world_size) * world_size
    if block_size_n_large == 0 or block_size_n_small == 0 or block_size_n_large == block_size_n_small:
        del shmem
        pytest.skip(f"Cannot create two distinct block sizes divisible by world_size={world_size}")

    iris_input = shmem.zeros((M, N), dtype=torch.float32)
    iris_output = shmem.zeros((M, N), dtype=torch.float32)

    shmem.barrier()

    # Step 1: run preamble with larger block sizes → allocates a smaller flags array
    config_large = Config(
        all_reduce_variant="ring",
        block_size_m=128,
        block_size_n=block_size_n_large,
    )
    workspace = shmem.ccl.all_reduce_preamble(iris_output, iris_input, config=config_large)

    # Step 2: call all_reduce with smaller block sizes that need more tiles —
    # workspace.matches() returns True (same shape/dtype/variant), preamble is skipped,
    # and the undersized flags array is detected.
    config_small = Config(
        all_reduce_variant="ring",
        block_size_m=64,
        block_size_n=block_size_n_small,
    )
    with pytest.raises(ValueError, match="Flags array too small"):
        shmem.ccl.all_reduce(iris_output, iris_input, config=config_small, workspace=workspace)

    shmem.barrier()
    del shmem
    import gc

    gc.collect()
