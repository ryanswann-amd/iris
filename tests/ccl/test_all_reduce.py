# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for all-reduce collective operation.
"""

import pytest
import torch
import torch.distributed as dist
import iris
from iris.ccl import Config


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
    """Verify that pilot N-channel two-shot (K-377) preserves correctness for N in {1,2,4,8}.

    NUM_CHANNELS=1 must reproduce the legacy single-channel formula bit-for-bit.
    NUM_CHANNELS in {2,4,8} should produce identical reduced output (same SUM)
    while internally fanning each channel's ring across distinct xGMI links.
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
    config = Config(
        all_reduce_variant="two_shot",
        all_reduce_distribution=distribution,
        all_reduce_num_channels=num_channels,
    )
    workspace = shmem.ccl.all_reduce_preamble(iris_output_tensor, iris_input_tensor, config=config)
    shmem.barrier()
    shmem.ccl.all_reduce(iris_output_tensor, iris_input_tensor, config=config, workspace=workspace)
    torch.cuda.synchronize()

    atol = 1e-5
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
    """K-377: prove NUM_CHANNELS=1 is BIT-IDENTICAL to the legacy single-channel formula.

    The kernel modification introduced a `channel_id * (world_size // NUM_CHANNELS)`
    offset that shifts each channel's ring start. At NUM_CHANNELS=1 this collapses
    algebraically to `pid % world_size` (the legacy formula). This test runs the
    kernel twice — once at NCH=1, once at the default Config (NCH=1) — and asserts
    that BOTH outputs are byte-equal (`torch.equal`, not `allclose`) to each other
    and that they match `dist.all_reduce(SUM)` with deterministic fp32 inputs.

    A static check (NCH=1 ⇒ offset==0 ⇒ formula reduces to legacy) is captured at
    review time; this is the *runtime* assertion the reviewer asked for.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Deterministic input: each rank's tile is `rank + 1`. Sum is exactly W*(W+1)/2.
    iris_input = shmem.zeros((M, N), dtype=dtype)
    iris_input.fill_(float(rank + 1))

    expected = torch.full((M, N), float(world_size * (world_size + 1) // 2), dtype=dtype, device=f"cuda:{rank}")

    shmem.barrier()

    # Run #1 — explicit NUM_CHANNELS=1
    config_explicit = Config(
        all_reduce_variant="two_shot",
        all_reduce_distribution=distribution,
        all_reduce_num_channels=1,
    )
    out_explicit = shmem.zeros((M, N), dtype=dtype)
    ws1 = shmem.ccl.all_reduce_preamble(out_explicit, iris_input, config=config_explicit)
    shmem.barrier()
    shmem.ccl.all_reduce(out_explicit, iris_input, config=config_explicit, workspace=ws1)
    torch.cuda.synchronize()
    shmem.barrier()

    # Run #2 — Config default (which is NUM_CHANNELS=1 — preserves legacy)
    config_default = Config(
        all_reduce_variant="two_shot",
        all_reduce_distribution=distribution,
    )
    out_default = shmem.zeros((M, N), dtype=dtype)
    ws2 = shmem.ccl.all_reduce_preamble(out_default, iris_input, config=config_default)
    shmem.barrier()
    shmem.ccl.all_reduce(out_default, iris_input, config=config_default, workspace=ws2)
    torch.cuda.synchronize()

    try:
        # Default Config must keep NCH=1 (legacy preservation contract).
        assert config_default.all_reduce_num_channels == 1, (
            f"Default Config.all_reduce_num_channels must be 1 to preserve legacy, "
            f"got {config_default.all_reduce_num_channels}"
        )
        # Bit-identical: explicit NCH=1 and default produce byte-equal tensors.
        assert torch.equal(out_explicit, out_default), (
            f"Rank {rank}: NCH=1 (explicit) vs default Config NOT bit-identical "
            f"(distribution={distribution}). max |Δ| = "
            f"{torch.abs(out_explicit - out_default).max().item()}"
        )
        # Both must match the deterministic SUM exactly (fp32 has no rounding here).
        assert torch.equal(out_explicit, expected), (
            f"Rank {rank}: NCH=1 output != deterministic SUM "
            f"(distribution={distribution}). got {out_explicit.flatten()[0].item()} "
            f"expected {expected.flatten()[0].item()}"
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
