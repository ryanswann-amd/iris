# Triton Reference MoE Implementation

This directory contains a standalone reference implementation of MoE using standard Triton kernels (no Iris dependency).

## Purpose

Use this as a baseline/reference to:
- Test the standard Triton MoE implementation on a different machine
- Compare performance between Triton and Iris implementations
- Debug and validate MoE functionality without Iris dependencies

## Files

- `moe_triton_reference.py` - Reference MoE implementation using Triton's native symmetric memory
- `test_triton_reference.py` - Test and benchmark script for the reference implementation
- `bench_mlp.py` - Comprehensive MLP/MoE benchmark with shape sweeps (**from triton repo**)
- `bench_utils.py` - Utilities for benchmarking (numerics, dtypes, etc.)
- `distributed.py` - Distributed MoE test utilities and routing functions

## Usage

### Running the test

```bash
cd examples/22_moe/reference
python test_triton_reference.py
```

This will:
1. Test the Triton reference MoE on 8 GPUs
2. Benchmark the performance and report timing

### Customizing GPU count

Edit the last line in `test_triton_reference.py`:

```python
if __name__ == "__main__":
    run_test(world_size=4)  # Change to desired number of GPUs
```

### Running comprehensive benchmarks (from triton repo)

The `bench_mlp.py` script provides shape sweeps and detailed profiling:

```bash
# Example: Benchmark MoE with different configurations
python bench_mlp.py --batch-per-expt 8 --dim1 2048 --dim2 8192 \
    --n-expts-tot 16 --n-expts-act 2 --EP 8
```

Key parameters:
- `--batch-per-expt`: Tokens per expert
- `--dim1`: Input/output dimension
- `--dim2`: Hidden dimension
- `--n-expts-tot`: Total number of experts
- `--n-expts-act`: Top-K experts activated
- `--EP`: Expert parallelism (number of GPUs sharding experts)
- `--x-dtype`: Input dtype (bf16, fp8)
- `--w-dtype`: Weight dtype (bf16, fp8, mx4)

## Differences from Iris Implementation

| Feature | Triton Reference | Iris V2 |
|---------|-----------------|---------|
| Communication | Triton symmetric memory (`convert_dp_to_ep`, `convert_ep_to_dp`) | Iris symmetric memory (`iris.store`, `shmem.barrier()`) |
| Memory allocation | `symm_mem_pool.make_empty()` | `shmem.zeros()` |
| Barriers | `hdl.barrier(channel=0)` | `shmem.barrier()` |

## Benchmark Scripts

Two types of benchmarks are available:

1. **Simple test/benchmark** (`test_triton_reference.py`)
   - Quick correctness test + basic timing
   - Easy to run and understand
   - Good for verifying the setup works

2. **Comprehensive benchmark** (`bench_mlp.py`)
   - Shape sweeps and detailed profiling
   - Supports various dtypes and parallelism modes
   - Originally from the triton repo
   - Advanced users and performance tuning

## Requirements

- PyTorch with NCCL support
- Triton
- Multiple CUDA GPUs (default: 8)

No Iris installation required!

