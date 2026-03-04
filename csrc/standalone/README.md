# Standalone P2P VMem Atomic Examples (HIP/C++)

Minimal, standalone two-process HIP/C++ programs that demonstrate the three
GPU memory allocation paths for multi-GPU P2P atomic operations on AMD GPUs.

## Background

For correct cross-GPU (P2P) atomic operations, the physical memory must be
**fine-grained** (`CoarseGrain=0` in the KFD driver).  Two common allocation
paths produce fine-grained memory; one (the default HIP Virtual Memory API)
always produces coarse-grained memory:

| Path | API | KFD CoarseGrain | P2P atomics |
|------|-----|:---------:|:-----------:|
| 1 | `hipExtMallocWithFlags(hipDeviceMallocFinegrained)` | 0 | ✓ |
| **2** | **`hipMemCreate` (any `hipMemAllocationType`)** | **1 (hardcoded)** | **✗ crashes** |
| 3 | `hsa_amd_vmem_handle_create(fine_grained_pool)` | 0 | ✓ |

Path 2 always fails because HIP/CLR's `SvmBuffer::malloc(ROCCLR_MEM_PHYMEM)`
hardcodes the coarse-grained GPU pool regardless of the `prop.type` field (even
`hipMemAllocationTypeUncached = 0x40000000` is silently ignored).

These examples let you reproduce the bug and the fix on any system with ≥2
AMD GPUs running ROCm.

## Files

| File | Description |
|------|-------------|
| `p2p_atomics_hsa.cpp` | **Path 3** — HSA fine-grained VMem (correct) |
| `p2p_atomics_hip.cpp` | **Path 2** — HIP VMem (demonstrates the coarse-grained bug) |
| `Makefile` | Builds both examples |

## Build

```bash
# From this directory
make

# With a custom ROCm installation
make ROCM_DIR=/path/to/rocm
```

## Run

### HSA example (Path 3 — fine-grained, CORRECT)

```bash
./p2p_atomics_hsa [N_ITERS]
```

- Spawns two processes (rank 0 → GPU 0, rank 1 → GPU 1) via `fork()`
- Allocates physical memory with `hsa_amd_vmem_handle_create` on the
  **fine-grained** GPU memory pool
- Exports handles as DMA-BUF file descriptors via SCM_RIGHTS
- Imports peer handles and maps them to local virtual address space
- Runs P2P atomic `atomicAdd` at **agent scope** and **system scope**
- Expected output: both scopes pass with 0 failures

```
p2p_atomics_hsa: PATH 3 — HSA fine-grained VMem
  N_ITERS=200  GPUs=2

[rank 0] starting on GPU 0
[rank 0] fine-grained pool granularity = 2097152 bytes
[rank 0] my_va=0x...  peer_va=0x...
[rank 0] agent-scope: 0/200 failures   sys-scope: 0/200 failures
[rank 0] PASS
...
Overall: PASS
```

### HIP example (Path 2 — coarse-grained, BUG DEMONSTRATION)

```bash
# Safe mode: P2P non-atomic reads only (setup verification)
./p2p_atomics_hip [--pinned|--uncached]

# Atomic mode: P2P atomics — WARNING: causes GPU page fault!
./p2p_atomics_hip --atomics [--agent|--sys] [N_ITERS]
```

#### Options

| Flag | Description |
|------|-------------|
| `--pinned` | `hipMemAllocationTypePinned` (0x1) — default |
| `--uncached` | `hipMemAllocationTypeUncached` (0x40000000) — AMD extension |
| `--atomics` | Enable P2P atomic kernels (**WARNING: crashes on coarse-grained memory**) |
| `--agent` | Use agent-scope atomics (most likely to crash) |
| `--sys` | Use system-scope atomics (default when `--atomics` given) |

#### Expected behaviour

| Mode | Expected |
|------|----------|
| `--pinned` (read only) | PASS — P2P non-atomic reads work on coarse-grained memory |
| `--uncached` (read only) | PASS or driver-level skip — uncached type is accepted but silently ignored by CLR |
| `--atomics --agent` | **CRASH** — GPU page fault (SIGSEGV) from coarse-grained P2P atomic |
| `--atomics --sys` | **CRASH** — system-scope P2P atomics also fault on coarse-grained memory |
| `--uncached --atomics` | **Same crash** — uncached type does NOT change the allocation to fine-grained |

The key finding from testing on MI300X:

> `hipMemAllocationTypeUncached` (0x40000000) is accepted by `hipMemCreate`
> (returns `hipSuccess`) but **does not change the memory grain**.  CLR always
> routes through the coarse-grained pool regardless of `prop.type`.  P2P atomics
> still cause GPU page faults identical to `hipMemAllocationTypePinned`.

## Implementation notes

- Both programs use `fork()` + `socketpair(AF_UNIX)` — no MPI dependency
- DMA-BUF file descriptors are exchanged via `sendmsg(SCM_RIGHTS)` /
  `recvmsg(SCM_RIGHTS)`
- The barrier between ranks is a 1-byte exchange over the Unix socket
- Device kernels use `__hip_atomic_fetch_add` with `__HIP_MEMORY_SCOPE_AGENT`
  and `__HIP_MEMORY_SCOPE_SYSTEM` for the two scope levels

## The fix

Use `hsa_amd_vmem_handle_create` directly (Path 3) with the fine-grained pool
found by iterating `hsa_amd_agent_iterate_memory_pools` and checking
`HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED`.  This is exactly what Iris does
in its `HsaVMemAllocator` (`iris/allocators/hsa_vmem_allocator.py`).
