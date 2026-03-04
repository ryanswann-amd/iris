# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Minimal P2P atomic reproducible — comparing three GPU memory allocation paths.

## Background: Three paths for P2P-atomic-safe memory on AMD GPUs

For correct cross-GPU (P2P) atomic operations in Triton (scope=cta/gpu/sys),
the physical memory must be **fine-grained**.  Three paths exist; only Paths 1
and 3 produce fine-grained memory.

```
Path 1 — hipExtMallocWithFlags(hipDeviceMallocFinegrained)
  Stack:  HIP → hsa_amd_memory_pool_allocate (fine-grained GPU pool)
                → KFD: hsaKmtAllocMemory(CoarseGrain=0)
  P2P:    hipExternalMemoryHandleTypeOpaqueFd (dma-buf) → hipImportExternalMemory
  Result: fine-grained, KNOWN GOOD  ✓

Path 2 — hipMemCreate + hipMemAddressReserve + hipMemMap
  Stack:  HIP → CLR SvmBuffer::malloc(ROCCLR_MEM_PHYMEM)
                → hsa_amd_vmem_handle_create (coarse-grained GPU pool, hardcoded)
                → KFD: hsaKmtAllocMemory(CoarseGrain=1, NoAddress=1)
  P2P:    hipMemImportFromShareableHandle + hipMemMap
  Result: ALWAYS coarse-grained — P2P atomics (scope=cta/gpu) fail  ✗

Path 3 — hsa_amd_vmem_handle_create on fine-grained pool (direct HSA)
  Stack:  hsa_amd_vmem_handle_create(fine_grained_pool, ...)  ← caller chooses pool
                → KFD: hsaKmtAllocMemory(CoarseGrain=0, NoAddress=1)
  P2P:    hsa_amd_vmem_export_shareable_handle → hsa_amd_vmem_import_shareable_handle
                → hsa_amd_vmem_map + hsa_amd_vmem_set_access
  Result: fine-grained physical memory — this test validates P2P atomic correctness
```

The key difference between Paths 2 and 3: `hipMemCreate` in HIP/CLR hardcodes the
coarse-grained GPU pool.  `hsa_amd_vmem_handle_create` takes an explicit pool
argument, so we can pass the fine-grained pool instead.

## What this test does

Each test function:
1. Both ranks allocate fine-grained VMem (Path 3) via HSA APIs directly
2. Exchange DMA-BUF file descriptors via Unix SCM_RIGHTS
3. Each rank imports the peer's handle and maps it into a reserved VA range
4. Both ranks run a single-element atomic_add kernel repeatedly
5. After a barrier, each rank checks that its counter == world_size
6. Failures indicate non-fine-grained coherency (same symptom as Path 2)

No iris machinery (no SymmetricHeap, no bump allocator, no refresh_peer_access).
Just raw HSA API calls + torchrun for process management.
"""

import os
import struct

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl

from iris.hip import (
    hsa_init,
    hsa_get_gpu_agents,
    hsa_get_fine_grained_pool,
    hsa_get_pool_granularity,
    hsa_vmem_address_reserve,
    hsa_vmem_address_free,
    hsa_vmem_handle_create,
    hsa_vmem_handle_release,
    hsa_vmem_map,
    hsa_vmem_unmap,
    hsa_vmem_set_access,
    hsa_vmem_export_shareable_handle,
    hsa_vmem_import_shareable_handle,
)
from iris.fd_passing import setup_fd_infrastructure, send_fd, recv_fd

# ---------------------------------------------------------------------------
# Session-level HSA initialisation
#
# hsa_init / hsa_shut_down use a reference count internally.  Calling
# hsa_init() + hsa_shut_down() inside every test works on its own but
# causes hsa_init() to fail with OUT_OF_RESOURCES in the *next* test
# when the runtime hasn't fully released its internal threads yet.
#
# We therefore call hsa_init() once per process here and let the OS
# clean up the runtime when the process exits.
# ---------------------------------------------------------------------------
hsa_init()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def _local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def _tensor_at(va: int, n_floats: int, device: torch.device) -> torch.Tensor:
    """Create a float32 tensor backed by GPU memory at *va* (no copy)."""

    class _CUDAMem:
        def __init__(self, ptr, n):
            self._ptr = ptr
            self._n = n

        @property
        def __cuda_array_interface__(self):
            return {
                "shape": (self._n * 4,),
                "typestr": "|u1",
                "data": (self._ptr, False),
                "version": 3,
            }

    return torch.as_tensor(_CUDAMem(va, n_floats), device=device).view(torch.float32)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit
def _atomic_add_one(ptr, scope: tl.constexpr, sem: tl.constexpr):
    """Atomically add 1.0 to ptr[0]."""
    tl.atomic_add(ptr, 1.0, scope=scope, sem=sem)


# ---------------------------------------------------------------------------
# Core fixture: raw HSA VMem P2P setup
# ---------------------------------------------------------------------------


class _HsaVMemP2P:
    """
    Minimal per-test HSA VMem setup:
      - one fine-grained physical allocation per rank
      - DMA-BUF export/import with peers
      - all resources released on close()
    """

    def __init__(self, alloc_size: int):
        self.rank = _rank()
        self.world_size = _world_size()
        self.local_rank = _local_rank()
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.local_rank)

        agents = hsa_get_gpu_agents()
        assert len(agents) > self.local_rank, f"Expected >{self.local_rank} GPU agents"
        self.agents = agents
        self.agent = agents[self.local_rank]

        pool = hsa_get_fine_grained_pool(self.agent)
        gran = hsa_get_pool_granularity(pool)
        # Align alloc_size to pool granularity (required by hsa_vmem_handle_create)
        self.size = (alloc_size + gran - 1) & ~(gran - 1)

        # Allocate fine-grained physical memory (Path 3 key step)
        self.handle = hsa_vmem_handle_create(pool, self.size)

        # Reserve virtual address space and map
        self.va = hsa_vmem_address_reserve(self.size)
        hsa_vmem_map(self.va, self.size, self.handle)
        hsa_vmem_set_access(self.va, self.size, agents)

        # Exchange DMA-BUF handles with all peers
        fd = hsa_vmem_export_shareable_handle(self.handle)
        my_meta = struct.pack("QQ", self.va, self.size)

        fc = setup_fd_infrastructure(self.rank, self.world_size)
        self.peer_vas: dict = {}
        self._peer_handles: list = []

        if fc:
            for peer, sock in fc.items():
                if peer == self.rank:
                    continue
                peer_va = hsa_vmem_address_reserve(self.size)
                if self.rank > peer:
                    send_fd(sock, fd, payload=my_meta)
                    peer_fd, pmeta = recv_fd(sock, payload_size=16)
                else:
                    peer_fd, pmeta = recv_fd(sock, payload_size=16)
                    send_fd(sock, fd, payload=my_meta)

                peer_base, peer_size = struct.unpack("QQ", pmeta)
                peer_handle = hsa_vmem_import_shareable_handle(peer_fd)
                os.close(peer_fd)

                peer_alloc = (peer_size + gran - 1) & ~(gran - 1)
                hsa_vmem_map(peer_va, peer_alloc, peer_handle)
                hsa_vmem_set_access(peer_va, peer_alloc, agents)

                self._peer_handles.append((peer_va, peer_alloc, peer_handle))
                self.peer_vas[peer] = peer_va

        os.close(fd)

    def local_tensor(self, offset_bytes: int = 0) -> torch.Tensor:
        """Float32 tensor at va+offset_bytes (local fine-grained memory)."""
        return _tensor_at(self.va + offset_bytes, 1, self.device)

    def peer_tensor(self, peer: int, offset_bytes: int = 0) -> torch.Tensor:
        """Float32 tensor at peer_va+offset_bytes (imported peer memory)."""
        return _tensor_at(self.peer_vas[peer] + offset_bytes, 1, self.device)

    def close(self):
        for pva, psize, ph in self._peer_handles:
            try:
                hsa_vmem_unmap(pva, psize)
            except Exception:
                pass
            try:
                hsa_vmem_address_free(pva, psize)
            except Exception:
                pass
            try:
                hsa_vmem_handle_release(ph)
            except Exception:
                pass
        self._peer_handles.clear()
        try:
            hsa_vmem_unmap(self.va, self.size)
        except Exception:
            pass
        try:
            hsa_vmem_handle_release(self.handle)
        except Exception:
            pass
        try:
            hsa_vmem_address_free(self.va, self.size)
        except Exception:
            pass

    def __del__(self):
        self.close()


# ---------------------------------------------------------------------------
# Test: pool discovery (no P2P, single-rank safe)
# ---------------------------------------------------------------------------


def test_hsa_vmem_pool_discovery():
    """Verify hsa_get_fine_grained_pool finds a non-zero, allocatable pool."""
    agents = hsa_get_gpu_agents()
    assert len(agents) >= 1, "No GPU agents found"
    lr = _local_rank()
    assert len(agents) > lr
    pool = hsa_get_fine_grained_pool(agents[lr])
    assert pool.handle != 0
    gran = hsa_get_pool_granularity(pool)
    assert gran > 0


# ---------------------------------------------------------------------------
# Test: single-GPU alloc + map + local atomic (no P2P)
# ---------------------------------------------------------------------------


def test_hsa_vmem_single_gpu_alloc_and_atomic():
    """
    Allocate fine-grained VMem via Path 3, map it, run a local atomic.

    Validates the basic reserve→create→map→set_access→atomic pipeline
    without any cross-process communication.
    """
    local_rank = _local_rank()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    agents = hsa_get_gpu_agents()
    pool = hsa_get_fine_grained_pool(agents[local_rank])
    gran = hsa_get_pool_granularity(pool)

    va = hsa_vmem_address_reserve(gran)
    handle = hsa_vmem_handle_create(pool, gran)
    try:
        hsa_vmem_map(va, gran, handle)
        hsa_vmem_set_access(va, gran, agents)

        t = _tensor_at(va, 4, device)
        t.fill_(0.0)
        _atomic_add_one[(1,)](t, "sys", "acq_rel")
        torch.cuda.synchronize()
        assert abs(t[0].item() - 1.0) < 0.01, f"Local atomic failed: got {t[0].item()}"
        hsa_vmem_unmap(va, gran)
    finally:
        hsa_vmem_handle_release(handle)
    hsa_vmem_address_free(va, gran)


# ---------------------------------------------------------------------------
# Test: P2P atomics — scope × sem sweep (the key correctness test)
# ---------------------------------------------------------------------------


def _run_p2p_atomics(scope: str, sem: str, n_iters: int = 200) -> int:
    """
    Run *n_iters* P2P atomic_add rounds and return the number of failures.

    Setup (once, outside the loop):
      - Each rank allocates fine-grained VMem (Path 3)
      - DMA-BUF handles are exchanged between all pairs of ranks
      - Each rank imports the peer handle and maps it at a local VA

    Per iteration:
      - Zero the local counter
      - Barrier (all ranks ready)
      - LOCAL atomic: this rank adds 1 to its own counter
      - REMOTE atomic: this rank adds 1 to the peer's counter
      - Barrier (all atomics done)
      - Read local counter; expect world_size

    Because the physical memory was created from the fine-grained pool,
    and the imported mapping should preserve that property, both
    scope=cta and scope=gpu atomics should be coherent across GPUs.
    A non-zero failure count indicates coarse-grained behaviour.
    """
    rank = _rank()
    world_size = _world_size()
    p2p = _HsaVMemP2P(alloc_size=4 << 20)
    try:
        failures = 0
        local_t = p2p.local_tensor()

        for _ in range(n_iters):
            local_t.fill_(0.0)
            dist.barrier()

            # All ranks add 1 to their own counter
            _atomic_add_one[(1,)](local_t, scope, sem)

            # All ranks add 1 to every other rank's counter
            for peer_va in p2p.peer_vas.values():
                peer_t = _tensor_at(peer_va, 1, p2p.device)
                _atomic_add_one[(1,)](peer_t, scope, sem)

            torch.cuda.synchronize()
            dist.barrier()

            got = local_t[0].item()
            if abs(got - float(world_size)) > 0.5:
                failures += 1
    finally:
        p2p.close()

    return failures


@pytest.mark.parametrize("scope", ["cta", "gpu", "sys"])
@pytest.mark.parametrize("sem", ["acquire", "release", "acq_rel"])
def test_hsa_vmem_p2p_atomics(scope, sem):
    """
    Path 3 (HSA VMem, fine-grained pool) — P2P atomic correctness sweep.

    Runs 200 iterations for each (scope, sem) combination and asserts zero
    failures.  Any failure means the imported mapping is coarse-grained and
    does not support the requested atomic scope.

    Compare with Path 1 (hipExtMallocWithFlags), which is the known-good
    baseline and always produces zero failures.
    """
    failures = _run_p2p_atomics(scope, sem, n_iters=200)
    assert failures == 0, (
        f"HSA VMem (Path 3) P2P atomic failures: {failures}/200 "
        f"(scope={scope}, sem={sem}). "
        f"Non-zero indicates coarse-grained coherency on the imported mapping."
    )
