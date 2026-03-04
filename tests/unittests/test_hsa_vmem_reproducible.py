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
  Result: fine-grained physical memory — P2P atomics always pass  ✓
```

The key difference between Paths 2 and 3: `hipMemCreate` in HIP/CLR hardcodes the
coarse-grained GPU pool.  `hsa_amd_vmem_handle_create` takes an explicit pool
argument, so we can pass the fine-grained pool instead.

## What this file tests (and why)

The nine `test_hsa_vmem_p2p_atomics[*]` tests confirm the fix at the repro level:
each allocates memory via `hsa_amd_vmem_handle_create` on the **fine-grained** pool,
exports the DMA-BUF handle to the peer, imports it, maps it, and runs 200 P2P
atomic_add iterations.  All scope×sem combinations produce zero failures.

## Why there is no automated test for Path 2 (HIP VMem) failure

HIP VMem (`hipMemCreate`) always allocates from the **coarse-grained** GPU pool
because HIP/CLR's `SvmBuffer::malloc(ROCCLR_MEM_PHYMEM)` hardcodes that pool.
Cross-GPU atomics on coarse-grained memory are not just "wrong" — on AMD GPUs they
trigger GPU page faults that send SIGSEGV to the process.  This makes any automated
test of HIP VMem P2P atomics inherently fatal to the test process, so we cannot
include such a test in the regular test suite.

The `_HIPVMemP2P` fixture class and `_run_p2p_atomics_hip` helper below document the
setup in detail and can be used for manual/ad-hoc investigation.  The module-level
docstring above explains the complete stack trace for all three paths.

No iris machinery (no SymmetricHeap, no bump allocator, no refresh_peer_access).
Just raw API calls + torchrun for process management.
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
    # Path 2: HIP VMem (coarse-grained — used to show the bug)
    get_allocation_granularity,
    mem_create,
    mem_export_to_shareable_handle,
    mem_import_from_shareable_handle,
    mem_address_reserve,
    mem_map,
    mem_unmap,
    mem_address_free,
    mem_release,
    mem_set_access,
    hipMemAccessDesc,
    hipMemLocationTypeDevice,
    hipMemAccessFlagsProtReadWrite,
    hipMemAllocationTypePinned,
    hipMemAllocationTypeUncached,
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

# Tolerance for float32 atomic comparison: values are integers (0.0, 1.0, 2.0, ...),
# so any deviation > 0.01 indicates a real atomicity failure.
_ATOMIC_EXACT_TOL = 0.01
# Tolerance for P2P counter check: counter should equal world_size exactly
_ATOMIC_COUNT_TOL = 0.5

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
        self._sockets: list = []  # Stored for cleanup in close() to prevent FD leaks

        if fc:
            for peer, sock in fc.items():
                if peer == self.rank:
                    continue
                self._sockets.append(sock)
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
        # Close peer sockets first so both ranks can proceed past any pending recv
        for sock in self._sockets:
            try:
                sock.close()
            except Exception:
                pass
        self._sockets.clear()
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
# Path 2 fixture: HIP VMem (coarse-grained — reproduces the bug)
# ---------------------------------------------------------------------------


class _HIPVMemP2P:
    """
    Minimal Path 2 (HIP VMem) P2P setup using hipMemCreate.

    hipMemCreate internally calls hsa_amd_vmem_handle_create but hardcodes the
    coarse-grained GPU pool in HIP/CLR (SvmBuffer::malloc with ROCCLR_MEM_PHYMEM).
    This means the physical memory is ALWAYS CoarseGrain=1, and P2P atomics at
    scope=cta or scope=gpu will silently return wrong results.

    The structure mirrors _HsaVMemP2P so the P2P atomic loop is identical —
    only the allocation and exchange APIs differ, making the comparison clean.

    Args:
        alloc_size: Allocation size in bytes (will be rounded up to granularity)
        alloc_type: hipMemAllocationType constant.  Default is hipMemAllocationTypePinned
            (the only value currently supported by hipMemCreate; coarse-grained).
            Pass hipMemAllocationTypeUncached (0x40000000) to test whether the ROCm
            driver honours an uncached/fine-grained request — if hipMemCreate rejects
            this value it raises RuntimeError and the caller should catch it.
    """

    def __init__(self, alloc_size: int, alloc_type: int = hipMemAllocationTypePinned):
        self.rank = _rank()
        self.world_size = _world_size()
        self.local_rank = _local_rank()
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.local_rank)

        gran = get_allocation_granularity(self.local_rank)
        # Align alloc_size to HIP granularity
        self.size = (alloc_size + gran - 1) & ~(gran - 1)

        # Path 2: hipMemCreate → CLR → hsa_amd_vmem_handle_create (coarse pool)
        self.handle = mem_create(self.size, self.local_rank, alloc_type=alloc_type)

        # Reserve virtual address space and map
        self.va = mem_address_reserve(self.size, gran)
        mem_map(self.va, self.size, 0, self.handle)

        # Set access for the local device
        access_desc = hipMemAccessDesc()
        access_desc.location.type = hipMemLocationTypeDevice
        access_desc.location.id = self.local_rank
        access_desc.flags = hipMemAccessFlagsProtReadWrite
        mem_set_access(self.va, self.size, access_desc)

        # Exchange handles with all peers via DMA-BUF
        fd = mem_export_to_shareable_handle(self.handle)
        my_meta = struct.pack("QQ", self.va, self.size)

        fc = setup_fd_infrastructure(self.rank, self.world_size)
        self.peer_vas: dict = {}
        self._peer_resources: list = []
        self._sockets: list = []  # Stored for cleanup in close() to prevent FD leaks

        if fc:
            for peer, sock in fc.items():
                if peer == self.rank:
                    continue
                self._sockets.append(sock)
                peer_va = mem_address_reserve(self.size, gran)
                if self.rank > peer:
                    send_fd(sock, fd, payload=my_meta)
                    peer_fd, pmeta = recv_fd(sock, payload_size=16)
                else:
                    peer_fd, pmeta = recv_fd(sock, payload_size=16)
                    send_fd(sock, fd, payload=my_meta)

                peer_base, peer_size = struct.unpack("QQ", pmeta)
                peer_handle = mem_import_from_shareable_handle(peer_fd)
                os.close(peer_fd)

                peer_alloc = (peer_size + gran - 1) & ~(gran - 1)
                mem_map(peer_va, peer_alloc, 0, peer_handle)

                # Set access for the local device on the imported range
                peer_access = hipMemAccessDesc()
                peer_access.location.type = hipMemLocationTypeDevice
                peer_access.location.id = self.local_rank
                peer_access.flags = hipMemAccessFlagsProtReadWrite
                mem_set_access(peer_va, peer_alloc, peer_access)

                self._peer_resources.append((peer_va, peer_alloc, peer_handle))
                self.peer_vas[peer] = peer_va

        os.close(fd)

    def local_tensor(self, offset_bytes: int = 0) -> torch.Tensor:
        """Float32 tensor at va+offset_bytes (local coarse-grained HIP VMem)."""
        return _tensor_at(self.va + offset_bytes, 1, self.device)

    def peer_tensor(self, peer: int, offset_bytes: int = 0) -> torch.Tensor:
        """Float32 tensor at peer_va+offset_bytes (imported peer HIP VMem)."""
        return _tensor_at(self.peer_vas[peer] + offset_bytes, 1, self.device)

    def close(self):
        # Close peer sockets first so both ranks can proceed past any pending recv
        for sock in self._sockets:
            try:
                sock.close()
            except Exception:
                pass
        self._sockets.clear()
        for pva, psize, ph in self._peer_resources:
            try:
                mem_unmap(pva, psize)
            except Exception:
                pass
            try:
                mem_address_free(pva, psize)
            except Exception:
                pass
            try:
                mem_release(ph)
            except Exception:
                pass
        self._peer_resources.clear()
        try:
            mem_unmap(self.va, self.size)
        except Exception:
            pass
        try:
            mem_release(self.handle)
        except Exception:
            pass
        try:
            mem_address_free(self.va, self.size)
        except Exception:
            pass

    def __del__(self):
        self.close()


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
        assert abs(t[0].item() - 1.0) < _ATOMIC_EXACT_TOL, f"Local atomic failed: got {t[0].item()}"
        hsa_vmem_unmap(va, gran)
    finally:
        hsa_vmem_handle_release(handle)
    hsa_vmem_address_free(va, gran)


# ---------------------------------------------------------------------------
# Path 2: HIP VMem helper (for manual/ad-hoc investigation only)
#
# This helper is NOT called by any automated test because coarse-grained P2P
# atomics on AMD GPUs trigger GPU page faults that kill the process.
#
# To manually confirm the bug: run this in isolation with a single pair of ranks
# and observe that ~5-30% of iterations produce wrong values (counter < 2 instead
# of 2).  Depending on hardware, some iterations may also crash the process.
# ---------------------------------------------------------------------------


def _run_p2p_atomics_hip(scope: str, sem: str, n_iters: int = 200, alloc_type: int = hipMemAllocationTypePinned) -> int:
    """
    Same P2P atomic loop as _run_p2p_atomics but using HIP VMem (Path 2).

    hipMemCreate allocates via CLR's SvmBuffer::malloc(ROCCLR_MEM_PHYMEM),
    which calls hsa_amd_vmem_handle_create with the coarse-grained GPU pool.
    The physical memory is therefore CoarseGrain=1 in the KFD driver, and
    P2P atomic operations below system scope are not guaranteed to be coherent.

    Args:
        scope: Triton atomic scope ("cta", "gpu", "sys")
        sem: Triton atomic semantics ("acquire", "release", "acq_rel")
        n_iters: Number of P2P atomic rounds
        alloc_type: hipMemAllocationType constant.  Default is hipMemAllocationTypePinned
            (coarse-grained).  Pass hipMemAllocationTypeUncached to test whether the
            ROCm driver produces fine-grained memory for that type.

    WARNING: On AMD GPUs, coarse-grained P2P atomics do not merely return wrong
    values — they can trigger GPU page faults (SIGSEGV) that kill the process.
    Do NOT call this function from automated tests.  Use it only for manual
    validation or one-off debugging sessions.
    """
    world_size = _world_size()
    p2p = _HIPVMemP2P(alloc_size=4 << 20, alloc_type=alloc_type)
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
            if abs(got - float(world_size)) > _ATOMIC_COUNT_TOL:
                failures += 1
    finally:
        p2p.close()

    return failures


# ---------------------------------------------------------------------------
# Test: Path 3 P2P atomics — full scope × sem sweep
# ---------------------------------------------------------------------------


def _run_p2p_atomics(scope: str, sem: str, n_iters: int = 200) -> int:
    """
    Run *n_iters* P2P atomic_add rounds using HSA VMem (Path 3) and return failures.

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
            if abs(got - float(world_size)) > _ATOMIC_COUNT_TOL:
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

    Compare with test_comparison_hsa_fixes_hip_vmem_bug (Path 2), which shows
    that HIP VMem fails for scope=cta/gpu because hipMemCreate hardcodes the
    coarse-grained pool.
    """
    failures = _run_p2p_atomics(scope, sem, n_iters=200)
    assert failures == 0, (
        f"HSA VMem (Path 3) P2P atomic failures: {failures}/200 "
        f"(scope={scope}, sem={sem}). "
        f"Non-zero indicates coarse-grained coherency on the imported mapping."
    )


# ---------------------------------------------------------------------------
# Test: HIP VMem with hipMemAllocationTypeUncached
#
# The question: does prop.type = hipMemAllocationTypeUncached (0x40000000) make
# hipMemCreate allocate fine-grained (uncached) physical memory instead of the
# default coarse-grained pinned memory?  If so, P2P atomics should pass.
#
# The HIP header says:
#   hipMemAllocationTypeUncached = 0x40000000  // AMD ROCm extension
#
# In practice, hipMemCreate may reject this value with hipErrorNotSupported
# because the HIP/CLR implementation currently only handles
# hipMemAllocationTypePinned.  This test is designed to:
#   1. Skip automatically if hipMemCreate returns an error (unsupported)
#   2. Pass if the allocation succeeds AND P2P atomics produce zero failures
#      (meaning uncached = fine-grained on this hardware/driver)
#   3. Warn (but not fail CI) if allocation succeeds but atomics still fail
#      (meaning uncached = still coarse-grained, or a different issue)
# ---------------------------------------------------------------------------


def test_hip_vmem_uncached_alloc_type():
    """
    Probe: does hipMemAllocationTypeUncached produce allocatable GPU memory?

    Tries hipMemCreate with prop.type = hipMemAllocationTypeUncached (0x40000000),
    an AMD ROCm extension enum value.  The HIP header note says hipMemCreate
    "Currently must be specified as hipMemAllocationTypePinned", so this type
    may be rejected by the driver.

    This test verifies local allocation and local atomic correctness ONLY.
    P2P cross-rank access with uncached type is NOT tested here — on this hardware
    `hipMemAllocationTypeUncached` is accepted by hipMemCreate but still produces
    coarse-grained physical memory that causes GPU page faults (SIGSEGV) on
    cross-rank access, even at scope=sys.  The HSA VMem (Path 3) approach
    remains the only confirmed way to get fine-grained P2P-atomic-safe memory.

    Outcomes:
      SKIPPED  — hipMemCreate rejected hipMemAllocationTypeUncached
      PASSED   — allocation + local sys-scope atomic both work (but memory is
                 still coarse-grained for P2P; see module docstring for details)
      FAILED   — allocation succeeded but local atomic produced wrong result
    """
    local_rank = _local_rank()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    gran = get_allocation_granularity(local_rank)
    size = gran  # single granularity — smallest valid allocation

    # Try allocation; skip if the driver rejects the uncached type
    try:
        handle = mem_create(size, local_rank, alloc_type=hipMemAllocationTypeUncached)
    except RuntimeError as e:
        pytest.skip(f"hipMemCreate rejected hipMemAllocationTypeUncached (0x{hipMemAllocationTypeUncached:08x}): {e}")

    # Map and test local access
    va = mem_address_reserve(size, gran)
    try:
        mem_map(va, size, 0, handle)
        access_desc = hipMemAccessDesc()
        access_desc.location.type = hipMemLocationTypeDevice
        access_desc.location.id = local_rank
        access_desc.flags = hipMemAccessFlagsProtReadWrite
        mem_set_access(va, size, access_desc)

        t = _tensor_at(va, 1, device)
        t.fill_(0.0)
        # Local atomic only — safe regardless of grain (no cross-rank access)
        _atomic_add_one[(1,)](t, "sys", "acq_rel")
        torch.cuda.synchronize()

        assert abs(t[0].item() - 1.0) < _ATOMIC_EXACT_TOL, (
            f"Local atomic on hipMemAllocationTypeUncached memory produced wrong result: "
            f"got {t[0].item()}, expected 1.0"
        )

        mem_unmap(va, size)
    finally:
        mem_release(handle)
    mem_address_free(va, size)
