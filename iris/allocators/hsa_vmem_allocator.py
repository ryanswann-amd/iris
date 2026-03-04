# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
HSA Virtual Memory allocator for Iris symmetric heap — Path 3.

## Memory Path Overview

Three paths exist for allocating GPU memory that supports cross-GPU (P2P)
atomic operations in Iris. They differ in the API layer used and the resulting
memory coherency:

### Path 1: hipExtMallocWithFlags (Fine-Grained malloc)
  API:  hipExtMallocWithFlags(ptr, size, hipDeviceMallocFinegrained)
  HSA:  hsa_amd_memory_pool_allocate on fine-grained device pool
  KFD:  hsaKmtAllocMemory with CoarseGrain=0

  Pros: Simple, fine-grained → P2P atomics work
  Cons: No VA control; heap is a flat contiguous region

  Used by: VMemAllocator (iris/allocators/vmem_allocator.py)

### Path 2: HIP Virtual Memory (hipMemCreate + hipMemAddressReserve)
  APIs: hipMemAddressReserve → hipMemCreate → hipMemMap → hipMemSetAccess
  HIP→CLR: ROCCLR_MEM_PHYMEM flag → SvmBuffer::malloc
  CLR→HSA: hsa_amd_vmem_handle_create on COARSE-GRAINED pool
  KFD:  CoarseGrain=1, NoAddress=1

  Pros: Full VA space control
  Cons: ALWAYS coarse-grained → P2P atomics (scope=cta/gpu) fail intermittently
        HIP hardcodes the coarse-grained pool for hipMemCreate

  Not used in Iris (removed due to atomic failures)

### Path 3: HSA Virtual Memory (this allocator)
  APIs: hsa_amd_vmem_address_reserve → hsa_amd_vmem_handle_create (fine-grained
        pool) → hsa_amd_vmem_map → hsa_amd_vmem_set_access
  KFD:  CoarseGrain=0 (from fine-grained pool), NoAddress=1

  Pros: Fine-grained + full VA space control (best of both)
  Cons: More complex setup (enumerate HSA agents and pools at init)

  The key advantage over Path 2: hsa_amd_vmem_handle_create takes an EXPLICIT
  pool argument, so we can pass the fine-grained GPU local pool instead of the
  coarse-grained pool that HIP/CLR hardcodes.

  Used by: HsaVMemAllocator (this file)

See iris/hip.py for the full architecture diagram.
"""

import math
import struct
import torch
from typing import Dict, Optional
from threading import Lock

from .base import BaseAllocator
from ..hip import (
    hsa_init,
    hsa_shut_down,
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
    hsa_amd_vmem_alloc_handle_t,
    hsa_agent_t,
)
from ..fd_passing import send_fd, recv_fd, managed_fd


class HsaVMemAllocator(BaseAllocator):
    """
    HSA Virtual Memory allocator using fine-grained GPU local memory (Path 3).

    Combines the VA control of VMem with fine-grained memory for correct P2P
    atomic operations. Uses HSA APIs directly instead of HIP VMem APIs to
    explicitly choose the fine-grained GPU memory pool.

    Key advantage over HIP VMem (Path 2):
      - hsa_amd_vmem_handle_create takes an explicit pool argument
      - We pass the fine-grained pool → KFD allocates with CoarseGrain=0
      - P2P atomics (scope=cta/gpu/sys) work correctly

    Key advantage over Path 1 (malloc_fine_grained):
      - Full virtual address space control
      - Can reserve a large VA range and map individual segments on demand

    Args:
        heap_size: Total size of the heap in bytes
        device_id: GPU device ID
        rank: Current rank ID
        world_size: Total number of ranks
    """

    def __init__(
        self,
        heap_size: int,
        device_id: int,
        rank: int,
        world_size: int,
    ):
        super().__init__(heap_size, device_id, rank, world_size)
        self.device = torch.device(f"cuda:{device_id}")
        self.lock = Lock()

        # Initialize HSA runtime
        hsa_init()

        # Find this device's GPU agent and fine-grained pool
        all_agents = hsa_get_gpu_agents()
        if len(all_agents) <= device_id:
            hsa_shut_down()
            raise RuntimeError(f"Not enough GPU agents: device_id={device_id}, found {len(all_agents)} agents")
        self._agent: hsa_agent_t = all_agents[device_id]
        self._all_agents = all_agents
        self._fine_pool = hsa_get_fine_grained_pool(self._agent)

        # Two granularities:
        #   _pool_granularity: required alignment for hsa_vmem_handle_create (typically 2MB)
        #   granularity: per-tensor allocation alignment within the pre-mapped heap (4KB, same as HIP)
        # Individual tensor allocations do NOT need to be 2MB-aligned; only the physical
        # handle creation requires pool granularity alignment.
        self._pool_granularity = hsa_get_pool_granularity(self._fine_pool)
        self.granularity = 4096  # 4KB per-tensor allocation alignment (same as HIP VMem)

        # Align heap size to pool granularity (required by hsa_vmem_handle_create)
        self.aligned_heap_size = (heap_size + self._pool_granularity - 1) & ~(self._pool_granularity - 1)

        # Reserve the virtual address space upfront
        self.base_va = hsa_vmem_address_reserve(self.aligned_heap_size)

        # Allocate and map the entire heap as a single fine-grained physical block.
        # We could map on demand, but a single upfront allocation is simpler and
        # matches the behavior of VMemAllocator (Path 1).
        self._heap_handle: hsa_amd_vmem_alloc_handle_t = hsa_vmem_handle_create(self._fine_pool, self.aligned_heap_size)
        hsa_vmem_map(self.base_va, self.aligned_heap_size, self._heap_handle)

        # Set read/write access for all GPU agents in the system.
        # This allows all GPUs to access the memory for P2P atomics.
        hsa_vmem_set_access(self.base_va, self.aligned_heap_size, self._all_agents)

        self._peer_handles: Dict[int, hsa_amd_vmem_alloc_handle_t] = {}
        self._peer_vas: Dict[int, int] = {}
        self.heap_bases_array = None
        self._closed = False

    def get_base_address(self) -> int:
        """Get the base virtual address of the heap."""
        return self.base_va

    def get_minimum_allocation_size(self) -> int:
        """Minimum allocation size (one granule for HSA VMem compatibility)."""
        return self.granularity

    def allocate(self, num_elements: int, dtype: torch.dtype, alignment: int = 1024) -> torch.Tensor:
        """
        Allocate a tensor from the fine-grained HSA VMem heap using bump allocation.

        Args:
            num_elements: Number of elements to allocate
            dtype: PyTorch data type
            alignment: Alignment requirement in bytes

        Returns:
            PyTorch tensor wrapping the allocated memory

        Raises:
            RuntimeError: If the heap is out of space
        """
        with self.lock:
            element_size = torch.tensor([], dtype=dtype).element_size()
            size_in_bytes = num_elements * element_size
            aligned_size = math.ceil(size_in_bytes / alignment) * alignment

            if self.heap_offset + aligned_size > self.aligned_heap_size:
                raise RuntimeError(
                    f"HsaVMemAllocator: out of space for allocation of {aligned_size} bytes "
                    f"at offset {self.heap_offset} (heap size {self.aligned_heap_size})"
                )

            start = self.heap_offset
            self.heap_offset += aligned_size
            target_va = self.base_va + start

            class CUDAArrayInterface:
                def __init__(self, ptr, size_bytes):
                    self.ptr = ptr
                    self.size_bytes = size_bytes

                @property
                def __cuda_array_interface__(self):
                    return {
                        "shape": (self.size_bytes,),
                        "typestr": "|u1",
                        "data": (self.ptr, False),
                        "version": 3,
                    }

            cuda_array = CUDAArrayInterface(target_va, size_in_bytes)
            tensor_bytes = torch.as_tensor(cuda_array, device=self.device)
            full = tensor_bytes.view(dtype)
            if num_elements == 0:
                return full.narrow(0, 1, 0)
            return full.narrow(0, 0, num_elements)

    def get_shareable_handle(self) -> tuple:
        """
        Export the heap's physical memory as a shareable DMA-BUF handle.

        Returns:
            tuple: (fd, base_ptr, base_size) where fd is the DMA-BUF file
                   descriptor and base_ptr/base_size describe the exported range

        Raises:
            RuntimeError: If export fails
        """
        fd = hsa_vmem_export_shareable_handle(self._heap_handle)
        return fd, self.base_va, self.aligned_heap_size

    def establish_peer_access(self, all_bases: Dict[int, int], connections: Optional[Dict] = None):
        """
        Establish HSA VMem access to peer memory for symmetric addressing.

        Exchanges DMA-BUF handles with peer ranks, imports peer handles, and
        maps them into a reserved VA range. The mapped_ptr for each peer is used
        as heap_bases[peer] for address translation in Triton kernels.

        Args:
            all_bases: Dictionary mapping rank -> base address
            connections: Optional peer connections (Unix sockets) for FD exchange
        """
        import numpy as np

        heap_bases_array = np.zeros(self.num_ranks, dtype=np.uint64)

        if connections is not None:
            # HsaVMemAllocator uses a fixed pre-allocated heap with stable physical memory.
            # Once peer access is established (handles and VAs created), the mappings
            # remain valid for the allocator's lifetime — no need to re-create on
            # every allocation. Skip re-creation if peer VAs are already set up.
            if self._peer_handles:
                # Already established — just repopulate heap_bases_array from existing VAs
                for peer, peer_va in self._peer_vas.items():
                    heap_bases_array[peer] = peer_va
                heap_bases_array[self.cur_rank] = all_bases[self.cur_rank]
                self.heap_bases_array = heap_bases_array
                return

            my_fd, my_base, my_size = self.get_shareable_handle()
            heap_base = self.get_base_address()
            my_metadata = struct.pack("QQQ", my_base, my_size, heap_base)

            with managed_fd(my_fd):
                for peer, sock in connections.items():
                    if peer == self.cur_rank:
                        continue

                    # Higher rank sends first to avoid deadlock
                    if self.cur_rank > peer:
                        send_fd(sock, my_fd, payload=my_metadata)
                        peer_handle_fd, peer_metadata = recv_fd(sock, payload_size=24)
                    else:
                        peer_handle_fd, peer_metadata = recv_fd(sock, payload_size=24)
                        send_fd(sock, my_fd, payload=my_metadata)

                    peer_base, peer_size, peer_heap_base = struct.unpack("QQQ", peer_metadata)

                    # Import peer's handle from DMA-BUF and map it to our VA space
                    with managed_fd(peer_handle_fd):
                        imported_handle = hsa_vmem_import_shareable_handle(peer_handle_fd)

                    # Reserve a new VA range for the peer's memory
                    peer_va = hsa_vmem_address_reserve(self.aligned_heap_size)
                    # Align to pool granularity (required by hsa_vmem_map)
                    peer_alloc_size = (peer_size + self._pool_granularity - 1) & ~(self._pool_granularity - 1)
                    hsa_vmem_map(peer_va, peer_alloc_size, imported_handle)
                    hsa_vmem_set_access(peer_va, peer_alloc_size, self._all_agents)

                    self._peer_handles[peer] = imported_handle
                    self._peer_vas[peer] = peer_va
                    heap_bases_array[peer] = peer_va

            heap_bases_array[self.cur_rank] = all_bases[self.cur_rank]
        else:
            heap_bases_array[self.cur_rank] = all_bases[self.cur_rank]

        self.heap_bases_array = heap_bases_array

    def get_device(self) -> torch.device:
        """Get the PyTorch device for this allocator."""
        return self.device

    def owns_tensor(self, tensor: torch.Tensor) -> bool:
        """
        Check if a tensor's memory belongs to this allocator's heap.

        Args:
            tensor: Tensor to check

        Returns:
            True if tensor is within this allocator's heap
        """
        if not tensor.is_cuda:
            return False
        if tensor.numel() == 0:
            return True
        ptr = tensor.data_ptr()
        return self.base_va <= ptr < self.base_va + self.aligned_heap_size

    def import_external_tensor(self, external_tensor: torch.Tensor) -> torch.Tensor:
        """
        Import an external PyTorch tensor into the symmetric heap (as_symmetric).

        Uses copy semantics: allocates space on the fine-grained symmetric heap
        and copies the data from the external tensor. The returned tensor is
        independent of the input tensor.

        Args:
            external_tensor: External PyTorch tensor to import (must be CUDA, contiguous)

        Returns:
            New tensor on the symmetric heap with a copy of the external tensor's data

        Raises:
            RuntimeError: If tensor is not a CUDA tensor or not contiguous
        """
        if not external_tensor.is_cuda:
            raise RuntimeError("Can only import CUDA tensors")
        if not external_tensor.is_contiguous():
            raise RuntimeError("Only contiguous tensors can be imported; call .contiguous() before as_symmetric()")
        num_elements = external_tensor.numel()
        dtype = external_tensor.dtype
        shape = external_tensor.shape
        heap_tensor = self.allocate(num_elements, dtype)
        heap_tensor = heap_tensor.reshape(shape).copy_(external_tensor)
        return heap_tensor

    def close(self):
        """Release all HSA VMem resources."""
        if self._closed:
            return
        self._closed = True

        # Release peer mappings
        for peer, peer_handle in self._peer_handles.items():
            peer_va = self._peer_vas.get(peer)
            if peer_va:
                try:
                    hsa_vmem_unmap(peer_va, self.aligned_heap_size)
                except Exception:
                    pass
                try:
                    hsa_vmem_address_free(peer_va, self.aligned_heap_size)
                except Exception:
                    pass
            try:
                hsa_vmem_handle_release(peer_handle)
            except Exception:
                pass
        self._peer_handles.clear()
        self._peer_vas.clear()

        # Release local heap
        if hasattr(self, "_heap_handle"):
            try:
                hsa_vmem_unmap(self.base_va, self.aligned_heap_size)
            except Exception:
                pass
            try:
                hsa_vmem_handle_release(self._heap_handle)
            except Exception:
                pass

        if hasattr(self, "base_va") and self.base_va:
            try:
                hsa_vmem_address_free(self.base_va, self.aligned_heap_size)
            except Exception:
                pass
            self.base_va = 0

        try:
            hsa_shut_down()
        except Exception:
            pass

    def __del__(self):
        """Cleanup HSA VMem resources on deletion."""
        self.close()
