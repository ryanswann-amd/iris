# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
VMem-based allocator using HIP's fine-grained memory APIs.

This allocator uses hipExtMallocWithFlags (fine-grained) for physical memory,
which is required for correct P2P atomic operations (scope=cta/gpu) across GPUs.
hipMemCreate creates coarse-grained memory that causes intermittent failures
for cross-GPU atomics.
"""

import math
import struct
import torch
from typing import Dict, Optional
from threading import Lock

from .base import BaseAllocator
from ..hip import (
    get_allocation_granularity,
    malloc_fine_grained,
    hip_free,
    export_dmabuf_handle,
    import_dmabuf_handle,
    destroy_external_memory,
)
from ..fd_passing import send_fd, recv_fd, managed_fd


class VMemAllocator(BaseAllocator):
    """
    Fine-grained memory allocator for Iris symmetric heap.

    Uses hipExtMallocWithFlags with hipDeviceMallocFinegrained for all physical
    memory, which ensures correct P2P atomic operations (scope=cta/gpu) across GPUs.

    hipMemCreate (used in the previous VMem approach) creates coarse-grained memory
    that causes intermittent failures for cross-GPU atomics. This allocator fixes
    that by using fine-grained memory throughout.

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
        # Keep granularity for alignment and compatibility with existing tests
        self.granularity = get_allocation_granularity(self.device_id)
        self.aligned_heap_size = (heap_size + self.granularity - 1) & ~(self.granularity - 1)

        # Allocate the entire heap upfront as a single fine-grained block.
        # Fine-grained (hipExtMallocWithFlags / hipDeviceMallocFinegrained) memory
        # is required for correct cross-GPU atomic operations.
        self._alloc_ptr = malloc_fine_grained(self.aligned_heap_size)
        self.base_va = self._alloc_ptr.value

        self._peer_ext_mem_handles: Dict[int, object] = {}
        self.heap_bases_array = None

    def get_base_address(self) -> int:
        """Get the base address of the heap."""
        return self.base_va

    def get_minimum_allocation_size(self) -> int:
        """Minimum allocation size in bytes (one granule for alignment compatibility)."""
        return self.granularity

    def allocate(self, num_elements: int, dtype: torch.dtype, alignment: int = 1024) -> torch.Tensor:
        """
        Allocate a tensor from the fine-grained heap using bump allocation.

        Args:
            num_elements: Number of elements to allocate
            dtype: PyTorch data type
            alignment: Alignment requirement in bytes

        Returns:
            PyTorch tensor wrapping the allocated memory

        Raises:
            RuntimeError: If allocation fails or heap is full
        """
        with self.lock:
            element_size = torch.tensor([], dtype=dtype).element_size()
            size_in_bytes = num_elements * element_size
            aligned_size = math.ceil(size_in_bytes / alignment) * alignment

            if self.heap_offset + aligned_size > self.aligned_heap_size:
                raise RuntimeError(
                    f"Out of VMem heap space for allocation: "
                    f"need {aligned_size} bytes at offset {self.heap_offset}, "
                    f"but heap size is {self.aligned_heap_size}. "
                    f"available: {self.aligned_heap_size - self.heap_offset} bytes"
                )

            start = self.heap_offset
            self.heap_offset += aligned_size

            target_va = self.base_va + start

            class CUDAArrayInterface:
                def __init__(self, ptr, size_bytes, device):
                    self.ptr = ptr
                    self.size_bytes = size_bytes
                    self.device = device

                @property
                def __cuda_array_interface__(self):
                    return {
                        "shape": (self.size_bytes,),
                        "typestr": "|u1",
                        "data": (self.ptr, False),
                        "version": 3,
                    }

            cuda_array = CUDAArrayInterface(target_va, size_in_bytes, self.device)
            tensor_bytes = torch.as_tensor(cuda_array, device=self.device)
            full = tensor_bytes.view(dtype)
            if num_elements == 0:
                return full.narrow(0, 1, 0)
            return full.narrow(0, 0, num_elements)

    def get_shareable_handle(self) -> tuple:
        """
        Get a shareable DMA-BUF handle for the heap.

        Returns:
            tuple: (fd, base_ptr, base_size) from export_dmabuf_handle
        """
        return export_dmabuf_handle(self.base_va, self.aligned_heap_size)

    def establish_peer_access(self, all_bases: Dict[int, int], connections: Optional[Dict] = None):
        """
        Establish fine-grained access to peer memory for symmetric addressing.

        Uses hipImportExternalMemory (import_dmabuf_handle) which preserves the
        fine-grained memory type, ensuring correct cross-GPU atomic operations.

        Args:
            all_bases: Dictionary mapping rank -> base address
            connections: Optional peer connections for handle exchange
        """
        import numpy as np

        heap_bases_array = np.zeros(self.num_ranks, dtype=np.uint64)

        if connections is not None:
            for handle in self._peer_ext_mem_handles.values():
                try:
                    destroy_external_memory(handle)
                except Exception:
                    pass
            self._peer_ext_mem_handles.clear()

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
                        peer_handle, peer_metadata = recv_fd(sock, payload_size=24)
                    else:
                        peer_handle, peer_metadata = recv_fd(sock, payload_size=24)
                        send_fd(sock, my_fd, payload=my_metadata)

                    peer_base, peer_size, peer_heap = struct.unpack("QQQ", peer_metadata)

                    with managed_fd(peer_handle):
                        mapped_ptr, ext_mem_handle = import_dmabuf_handle(peer_handle, peer_size, peer_heap, peer_base)
                        heap_bases_array[peer] = mapped_ptr
                        self._peer_ext_mem_handles[peer] = ext_mem_handle

            heap_bases_array[self.cur_rank] = all_bases[self.cur_rank]
        else:
            heap_bases_array[self.cur_rank] = all_bases[self.cur_rank]

        self.heap_bases_array = heap_bases_array

    def get_device(self) -> torch.device:
        """
        Get the PyTorch device for this allocator.

        Returns:
            PyTorch device object
        """
        return self.device

    def owns_tensor(self, tensor: torch.Tensor) -> bool:
        """
        Check if a tensor's memory belongs to this allocator's heap.

        Args:
            tensor: Tensor to check

        Returns:
            True if tensor is within this allocator's heap, False otherwise
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

        Allocates space on the fine-grained symmetric heap and copies the data
        from the external tensor. The returned tensor resides on the symmetric heap
        and can be used in RMA operations across ranks.

        Note: Unlike the previous VMem implementation, the returned tensor does not
        share physical memory with the original. Modifications to one are not visible
        in the other. This is the same semantics as TorchAllocator.

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
        """Explicitly release fine-grained memory resources."""
        if hasattr(self, "_closed") and self._closed:
            return

        for handle in self._peer_ext_mem_handles.values():
            try:
                destroy_external_memory(handle)
            except Exception:
                pass
        self._peer_ext_mem_handles.clear()

        if hasattr(self, "_alloc_ptr") and self._alloc_ptr is not None:
            hip_free(self._alloc_ptr)
            self._alloc_ptr = None
            self.base_va = 0

        self._closed = True

    def __del__(self):
        """Cleanup fine-grained memory resources on deletion."""
        self.close()
