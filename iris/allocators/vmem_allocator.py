# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
VMem-based allocator using HIP's virtual memory management APIs.

This allocator provides fine-grained control over virtual and physical memory,
enabling features like memory oversubscription and on-demand paging.
"""

import torch
from typing import Dict
from threading import Lock

from .base import BaseAllocator
from ..hip import (
    get_allocation_granularity,
    get_address_range,
    export_dmabuf_handle,
    import_dmabuf_handle,
    destroy_external_memory,
    mem_create,
    mem_address_reserve,
    mem_map,
    mem_unmap,
    mem_address_free,
    mem_release,
    mem_set_access,
    hipMemAccessDesc,
    hipMemLocationTypeDevice,
    hipMemAccessFlagsProtReadWrite,
)


class VMemAllocator(BaseAllocator):
    """
    Virtual Memory allocator using HIP's VMem APIs.

    Features:
    - Reserve large virtual address (VA) space upfront
    - Map physical memory on demand
    - Support memory oversubscription
    - Fine-grained control over allocations

    Args:
        heap_size: Total size of the heap in bytes
        device: PyTorch device (e.g., "cuda:0")
        rank: Current rank ID
        world_size: Total number of ranks
        va_multiplier: VA space multiplier (reserve more VA than physical)
    """

    def __init__(
        self,
        heap_size: int,
        device_id: int,
        rank: int,
        world_size: int,
        va_multiplier: float = 1.0,
    ):
        super().__init__(heap_size, device_id, rank, world_size)
        self.va_multiplier = va_multiplier
        self.device = torch.device(f"cuda:{device_id}")

        # Thread safety
        self.lock = Lock()

        # Get allocation granularity
        self.granularity = get_allocation_granularity(self.device_id)

        # Align heap size to granularity
        self.aligned_heap_size = (heap_size + self.granularity - 1) & ~(self.granularity - 1)

        # Reserve VA space (use aligned heap size for now, multiplier for future oversubscription)
        self.va_size = self.aligned_heap_size

        # Create physical memory allocation
        self.local_handle = mem_create(self.aligned_heap_size, self.device_id)

        # Reserve VA space
        self.base_va = mem_address_reserve(self.va_size)

        # Map local physical memory to VA
        mem_map(self.base_va, self.aligned_heap_size, 0, self.local_handle)

        # CRITICAL: Track cumulative allocated size for hipMemSetAccess workaround
        # ROCm bug: must call hipMemSetAccess from base_va with cumulative size
        # See: https://github.com/ROCm/rocm-systems/issues/2667
        self.cumulative_allocated = self.aligned_heap_size

        # Set access permissions for current device (initial mapping)
        access_desc = hipMemAccessDesc()
        access_desc.location.type = hipMemLocationTypeDevice
        access_desc.location.id = self.device_id
        access_desc.flags = hipMemAccessFlagsProtReadWrite
        mem_set_access(self.base_va, self.cumulative_allocated, access_desc)

        # Track allocations: offset -> (size, is_imported, external_ptr)
        self.allocations: Dict[int, tuple] = {}
        self.current_offset = 0

    def get_base_address(self) -> int:
        """Get the base address of the heap."""
        return self.base_va

    def allocate(self, num_elements: int, dtype: torch.dtype, alignment: int = 1024) -> torch.Tensor:
        """
        Allocate memory from the VMem heap.

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
            # Calculate size in bytes
            element_size = torch.tensor([], dtype=dtype).element_size()
            size_bytes = num_elements * element_size

            # Align offset
            aligned_offset = (self.current_offset + alignment - 1) & ~(alignment - 1)

            # Check if we have enough space
            if aligned_offset + size_bytes > self.aligned_heap_size:
                raise RuntimeError(
                    f"VMem heap exhausted: requested {size_bytes} bytes at offset {aligned_offset}, "
                    f"but heap size is {self.aligned_heap_size}"
                )

            # Calculate address
            alloc_addr = self.base_va + aligned_offset

            # Track allocation (is_imported=False, external_ptr=None)
            self.allocations[aligned_offset] = (size_bytes, False, None)
            self.current_offset = aligned_offset + size_bytes

            # Create a torch tensor directly from the GPU pointer using __cuda_array_interface__
            class CUDAArrayInterface:
                def __init__(self, ptr, size_bytes, device):
                    self.ptr = ptr
                    self.size_bytes = size_bytes
                    self.device = device

                @property
                def __cuda_array_interface__(self):
                    return {
                        "shape": (self.size_bytes,),
                        "typestr": "|u1",  # uint8
                        "data": (self.ptr, False),  # (ptr, read_only)
                        "version": 3,
                    }

            cuda_array = CUDAArrayInterface(alloc_addr, size_bytes, self.device)
            tensor_bytes = torch.as_tensor(cuda_array, device=self.device)

            # Cast to correct dtype and reshape
            tensor = tensor_bytes.view(dtype)[:num_elements]

            return tensor

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

        ptr = tensor.data_ptr()

        # Check if pointer is within our local VA range only
        return self.base_va <= ptr < self.base_va + self.aligned_heap_size

    def import_external_tensor(self, external_tensor: torch.Tensor) -> torch.Tensor:
        """
        Import an external PyTorch tensor into the symmetric heap (as_symmetric).

        This creates a view into the symmetric heap that shares physical memory
        with the external tensor, handling PyTorch caching allocator offsets.

        Args:
            external_tensor: External PyTorch tensor to import

        Returns:
            New tensor view in symmetric heap that shares memory with external tensor

        Raises:
            RuntimeError: If import fails
        """

        with self.lock:
            if not external_tensor.is_cuda:
                raise RuntimeError("Can only import CUDA tensors")

            external_ptr = external_tensor.data_ptr()

            # Query the base allocation to handle PyTorch caching allocator offsets
            alloc_base, alloc_size = get_address_range(external_ptr)
            offset_in_alloc = external_ptr - alloc_base

            # Align allocation size to granularity
            aligned_size = (alloc_size + self.granularity - 1) & ~(self.granularity - 1)

            # Allocate VA space in our heap (bump allocation)
            aligned_offset = (self.current_offset + self.granularity - 1) & ~(self.granularity - 1)

            if aligned_offset + aligned_size > self.aligned_heap_size:
                raise RuntimeError(
                    f"VMem heap exhausted during import: need {aligned_size} bytes "
                    f"at offset {aligned_offset}, heap size is {self.aligned_heap_size}"
                )

            # Export external allocation as DMA-BUF (using base, not offset pointer)
            dmabuf_fd, export_base, export_size = export_dmabuf_handle(alloc_base, alloc_size)

            # Import DMA-BUF with automatic offset correction
            # This handles PyTorch caching allocator offsets correctly
            # Returns (pointer, ext_mem_handle) - we need to track the handle for cleanup
            remapped_ptr, ext_mem_handle = import_dmabuf_handle(dmabuf_fd, export_size, external_ptr, export_base)

            # Note: import_dmabuf_handle manages the FD internally, don't close it

            # Track this as an imported allocation
            # Store ext_mem_handle so we can destroy it in close()
            self.allocations[aligned_offset] = (aligned_size, True, alloc_base, ext_mem_handle)
            self.current_offset = aligned_offset + aligned_size

            # Create tensor using __cuda_array_interface__
            tensor_size = external_tensor.numel() * external_tensor.element_size()

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

            cuda_array = CUDAArrayInterface(remapped_ptr, tensor_size, self.device)
            tensor_bytes = torch.as_tensor(cuda_array, device=self.device)

            # View as original dtype and reshape
            imported_tensor = tensor_bytes.view(external_tensor.dtype).reshape(external_tensor.shape)

            return imported_tensor

    def close(self):
        """Explicitly release VMem resources."""
        if hasattr(self, "_closed") and self._closed:
            return

        with self.lock:
            # Clean up imported allocations only
            # Native allocations don't need individual cleanup - they're part of the base mapping
            for offset, alloc_info in self.allocations.items():
                # Handle both old (3-tuple) and new (4-tuple) formats
                if len(alloc_info) == 4:
                    size, is_imported, external_ptr, ext_mem_handle = alloc_info
                else:
                    size, is_imported, external_ptr = alloc_info
                    ext_mem_handle = None

                if is_imported and ext_mem_handle is not None:
                    # Imported allocation: destroy external memory handle
                    # This unmaps the imported memory
                    destroy_external_memory(ext_mem_handle)
                # Native allocations: no individual cleanup needed
                # They're sub-regions of the base mapping which we unmap below

            self.allocations.clear()

            # Unmap and free the initial local physical allocation
            if hasattr(self, "base_va") and self.base_va:
                mem_unmap(self.base_va, self.aligned_heap_size)
                mem_address_free(self.base_va, self.va_size)
                self.base_va = 0

            # Release local handle (this frees physical memory)
            if hasattr(self, "local_handle") and self.local_handle:
                mem_release(self.local_handle)
                self.local_handle = 0

            self._closed = True

    def __del__(self):
        """Cleanup VMem resources on deletion."""
        self.close()
