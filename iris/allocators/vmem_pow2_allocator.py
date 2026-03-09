# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Power-of-two VMem allocator using HIP's virtual memory management APIs.

This allocator provides efficient reuse of virtual memory allocations by
rounding all requests up to the next power-of-two size class and maintaining
per-class free lists for O(1) allocation and deallocation.
"""

import os
from typing import Dict, List, Tuple
from threading import Lock

import torch

from .base import BaseAllocator
from ..hip import (
    get_allocation_granularity,
    get_address_range,
    export_dmabuf_handle,
    mem_import_from_shareable_handle,
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


def _next_pow2(n: int) -> int:
    """Round n up to the next power of two (>= 1)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


class VMemPow2Allocator(BaseAllocator):
    """
    Power-of-two virtual memory allocator using HIP's VMem APIs.

    All allocation requests are rounded up to the nearest power-of-two size
    class.  Freed blocks are returned to per-class free lists and reused by
    subsequent allocations of the same (or smaller) size class, giving O(1)
    amortised alloc and free.

    Physical memory is **never unmapped** when a block is freed; only its
    logical ownership changes.  This keeps ``get_allocation_segments()``
    correct for the symmetric-heap multi-rank DMA-BUF exchange: every segment
    that has ever been allocated is still physically present at the same VA
    offset, so peer ranks can import it once and it stays valid.

    Args:
        heap_size:    Total virtual address space to reserve, in bytes.
        device_id:    HIP/CUDA device index.
        rank:         Current process rank.
        world_size:   Total number of ranks.
        va_multiplier: Reserved for future use (currently unused).
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
        self.lock = Lock()

        # HIP allocation granularity (always a power of two, e.g. 2 MiB on MI300X).
        self.granularity = get_allocation_granularity(self.device_id)

        # The minimum size class is the granularity itself.
        # Because granularity is always a power of two this doubles as min_size_class.
        self.min_size_class: int = self.granularity

        # Align the heap to the granularity.
        self.aligned_heap_size = (heap_size + self.granularity - 1) & ~(self.granularity - 1)
        self.va_size = self.aligned_heap_size
        self.base_va: int = mem_address_reserve(self.va_size, self.granularity, 0)

        # Bootstrap: map a minimal chunk at VA base so mem_set_access has
        # something to work on (hipMemCreate(0) is invalid).
        self.minimal_size: int = self.min_size_class
        self.minimal_handle = mem_create(self.minimal_size, self.device_id)
        mem_map(self.base_va, self.minimal_size, 0, self.minimal_handle)

        # Access descriptors: allow read/write from every peer device.
        self.access_descs: List[hipMemAccessDesc] = []
        for peer_device_id in range(world_size):
            desc = hipMemAccessDesc()
            desc.location.type = hipMemLocationTypeDevice
            desc.location.id = peer_device_id
            desc.flags = hipMemAccessFlagsProtReadWrite
            self.access_descs.append(desc)

        self.cumulative_mapped_size: int = self.minimal_size
        mem_set_access(self.base_va, self.cumulative_mapped_size, self.access_descs)

        # Physical-segment tracking (for get_allocation_segments / cleanup).
        # Maps VA-offset -> (size, is_imported, handle, va).
        self.allocations: Dict[int, Tuple] = {}
        # Ordered list of (offset, size) for get_allocation_segments().
        self.allocation_order: List[Tuple[int, int]] = []
        self._track_allocation(0, self.minimal_size, False, self.minimal_handle, self.base_va)

        # Next available VA offset for a brand-new physical segment.
        self.current_offset: int = self.minimal_size

        # Free lists: size_class (power-of-two bytes) -> [(offset, va), …]
        self.free_lists: Dict[int, List[Tuple[int, int]]] = {}

        # Logical-allocation tracking: va -> size_class (needed by free()).
        self.logical_allocations: Dict[int, int] = {}

        self.world_size = world_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _track_allocation(self, offset: int, size: int, is_imported: bool, handle, va: int):
        """Record a physical segment for cleanup and segmented DMA-BUF export."""
        self.allocations[offset] = (size, is_imported, handle, va)
        self.allocation_order.append((offset, size))

    def _size_class(self, size_bytes: int) -> int:
        """Return the smallest power-of-two size class >= size_bytes and >= granularity."""
        raw = _next_pow2(max(size_bytes, 1))
        return max(raw, self.min_size_class)

    def _map_new_segment(self, size_class: int) -> Tuple[int, int]:
        """
        Map a fresh physical segment of ``size_class`` bytes at the next
        available VA offset.

        The VA offset is aligned to the HIP allocation granularity (not to
        ``size_class``) so that consecutive segments occupy contiguous VA
        ranges.  Contiguous mapping is required because HIP's
        ``hipMemSetAccess`` must be called cumulatively from ``base_va`` and
        treats any unmapped gap as an invalid argument.

        Returns:
            (offset, va) of the newly mapped segment.

        Raises:
            RuntimeError: If the heap VA space is exhausted.
        """
        # Align to granularity (not size_class) to keep the VA range gap-free.
        aligned_offset = (self.current_offset + self.granularity - 1) & ~(self.granularity - 1)

        if aligned_offset + size_class > self.aligned_heap_size:
            raise RuntimeError(
                f"VMemPow2Allocator: out of VA space. "
                f"Need {size_class} bytes at offset {aligned_offset}, "
                f"heap size is {self.aligned_heap_size}, "
                f"current offset is {self.current_offset}."
            )

        va = self.base_va + aligned_offset
        handle = mem_create(size_class, self.device_id)
        mem_map(va, size_class, 0, handle)

        new_cumulative = aligned_offset + size_class
        if new_cumulative > self.cumulative_mapped_size:
            self.cumulative_mapped_size = new_cumulative
            mem_set_access(self.base_va, self.cumulative_mapped_size, self.access_descs)

        self._track_allocation(aligned_offset, size_class, False, handle, va)
        self.current_offset = aligned_offset + size_class
        return aligned_offset, va

    # ------------------------------------------------------------------
    # BaseAllocator interface
    # ------------------------------------------------------------------

    def get_base_address(self) -> int:
        """Return the base virtual address of this allocator's VA range."""
        return self.base_va

    def get_minimum_allocation_size(self) -> int:
        """Minimum allocation size in bytes (one size-class / granule)."""
        return self.granularity

    def allocate(self, num_elements: int, dtype: torch.dtype, alignment: int = 1024) -> torch.Tensor:
        """
        Allocate a tensor on the power-of-two symmetric heap.

        The physical size is rounded up to the next power-of-two size class
        (and is at least ``granularity`` bytes).  If a block of the required
        size class is already on the free list it is reused; otherwise a new
        physical segment is mapped.

        Args:
            num_elements: Number of tensor elements.
            dtype:        PyTorch data type.
            alignment:    Ignored for this allocator (alignment is provided by
                          the power-of-two size class itself).

        Returns:
            A PyTorch tensor of shape ``(num_elements,)`` backed by symmetric
            heap memory.

        Raises:
            RuntimeError: If the heap VA space is exhausted.
        """
        with self.lock:
            element_size = torch.tensor([], dtype=dtype).element_size()
            size_bytes = num_elements * element_size
            size_class = self._size_class(size_bytes)

            # Try the free list first.
            free_entry = None
            if size_class in self.free_lists and self.free_lists[size_class]:
                free_entry = self.free_lists[size_class].pop()

            if free_entry is not None:
                offset, va = free_entry
            else:
                offset, va = self._map_new_segment(size_class)

            # Record the logical allocation so free() can find the size class.
            self.logical_allocations[va] = size_class

            # Expose the physical memory as a PyTorch tensor via __cuda_array_interface__.
            interface_size = (size_class // element_size) * element_size

            class _CUDAArrayInterface:
                def __init__(self_, ptr: int, nbytes: int, device: torch.device):
                    self_.ptr = ptr
                    self_.nbytes = nbytes
                    self_.device = device

                @property
                def __cuda_array_interface__(self_):
                    return {
                        "shape": (self_.nbytes,),
                        "typestr": "|u1",
                        "data": (self_.ptr, False),
                        "version": 3,
                    }

            cuda_array = _CUDAArrayInterface(va, interface_size, self.device)
            tensor_bytes = torch.as_tensor(cuda_array, device=self.device)
            full = tensor_bytes.view(dtype)
            if num_elements == 0:
                return full.narrow(0, 1, 0)
            return full.narrow(0, 0, num_elements)

    def free(self, tensor: torch.Tensor) -> None:
        """
        Return a tensor's physical block to the appropriate free list.

        The physical memory is **not** unmapped; it remains accessible at its
        VA so that peer-rank DMA-BUF imports stay valid.  The block is simply
        made available for reuse by the next ``allocate`` call of the same
        size class.

        Args:
            tensor: A tensor previously returned by :meth:`allocate`.

        Raises:
            ValueError: If the tensor was not allocated by this allocator.
        """
        if tensor.numel() == 0:
            # Zero-element tensors share the minimal bootstrap block; skip.
            return

        with self.lock:
            va = tensor.data_ptr()
            if va not in self.logical_allocations:
                raise ValueError(
                    f"VMemPow2Allocator.free(): tensor at VA 0x{va:x} was not "
                    "allocated by this allocator (or was already freed)."
                )
            size_class = self.logical_allocations.pop(va)
            offset = va - self.base_va
            self.free_lists.setdefault(size_class, []).append((offset, va))

    def get_device(self) -> torch.device:
        """Return the PyTorch device for this allocator."""
        return self.device

    def owns_tensor(self, tensor: torch.Tensor) -> bool:
        """
        Return True if *tensor* was allocated from this allocator's heap.

        Args:
            tensor: PyTorch tensor to check.

        Returns:
            True if the tensor's data pointer lies within the heap VA range.
        """
        if not tensor.is_cuda:
            return False
        if tensor.numel() == 0:
            return True
        ptr = tensor.data_ptr()
        return self.base_va <= ptr < self.base_va + self.aligned_heap_size

    # ------------------------------------------------------------------
    # Symmetric-heap segment API (used by SymmetricHeap.refresh_peer_access)
    # ------------------------------------------------------------------

    def get_allocation_segments(self) -> List[Tuple[int, int, int]]:
        """
        Return the ordered list of physical segments for DMA-BUF export.

        Each element is ``(offset, size, va)`` describing one physically-backed
        segment that must be exported and imported across ranks.  Segments on
        the free list are included because they are still physically mapped and
        their peer imports must remain valid.

        Returns:
            List of ``(offset, size, va)`` tuples in allocation order.
        """
        segments = []
        for offset, size in self.allocation_order:
            va = self.base_va + offset
            segments.append((offset, size, va))
        return segments

    # ------------------------------------------------------------------
    # as_symmetric() support
    # ------------------------------------------------------------------

    def import_external_tensor(self, external_tensor: torch.Tensor) -> torch.Tensor:
        """
        Import an external PyTorch tensor into the symmetric heap.

        This remaps the external tensor's physical memory into the symmetric
        heap VA range so that peer ranks can access it via the standard
        DMA-BUF exchange.  The returned tensor **shares physical memory** with
        the original; changes to one are immediately visible in the other.

        Args:
            external_tensor: A contiguous CUDA tensor allocated by PyTorch.

        Returns:
            A tensor view in the symmetric heap that shares memory with
            *external_tensor*.

        Raises:
            RuntimeError: If the tensor is not a contiguous CUDA tensor, or
                          if the heap VA space is exhausted.
        """
        with self.lock:
            if not external_tensor.is_cuda:
                raise RuntimeError("VMemPow2Allocator: can only import CUDA tensors.")
            if not external_tensor.is_contiguous():
                raise RuntimeError(
                    "VMemPow2Allocator: only contiguous tensors can be imported; "
                    "call .contiguous() before as_symmetric()."
                )

            external_ptr = external_tensor.data_ptr()
            alloc_base, alloc_size = get_address_range(external_ptr)
            offset_in_alloc = external_ptr - alloc_base
            aligned_size = (alloc_size + self.granularity - 1) & ~(self.granularity - 1)
            aligned_offset = (self.current_offset + self.granularity - 1) & ~(self.granularity - 1)

            if aligned_offset + aligned_size > self.aligned_heap_size:
                raise RuntimeError(
                    f"VMemPow2Allocator: out of VA space for import. "
                    f"Need {aligned_size} bytes at offset {aligned_offset}, "
                    f"heap size is {self.aligned_heap_size}."
                )

            dmabuf_fd, export_base, export_size = export_dmabuf_handle(alloc_base, alloc_size)
            aligned_export_size = (export_size + self.granularity - 1) & ~(self.granularity - 1)
            target_va = self.base_va + aligned_offset
            imported_handle = mem_import_from_shareable_handle(dmabuf_fd)
            os.close(dmabuf_fd)

            mem_map(target_va, aligned_export_size, 0, imported_handle)

            new_cumulative = aligned_offset + aligned_export_size
            if new_cumulative > self.cumulative_mapped_size:
                self.cumulative_mapped_size = new_cumulative
                mem_set_access(self.base_va, self.cumulative_mapped_size, self.access_descs)

            tensor_va = target_va + offset_in_alloc
            self._track_allocation(aligned_offset, aligned_export_size, True, imported_handle, target_va)
            self.current_offset = aligned_offset + aligned_export_size

            tensor_size = external_tensor.numel() * external_tensor.element_size()

            class _CUDAArrayInterface:
                def __init__(self_, ptr: int, nbytes: int, device: torch.device):
                    self_.ptr = ptr
                    self_.nbytes = nbytes
                    self_.device = device

                @property
                def __cuda_array_interface__(self_):
                    return {
                        "shape": (self_.nbytes,),
                        "typestr": "|u1",
                        "data": (self_.ptr, False),
                        "version": 3,
                    }

            cuda_array = _CUDAArrayInterface(tensor_va, tensor_size, self.device)
            tensor_bytes = torch.as_tensor(cuda_array, device=self.device)
            return tensor_bytes.view(external_tensor.dtype).reshape(external_tensor.shape)

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release all VMem resources (unmap and free physical handles)."""
        if getattr(self, "_closed", False):
            return

        with self.lock:
            for offset, alloc_info in self.allocations.items():
                if len(alloc_info) == 4:
                    size, is_imported, handle, va = alloc_info
                    if handle is not None:
                        aligned_size = (size + self.granularity - 1) & ~(self.granularity - 1)
                        mem_unmap(va, aligned_size)
                        mem_release(handle)

            self.allocations.clear()
            self.free_lists.clear()
            self.logical_allocations.clear()

            if getattr(self, "base_va", 0):
                mem_address_free(self.base_va, self.va_size)
                self.base_va = 0

            self._closed = True

    def __del__(self) -> None:
        """Cleanup VMem resources on garbage collection."""
        self.close()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """
        Return a snapshot of allocator statistics.

        Returns:
            A dict with keys:

            * ``heap_size``            – requested heap size in bytes
            * ``aligned_heap_size``    – actual VA reservation in bytes
            * ``granularity``          – HIP allocation granularity in bytes
            * ``current_offset``       – bytes consumed from VA space
            * ``num_segments``         – number of physical segments ever mapped
            * ``num_live_allocations`` – logical allocations currently in use
            * ``free_list_counts``     – dict of {size_class: count} for free lists
        """
        with self.lock:
            return {
                "heap_size": self.heap_size,
                "aligned_heap_size": self.aligned_heap_size,
                "granularity": self.granularity,
                "current_offset": self.current_offset,
                "num_segments": len(self.allocation_order),
                "num_live_allocations": len(self.logical_allocations),
                "free_list_counts": {sc: len(bl) for sc, bl in self.free_lists.items() if bl},
            }
