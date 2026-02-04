# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Symmetric heap abstraction for Iris.

Provides a high-level interface for distributed symmetric memory management,
hiding the details of allocators and inter-process memory sharing.
"""

import numpy as np
import torch
import os

from iris.allocators import TorchAllocator, VMemAllocator
from iris.fd_passing import setup_fd_infrastructure
from iris._distributed_helpers import distributed_allgather


class SymmetricHeap:
    """
    High-level symmetric heap abstraction.

    Manages distributed memory with symmetric addressing across ranks,
    handling all allocator coordination and memory sharing internally.

    Supports multiple allocator backends: 'torch' (default) and 'vmem'.
    """

    def __init__(
        self,
        heap_size: int,
        device_id: int,
        cur_rank: int,
        num_ranks: int,
        allocator_type: str = "vmem",
    ):
        """
        Initialize symmetric heap.

        Args:
            heap_size: Size of the heap in bytes
            device_id: GPU device ID
            cur_rank: Current process rank
            num_ranks: Total number of ranks
            allocator_type: Type of allocator ("torch" or "vmem")

        Raises:
            ValueError: If allocator_type is not supported
        """
        self.heap_size = heap_size
        self.device_id = device_id
        self.cur_rank = cur_rank
        self.num_ranks = num_ranks

        # Allow environment variable override
        allocator_type = os.environ.get("IRIS_ALLOCATOR", allocator_type).lower()

        # Create allocator based on type
        if allocator_type == "torch":
            self.allocator = TorchAllocator(heap_size, device_id, cur_rank, num_ranks)
        elif allocator_type == "vmem":
            self.allocator = VMemAllocator(heap_size, device_id, cur_rank, num_ranks)
        else:
            raise ValueError(
                f"Unknown allocator type: {allocator_type}. Supported: 'torch', 'vmem'"
            )

        # All-gather local heap bases
        heap_base = self.allocator.get_base_address()
        local_base_arr = np.array([heap_base], dtype=np.uint64)
        all_bases_arr = distributed_allgather(local_base_arr).reshape(num_ranks).astype(np.uint64)
        all_bases = {rank: int(all_bases_arr[rank]) for rank in range(num_ranks)}

        # Setup FD passing infrastructure
        fd_conns = setup_fd_infrastructure(cur_rank, num_ranks)

        # Exchange handles and import peer memory via DMA-BUF
        if fd_conns is not None:
            from iris.fd_passing import send_fd, recv_fd, managed_fd
            from iris.hip import export_dmabuf_handle, import_dmabuf_handle

            # Export our memory as a shareable DMA-BUF handle
            my_base = self.allocator.get_base_address()
            my_handle_fd, _, _ = export_dmabuf_handle(my_base, heap_size)

            with managed_fd(my_handle_fd):
                for peer, sock in fd_conns.items():
                    if peer == cur_rank:
                        continue

                    # Exchange FDs (higher rank sends first to avoid deadlock)
                    if cur_rank > peer:
                        send_fd(sock, my_handle_fd)
                        peer_handle, _ = recv_fd(sock)
                    else:
                        peer_handle, _ = recv_fd(sock)
                        send_fd(sock, my_handle_fd)

                    # Import peer handle via DMA-BUF (same for all allocators)
                    with managed_fd(peer_handle):
                        mapped_addr, _ = import_dmabuf_handle(peer_handle, heap_size)
                        all_bases[peer] = mapped_addr

        # Create heap_bases tensor
        device = self.allocator.get_device()
        heap_bases_array = torch.zeros(num_ranks, dtype=torch.uint64, device=device)
        for rank, base in all_bases.items():
            heap_bases_array[rank] = base
        self.heap_bases = heap_bases_array

    def allocate(self, num_elements: int, dtype: torch.dtype, alignment: int = 1024) -> torch.Tensor:
        """
        Allocate a tensor on the symmetric heap.

        Args:
            num_elements: Number of elements to allocate
            dtype: PyTorch data type
            alignment: Alignment requirement in bytes (default: 1024)

        Returns:
            Allocated tensor on the symmetric heap
        """
        return self.allocator.allocate(num_elements, dtype, alignment)

    def get_device(self) -> torch.device:
        """Get the torch device for this heap."""
        return self.allocator.get_device()

    def on_symmetric_heap(self, tensor: torch.Tensor) -> bool:
        """
        Check if a tensor is allocated on the symmetric heap.

        Args:
            tensor: PyTorch tensor to check

        Returns:
            True if tensor is on the symmetric heap, False otherwise
        """
        # Delegate to allocator
        return self.allocator.owns_tensor(tensor)

    def get_heap_bases(self) -> torch.Tensor:
        """Get heap base addresses for all ranks as a tensor."""
        return self.heap_bases
    
    def as_symmetric(self, external_tensor: torch.Tensor) -> torch.Tensor:
        """
        Import an external PyTorch tensor into the symmetric heap.
        
        This creates a new tensor in the symmetric heap that shares physical
        memory with the external tensor. Modifications to either tensor will
        be visible in both.
        
        Args:
            external_tensor: External PyTorch tensor to import
        
        Returns:
            New tensor in symmetric heap sharing memory with external tensor
        
        Raises:
            RuntimeError: If allocator doesn't support imports or import fails
        """
        # Check if allocator supports import (currently only VMem)
        if not hasattr(self.allocator, 'import_external_tensor'):
            raise RuntimeError(
                f"{type(self.allocator).__name__} does not support as_symmetric(). "
                "Use allocator_type='vmem' to enable this feature."
            )
        
        return self.allocator.import_external_tensor(external_tensor)
