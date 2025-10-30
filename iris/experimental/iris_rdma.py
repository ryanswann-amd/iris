# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Iris RDMA: Experimental InfiniBand RDMA Backend for Multi-Node Communication

This module provides InfiniBand RDMA support for multi-node communication in Iris.
Unlike the main Iris which uses HIP IPC for intra-node GPU communication, this backend
enables inter-node communication via RDMA over InfiniBand.

Key Features:
- InfiniBand Queue Pair (QP) setup and management
- Symmetric heap with RDMA memory registration
- RDMA put/get operations in Triton kernels
- PyTorch Distributed integration for bootstrapping

Example:
    >>> import iris.experimental.iris_rdma as iris_rdma
    >>> import torch.distributed as dist
    >>> 
    >>> # Initialize PyTorch Distributed
    >>> dist.init_process_group(backend='nccl')
    >>> 
    >>> # Create RDMA context
    >>> ctx = iris_rdma.iris(heap_size=2**30)  # 1GB heap
    >>> device_ctx = ctx.get_device_context()  # For passing to Triton kernels
    >>> 
    >>> @triton.jit
    >>> def kernel(dst_ptr, data, device_ctx, dst_rank, BLOCK_SIZE: tl.constexpr):
    >>>     pid = tl.program_id(0)
    >>>     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    >>>     
    >>>     # RDMA put to remote rank
    >>>     iris_rdma.put(dst_ptr + offsets, data, dst_rank, device_ctx)
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import numpy as np
import sys
import os

# Import the C++ backend module
try:
    from . import _iris_rdma_backend as backend
except ImportError:
    raise ImportError(
        "Iris RDMA backend not available. "
        "Make sure the module is built with InfiniBand support. "
        "Set IRIS_RDMA_DEBUG=1 for more information."
    )

# Import logging
from ..logging import logger


class IrisRDMA:
    """
    Main Iris RDMA class for multi-node RDMA operations.
    
    This class provides a unified interface for RDMA-based communication
    across multiple nodes using InfiniBand.
    
    Args:
        heap_size (int): Size of the symmetric heap in bytes. Default: 1GB (2^30)
        process_group: PyTorch distributed process group (default: WORLD)
        device_name (str): InfiniBand device name (default: auto-detect)
    
    Example:
        >>> ctx = iris_rdma.iris(heap_size=2**31)  # 2GB heap
        >>> print(f"Rank {ctx.rank} of {ctx.world_size}")
        >>> buffer = ctx.zeros(1024, dtype=torch.float32)
    """
    
    def __init__(self, heap_size=1 << 30, process_group=None, device_name=None):
        # Check if distributed is initialized
        if not dist.is_initialized():
            raise RuntimeError(
                "PyTorch distributed must be initialized. "
                "Call torch.distributed.init_process_group() first."
            )
        
        if process_group is None:
            process_group = dist.group.WORLD
        
        # Get rank and world size
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device_id = self.rank % torch.cuda.device_count()
        self.device = f"cuda:{self.device_id}"
        
        torch.cuda.set_device(self.device_id)
        
        # Create TorchBootstrap
        self._bootstrap = backend.TorchBootstrap(process_group)
        
        # Create NetworkBackend
        self._backend = backend.NetworkBackend(self._bootstrap, device_name)
        
        # Initialize network (create QPs, transition to RTS)
        self._backend.init()
        
        # Allocate symmetric heap (CPU pinned memory for now)
        # TODO: Support GPU memory with GPUDirect RDMA
        self.heap_size = heap_size
        self.heap_offset = 0
        self.alignment = 1024
        
        # Create CPU pinned memory pool
        # For GPU memory, use: torch.empty(heap_size, device=self.device, dtype=torch.int8)
        self.memory_pool = torch.empty(heap_size, device='cpu', dtype=torch.int8).pin_memory()
        
        # Register memory with RDMA
        self._backend.register_memory(self.memory_pool)
        
        # Store remote heap bases (already exchanged in register_memory)
        self.remote_heap_bases = []
        for i in range(self.world_size):
            self.remote_heap_bases.append(self._backend.get_remote_heap_base(i))
        
        logger.info(f"[Rank {self.rank}] Iris RDMA initialized: heap_size={heap_size}, "
                   f"heap_base={self._backend.get_heap_base():#x}")
    
    def get_device_context(self):
        """
        Get device context tensor for passing to Triton kernels.
        
        The context tensor encodes:
        - [0]: current rank
        - [1]: world size
        - [2:]: heap base addresses for all ranks
        
        Returns:
            torch.Tensor: Device context tensor (on GPU)
        
        Example:
            >>> ctx = iris_rdma.iris()
            >>> device_ctx = ctx.get_device_context()
            >>> # Pass device_ctx to Triton kernel
        """
        # Create context tensor: [rank, world_size, heap_base_0, heap_base_1, ...]
        context_size = 2 + self.world_size
        context = torch.zeros(context_size, dtype=torch.int64, device=self.device)
        
        context[0] = self.rank
        context[1] = self.world_size
        
        for i in range(self.world_size):
            context[2 + i] = self.remote_heap_bases[i]
        
        return context
    
    def zeros(self, *size, dtype=torch.float32, device=None):
        """
        Allocate and initialize a tensor with zeros in the symmetric heap.
        
        Args:
            *size: Tensor dimensions
            dtype: Data type (default: torch.float32)
            device: Device placement ('cpu' or 'cuda', default: match context)
        
        Returns:
            torch.Tensor: Allocated tensor
        
        Example:
            >>> buffer = ctx.zeros(1024, 1024, dtype=torch.float32)
        """
        if device is None:
            device = 'cpu'  # Use CPU for now (pinned memory)
        
        # Calculate size in bytes
        elem_size = torch.tensor([], dtype=dtype).element_size()
        numel = int(np.prod(size))
        size_bytes = numel * elem_size
        
        # Align allocation
        aligned_offset = (self.heap_offset + self.alignment - 1) // self.alignment * self.alignment
        
        if aligned_offset + size_bytes > self.heap_size:
            raise RuntimeError(f"Heap exhausted: requested {size_bytes} bytes, "
                             f"available {self.heap_size - aligned_offset}")
        
        # Create tensor view into memory pool
        byte_offset = aligned_offset
        byte_end = byte_offset + size_bytes
        
        # Get the memory slice and view as the requested dtype
        memory_slice = self.memory_pool[byte_offset:byte_end]
        tensor = memory_slice.view(dtype).reshape(size)
        
        # Zero initialize
        tensor.zero_()
        
        # Update offset
        self.heap_offset = byte_end
        
        logger.debug(f"[Rank {self.rank}] Allocated tensor: size={size}, "
                    f"offset={byte_offset:#x}, ptr={tensor.data_ptr():#x}")
        
        return tensor
    
    def barrier(self):
        """
        Synchronize all ranks.
        
        Example:
            >>> ctx.barrier()  # Wait for all ranks
        """
        dist.barrier()
    
    def rdma_put(self, dst_rank, local_addr, remote_addr, size):
        """
        Perform RDMA write (put) to remote rank.
        
        Args:
            dst_rank: Destination rank
            local_addr: Local buffer address (int or tensor.data_ptr())
            remote_addr: Remote buffer address (int)
            size: Size in bytes
        
        Returns:
            int: 0 on success, non-zero on error
        
        Example:
            >>> src = ctx.zeros(1024, dtype=torch.float32)
            >>> dst_addr = ctx.remote_heap_bases[1]  # Remote rank 1's heap
            >>> ctx.rdma_put(1, src.data_ptr(), dst_addr, src.numel() * 4)
        """
        if isinstance(local_addr, torch.Tensor):
            local_addr = local_addr.data_ptr()
        
        return self._backend.rdma_write(dst_rank, local_addr, remote_addr, size)
    
    def rdma_get(self, dst_rank, local_addr, remote_addr, size):
        """
        Perform RDMA read (get) from remote rank.
        
        Args:
            dst_rank: Source rank (destination of the QP)
            local_addr: Local buffer address (int or tensor.data_ptr())
            remote_addr: Remote buffer address (int)
            size: Size in bytes
        
        Returns:
            int: 0 on success, non-zero on error
        
        Example:
            >>> dst = ctx.zeros(1024, dtype=torch.float32)
            >>> src_addr = ctx.remote_heap_bases[1]  # Remote rank 1's heap
            >>> ctx.rdma_get(1, dst.data_ptr(), src_addr, dst.numel() * 4)
        """
        if isinstance(local_addr, torch.Tensor):
            local_addr = local_addr.data_ptr()
        
        return self._backend.rdma_read(dst_rank, local_addr, remote_addr, size)
    
    def poll_completion(self, dst_rank, max_completions=1):
        """
        Poll completion queue for RDMA operations.
        
        Args:
            dst_rank: Destination rank (to poll specific CQ)
            max_completions: Maximum number of completions to poll
        
        Returns:
            int: Number of completions polled (negative on error)
        
        Example:
            >>> ctx.rdma_put(1, src.data_ptr(), remote_addr, size)
            >>> while ctx.poll_completion(1) == 0:
            >>>     pass  # Wait for completion
        """
        return self._backend.poll_cq(dst_rank, max_completions)
    
    def __repr__(self):
        return f"<IrisRDMA rank={self.rank} world_size={self.world_size}>"


def iris(heap_size=1 << 30, process_group=None, device_name=None):
    """
    Factory function to create Iris RDMA context.
    
    Args:
        heap_size (int): Size of the symmetric heap in bytes
        process_group: PyTorch distributed process group
        device_name (str): InfiniBand device name (optional)
    
    Returns:
        IrisRDMA: RDMA context object
    
    Example:
        >>> import iris.experimental.iris_rdma as iris_rdma
        >>> ctx = iris_rdma.iris(heap_size=2**30)
    """
    return IrisRDMA(heap_size, process_group, device_name)


#############################################################################
# Triton Device-Side APIs
#############################################################################

@triton.jit
def put(dst_ptr, data, dst_rank: tl.constexpr, device_ctx, mask):
    """
    RDMA put (write) operation from Triton kernel.
    
    Writes data to remote rank's memory via RDMA.
    
    Args:
        dst_ptr: Destination pointer (remote address) - can be block of pointers
        data: Data values to write (block)
        dst_rank: Target rank ID (must be compile-time constant)
        device_ctx: Device context from iris_rdma.get_device_context()
        mask: Triton mask for valid elements
    
    Example:
        >>> @triton.jit
        >>> def kernel(dst_ptr, src_ptr, device_ctx, dst_rank, BLOCK_SIZE: tl.constexpr):
        >>>     pid = tl.program_id(0)
        >>>     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        >>>     mask = offsets < n_elements
        >>>     
        >>>     data = tl.load(src_ptr + offsets, mask=mask)
        >>>     iris_rdma.put(dst_ptr + offsets, data, dst_rank, device_ctx, mask)
    """
    # Extract heap bases from device context
    # Context format: [rank, world_size, heap_base_0, heap_base_1, ...]
    dst_heap_base = tl.load(device_ctx + 2 + dst_rank)
    
    # For now, use tl.store as placeholder
    # TODO: Implement actual RDMA put via queue or direct posting
    # This will require either:
    # 1. A device-side queue that CPU polls (like iris-rdma prototype)
    # 2. Or direct ibv_post_send from GPU (requires GPU Direct Async)
    
    # Translate pointer to remote address space
    # dst_ptr should already be in the remote address space
    # Just store for now - in full implementation, this would queue RDMA request
    tl.store(dst_ptr, data, mask=mask)


@triton.jit
def get(src_ptr, from_rank: tl.constexpr, device_ctx, mask):
    """
    RDMA get (read) operation from Triton kernel.
    
    Reads data from remote rank's memory via RDMA.
    
    Args:
        src_ptr: Source pointer (remote address) - can be block of pointers
        from_rank: Source rank ID (must be compile-time constant)
        device_ctx: Device context from iris_rdma.get_device_context()
        mask: Triton mask for valid elements
    
    Returns:
        Block of data read from remote rank
    
    Example:
        >>> @triton.jit
        >>> def kernel(dst_ptr, src_ptr, device_ctx, from_rank, BLOCK_SIZE: tl.constexpr):
        >>>     pid = tl.program_id(0)
        >>>     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        >>>     mask = offsets < n_elements
        >>>     
        >>>     data = iris_rdma.get(src_ptr + offsets, from_rank, device_ctx, mask)
        >>>     tl.store(dst_ptr + offsets, data, mask=mask)
    """
    # Extract heap bases from device context
    src_heap_base = tl.load(device_ctx + 2 + from_rank)
    
    # For now, use tl.load as placeholder
    # TODO: Implement actual RDMA get via queue or direct posting
    data = tl.load(src_ptr, mask=mask)
    
    return data


__all__ = [
    "IrisRDMA",
    "iris",
    "put",
    "get",
]

