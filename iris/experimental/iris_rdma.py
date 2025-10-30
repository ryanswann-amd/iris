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
    
    def __init__(self, heap_size=1 << 30, process_group=None, queue_size=512):
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
        
        # Allocate symmetric heap (CPU pinned memory for now)
        # TODO: Support GPU memory with GPUDirect RDMA
        self.heap_size = heap_size
        self.heap_offset = 0
        self.alignment = 1024
        
        # Create GPU memory pool
        self.memory_pool = torch.empty(heap_size, device=self.device, dtype=torch.int8)
        
        self._manager = backend.IrisManager(self._bootstrap, self.memory_pool, queue_size)
        self._manager.start_proxy_thread()
        
        self._backend = self._manager
        
        logger.info(f"[Rank {self.rank}] Using IrisManager with queue (size={queue_size})")
        
        self.remote_heap_bases = []
        for i in range(self.world_size):
            self.remote_heap_bases.append(self._manager.get_remote_heap_base(i))
        
        logger.info(f"[Rank {self.rank}] Iris RDMA initialized: heap_size={heap_size}, "
                   f"heap_base={self._manager.get_heap_base():#x}")
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, '_manager') and self._manager is not None:
            self._manager.stop_proxy_thread()
    
    def get_heap_base(self):
        """Get local heap base address"""
        return self._manager.get_heap_base()
    
    def get_queue_ptr(self):
        """Get queue pointer for Triton kernels"""
        return self._manager.get_queue_ptr()
    
    def get_device_context(self):
        """
        Get device context tensor for passing to Triton kernels.
        
        The context tensor encodes:
        - [0]: current rank
        - [1]: world size
        - [2]: queue pointer (for enqueueing RDMA operations)
        - [3:]: heap base addresses for all ranks
        
        Returns:
            torch.Tensor: Device context tensor (on GPU)
        
        Example:
            >>> ctx = iris_rdma.iris()
            >>> device_ctx = ctx.get_device_context()
            >>> # Pass device_ctx to Triton kernel
        """
        # Create context tensor: [rank, world_size, queue_ptr, heap_base_0, heap_base_1, ...]
        context_size = 3 + self.world_size
        context = torch.zeros(context_size, dtype=torch.int64, device=self.device)
        
        context[0] = self.rank
        context[1] = self.world_size
        context[2] = self.get_queue_ptr()
        
        for i in range(self.world_size):
            context[3 + i] = self.remote_heap_bases[i]
        
        return context
    
    def zeros(self, *size, dtype=torch.float32, device=None):
        """
        Allocate and initialize a tensor with zeros in the symmetric heap.
        
        Args:
            *size: Tensor dimensions
            dtype: Data type (default: torch.float32)
            device: Device placement (default: GPU for direct kernel access)
        
        Returns:
            torch.Tensor: Allocated tensor (on GPU by default)
        
        Example:
            >>> buffer = ctx.zeros(1024, 1024, dtype=torch.float32)
        """
        if device is None:
            device = self.device  # Use GPU by default (for GPUDirect)
        
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
        Synchronize all ranks and drain RDMA queue.
        
        Waits for:
        1. All enqueued RDMA operations to complete (queue drains)
        2. All ranks to reach this barrier
        
        Example:
            >>> ctx.barrier()  # Wait for all ranks and RDMA completion
        """
        # First, wait for queue to drain (all work processed)
        self.wait_queue_drain()
        
        # Then synchronize with other ranks
        dist.barrier()
    
    def wait_queue_drain(self, timeout=30.0):
        """
        Wait for the CPU proxy thread to process all enqueued work items.
        
        Spins until queue is empty (head == tail), meaning all work has been
        processed and popped by the CPU proxy thread.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Raises:
            TimeoutError: If queue doesn't drain within timeout
        """
        import time
        start = time.time()
        
        while time.time() - start < timeout:
            # Check if queue is empty (head == tail)
            if self._manager.is_queue_empty():
                return
            
            # Small sleep to avoid burning CPU
            time.sleep(0.0001)  # 100 microseconds
        
        raise TimeoutError(f"Queue did not drain within {timeout}s")
    
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


def iris(heap_size=1 << 30, process_group=None, queue_size=512):
    """
    Factory function to create Iris RDMA context.
    
    Args:
        heap_size (int): Size of the symmetric heap in bytes
        process_group: PyTorch distributed process group
        queue_size (int): Queue size for GPU->CPU RDMA operations
    
    Returns:
        IrisRDMA: RDMA context object
    
    Example:
        >>> import iris.experimental.iris_rdma as iris_rdma
        >>> ctx = iris_rdma.iris(heap_size=2**30)
    """
    return IrisRDMA(heap_size, process_group, queue_size)


#############################################################################
# Triton Device-Side APIs
#############################################################################

@triton.jit
def _wait_for_completion(queue_ptr, queue_pos):
    """
    Wait for CPU to process a queue item.
    
    Spins until tail pointer advances past our queue position,
    indicating the CPU has processed and popped our item.
    
    Args:
        queue_ptr: Queue context pointer
        queue_pos: Queue position to wait for (returned from _enqueue_rdma_op)
    """
    state_ptr = queue_ptr.to(tl.pointer_type(tl.uint64))
    
    # Load tail pointer (offset 2 in QueueState)
    # Use volatile and cache modifier to prevent caching
    tail_ptr = tl.load(state_ptr + 2, cache_modifier=".cv", volatile=True)
    tail_ptr_typed = tail_ptr.to(tl.pointer_type(tl.uint64))
    current_tail = tl.atomic_add(tail_ptr_typed, 0, sem='acquire', scope='sys')
    
    # Spin until CPU advances tail past our position
    while queue_pos >= current_tail:
        tail_ptr = tl.load(state_ptr + 2, cache_modifier=".cv", volatile=True)
        tail_ptr_typed = tail_ptr.to(tl.pointer_type(tl.uint64))
        current_tail = tl.atomic_add(tail_ptr_typed, 0, sem='acquire', scope='sys')


@triton.jit
def _enqueue_rdma_op(dst_ptr, src_ptr, to_rank: tl.constexpr, op_code: tl.constexpr, queue_ptr, mask):
    """
    Internal: Enqueue an RDMA operation to the queue.
    
    Args:
        dst_ptr: Destination pointer on remote rank
        src_ptr: Source pointer (local address where data is stored in registered heap)
        to_rank: Target rank ID
        op_code: Operation type (1=PUT, 2=GET)
        queue_ptr: Queue pointer from device context
        mask: Triton mask for valid elements
    """
    # Queue structure (from queue.hpp):
    # struct QueueState {
    #   WorkItem* items;      // offset 0
    #   uint64_t* head;       // offset 8
    #   uint64_t* tail;       // offset 16
    #   uint64_t* tailCache;  // offset 24
    #   int32_t size;         // offset 32
    # };
    
    state_ptr = queue_ptr.to(tl.pointer_type(tl.uint64))
    
    # Load QueueState fields
    items_ptr = tl.load(state_ptr + 0)
    head_ptr = tl.load(state_ptr + 1)
    tail_ptr = tl.load(state_ptr + 2)
    
    # Load size (at offset 32 bytes = 4 * uint64)
    size_ptr = queue_ptr.to(tl.pointer_type(tl.int32))
    size = tl.load(size_ptr + 8)
    
    # Atomic increment head to reserve slot
    head_ptr_typed = head_ptr.to(tl.pointer_type(tl.uint64))
    prev_head = tl.atomic_add(head_ptr_typed, 1, sem='relaxed', scope='sys')
    
    # Wait for slot to be free: spin if prev_head >= size + *tail
    size_u64 = size.to(tl.uint64)
    tail_ptr_typed = tail_ptr.to(tl.pointer_type(tl.uint64))
    current_tail = tl.atomic_add(tail_ptr_typed, 0, sem='acquire', scope='sys')
    
    while prev_head >= size_u64 + current_tail:
        current_tail = tl.atomic_add(tail_ptr_typed, 0, sem='acquire', scope='sys')
    
    # Calculate slot position
    slot_idx = prev_head % size_u64
    
    # WorkItem structure (32 bytes):
    # struct WorkItem {
    #   uint64_t dst_ptr;      // offset 0
    #   uint64_t src_ptr;      // offset 8
    #   uint32_t size_bytes;   // offset 16 - WRITE LAST as ready flag
    #   uint16_t rank;         // offset 20
    #   uint8_t  op_type;      // offset 22
    #   uint8_t  reserved;     // offset 23
    # };
    WORK_ITEM_SIZE_BYTES = 32
    
    slot_offset_bytes = slot_idx * WORK_ITEM_SIZE_BYTES
    
    # Get pointer to this work item
    items_ptr_u64 = items_ptr.to(tl.pointer_type(tl.uint64))
    slot_ptr_u64 = items_ptr_u64 + (slot_offset_bytes // 8).to(tl.int32)
    
    # Extract destination address (min of pointer block)
    dst_ptr_u64 = dst_ptr.to(tl.uint64)
    dst_ptr_val = tl.min(dst_ptr_u64, axis=0)
    
    # Extract source address (min of pointer block where data is stored)
    src_ptr_u64 = src_ptr.to(tl.uint64)
    src_ptr_val = tl.min(src_ptr_u64, axis=0)
    
    # Calculate size in bytes from pointer range
    # max_ptr - min_ptr gives us the byte distance to the last element
    # Add element_size to include the last element itself
    max_src_ptr = tl.max(src_ptr_u64, axis=0)
    element_size_bytes = 4  # float32
    num_bytes = (max_src_ptr - src_ptr_val + element_size_bytes).to(tl.uint32)
    size_bytes = num_bytes
    
    # Write header fields (but NOT size_bytes yet - it's the ready flag)
    # Write dst_ptr (offset 0)
    tl.store(slot_ptr_u64 + 0, dst_ptr_val)
    
    # Write src_ptr (offset 8)
    tl.store(slot_ptr_u64 + 1, src_ptr_val)
    
    # Write rank + op_type (offset 20-23)
    metadata = (to_rank & 0xFFFF) | ((op_code & 0xFF) << 16)
    slot_ptr_u32 = slot_ptr_u64.to(tl.pointer_type(tl.uint32))
    tl.store(slot_ptr_u32 + 5, metadata.to(tl.uint32))
    
    # Write size_bytes LAST as ready flag (offset 16)
    size_bytes_ptr = (slot_ptr_u32 + 4).to(tl.pointer_type(tl.uint32))
    tl.atomic_xchg(size_bytes_ptr, size_bytes, sem='release', scope='sys')
    
    # Return queue position for waiting
    return prev_head


@triton.jit
def put(dst_ptr, src_ptr, data, dst_rank: tl.constexpr, device_ctx, mask):
    """
    RDMA put (write) operation from Triton kernel.
    
    Enqueues data to be written to remote rank via RDMA.
    Data must first be stored in the registered heap at src_ptr location.
    The CPU proxy thread will dequeue and perform the actual RDMA write.
    
    Args:
        dst_ptr: Destination pointer (remote address) - can be block of pointers
        src_ptr: Source pointer (local address in registered heap) - can be block of pointers
        data: Data values to write (block) - will be stored at src_ptr
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
        >>>     data = generate_data(offsets)
        >>>     # Store data locally first (in registered heap)
        >>>     tl.store(src_ptr + offsets, data, mask=mask)
        >>>     # Enqueue RDMA operation
        >>>     iris_rdma.put(dst_ptr + offsets, src_ptr + offsets, data, dst_rank, device_ctx, mask)
    """
    # Extract queue pointer from device context
    # Context format: [rank, world_size, queue_ptr, heap_base_0, heap_base_1, ...]
    queue_ptr = tl.load(device_ctx + 2)
    
    # Store data in registered heap first
    tl.store(src_ptr, data, mask=mask)
    
    # Enqueue PUT operation (op_code=1)
    _enqueue_rdma_op(dst_ptr, src_ptr, dst_rank, 1, queue_ptr, mask)


@triton.jit
def get(dst_ptr, src_ptr, from_rank: tl.constexpr, device_ctx, mask):
    """
    RDMA get (read) operation from Triton kernel.
    
    Enqueues a request to read data from remote rank via RDMA and WAITS for completion.
    The CPU proxy thread will dequeue, perform the RDMA read, then pop the item.
    This function spins until the tail pointer advances, then data is ready at dst_ptr.
    
    Args:
        dst_ptr: Local destination pointer where data will be written - can be block of pointers
        src_ptr: Source pointer (remote address) - can be block of pointers
        from_rank: Source rank ID (must be compile-time constant)
        device_ctx: Device context from iris_rdma.get_device_context()
        mask: Triton mask for valid elements
    
    Example:
        >>> @triton.jit
        >>> def kernel(local_ptr, remote_ptr, device_ctx, from_rank, BLOCK_SIZE: tl.constexpr):
        >>>     pid = tl.program_id(0)
        >>>     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        >>>     mask = offsets < n_elements
        >>>     
        >>>     # RDMA read from remote rank - blocks until complete
        >>>     iris_rdma.get(local_ptr + offsets, remote_ptr + offsets, from_rank, device_ctx, mask)
        >>>     
        >>>     # Data is now ready at local_ptr, can use it immediately
        >>>     data = tl.load(local_ptr + offsets, mask=mask)
    """
    # Extract queue pointer from device context
    queue_ptr = tl.load(device_ctx + 2)
    
    # Enqueue GET operation (op_code=2)
    # For GET: src_ptr is remote source, dst_ptr is local destination
    queue_pos = _enqueue_rdma_op(src_ptr, dst_ptr, from_rank, 2, queue_ptr, mask)
    
    # Wait for CPU to complete the RDMA read
    _wait_for_completion(queue_ptr, queue_pos)
    
    # Data is now ready at dst_ptr (CPU has written it there via RDMA)


__all__ = [
    "IrisRDMA",
    "iris",
    "put",
    "get",
]

