# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Common base class and utilities shared between Iris and IrisGluon.

This module contains the shared implementation for initialization,
logging, device validation, and utility methods used by both
Triton and Gluon backends.
"""

import numpy as np
import math
import torch
import ctypes
import logging

from iris._distributed_helpers import (
    init_distributed,
    distributed_allgather,
    distributed_barrier,
    distributed_broadcast_scalar,
    distributed_broadcast_tensor,
)
from iris.hip import (
    set_device,
    get_cu_count,
    count_devices,
    get_ipc_handle,
    open_ipc_handle,
    get_ipc_handle_size,
)
from iris.logging import logger


class IrisBase:
    """
    Base class for Iris implementations containing shared functionality.

    This class provides common initialization, logging, device validation,
    and utility methods used by both Triton and Gluon backends.
    """

    def __init__(self, heap_size=1 << 30):
        """
        Initialize the Iris base class.

        Args:
            heap_size (int): Size of the symmetric heap in bytes. Default: 1GB (2^30)
        """
        # Initialize distributed environment
        comm, cur_rank, num_ranks = init_distributed()
        num_gpus = count_devices()

        gpu_id = cur_rank % num_gpus
        set_device(gpu_id)

        self.comm = comm
        self.num_ranks = num_ranks
        self.cur_rank = cur_rank
        self.gpu_id = gpu_id
        self.heap_size = heap_size
        self.heap_offset = 0
        self.alignment = 1024
        self.device = f"cuda:{gpu_id}"
        self.memory_pool = torch.empty(heap_size, device=self.device, dtype=torch.int8)

        heap_base = self.memory_pool.data_ptr()
        heap_base_ptr = ctypes.c_void_p(heap_base)

        heap_bases = np.zeros(num_ranks, dtype=np.uint64)
        heap_bases[cur_rank] = heap_base
        ipc_handle_size = get_ipc_handle_size()
        ipc_handles = np.zeros((num_ranks, ipc_handle_size), dtype=np.uint8)
        ipc_handle = get_ipc_handle(heap_base_ptr, cur_rank)

        distributed_barrier()

        all_ipc_handles = distributed_allgather(np.frombuffer(ipc_handle, dtype=np.uint8).copy())
        heap_base_bytes = np.array([heap_bases[cur_rank]], dtype=np.uint64).tobytes()
        all_heap_bases_bytes = distributed_allgather(np.frombuffer(heap_base_bytes, dtype=np.uint8).copy())
        all_heap_bases = np.frombuffer(all_heap_bases_bytes.tobytes(), dtype=np.uint64).reshape(num_ranks, -1)

        distributed_barrier()

        ipc_heap_bases = np.zeros(num_ranks, dtype=np.uintp)
        for rank in range(num_ranks):
            if rank != cur_rank:
                handle = open_ipc_handle(all_ipc_handles[rank], cur_rank)
                ipc_heap_bases[rank] = int(handle)
            else:
                ipc_heap_bases[rank] = heap_bases[rank]

        for i in range(num_ranks):
            self.debug(f"GPU {i}: Heap base {hex(int(ipc_heap_bases[i]))}")

        distributed_barrier()
        self.heap_bases = torch.from_numpy(ipc_heap_bases).to(device=self.device, dtype=torch.uint64)

        distributed_barrier()

    def _log_with_rank(self, level, message):
        """Helper method to log with rank information injected into the record."""
        if logger.isEnabledFor(level):
            record = logging.LogRecord(
                name=logger.name, level=level, pathname="", lineno=0, msg=message, args=(), exc_info=None
            )
            # Inject rank information into the record
            record.iris_rank = self.cur_rank
            record.iris_num_ranks = self.num_ranks
            logger.handle(record)

    def debug(self, message):
        """
        Log a debug message with rank information.

        Args:
            message (str): Human-readable message to log at debug level.
        """
        self._log_with_rank(logging.DEBUG, message)

    def info(self, message):
        """
        Log an info message with rank information.

        Args:
            message (str): Human-readable message to log at info level.
        """
        self._log_with_rank(logging.INFO, message)

    def warning(self, message):
        """
        Log a warning message with rank information.

        Args:
            message (str): Human-readable message to log at warning level.
        """
        self._log_with_rank(logging.WARNING, message)

    def error(self, message):
        """
        Log an error message with rank information.

        Args:
            message (str): Human-readable message to log at error level.
        """
        self._log_with_rank(logging.ERROR, message)

    def broadcast(self, value, source_rank):
        """
        Broadcast a value from one rank to all ranks.

        This method automatically detects the type of value and uses the appropriate
        broadcast mechanism:
        - For tensors and arrays: uses efficient PyTorch distributed tensor collectives
        - For scalars and other objects: uses object broadcast

        Args:
            value (Any): The value to broadcast. Can be a scalar, tensor, numpy array,
                or any picklable object. Only the ``source_rank`` value is used;
                other ranks should pass a placeholder (e.g., ``None``).
            source_rank (int): Rank id that holds the authoritative value.

        Returns:
            Any: The value broadcast to all ranks. Tensors and arrays are returned as
                numpy arrays; scalars and objects are returned in their original type.
        """
        # Check if the value on source_rank is a tensor or array-like
        if self.cur_rank == source_rank and value is not None:
            # Explicitly exclude strings and non-numeric types
            if isinstance(value, (str, dict, bool)):
                is_tensor = False
            elif isinstance(value, torch.Tensor):
                is_tensor = True
            elif isinstance(value, np.ndarray):
                is_tensor = True
            elif isinstance(value, (list, tuple)):
                # Try to convert list/tuple to tensor to check if it's numeric
                try:
                    torch.as_tensor(value)
                    is_tensor = True
                except (TypeError, ValueError):
                    is_tensor = False
            else:
                # For other types, try to convert and check
                try:
                    test_array = np.asarray(value)
                    # Check if it's a numeric dtype that torch can handle
                    if np.issubdtype(test_array.dtype, np.number):
                        torch.as_tensor(test_array)
                        is_tensor = True
                    else:
                        is_tensor = False
                except (TypeError, ValueError):
                    is_tensor = False
        else:
            is_tensor = False

        # Broadcast the type decision to all ranks
        is_tensor = distributed_broadcast_scalar(is_tensor, source_rank)

        if is_tensor:
            return distributed_broadcast_tensor(value, root=source_rank)
        else:
            return distributed_broadcast_scalar(value, source_rank)

    def _allocate(self, num_elements, dtype):
        """
        Internal method to allocate memory from the symmetric heap.

        Args:
            num_elements (int): Number of elements to allocate
            dtype (torch.dtype): Data type of the elements

        Returns:
            torch.Tensor: Allocated tensor on the symmetric heap
        """
        self.debug(f"allocate: num_elements = {num_elements}, dtype = {dtype}")

        element_size = torch.tensor([], dtype=dtype).element_size()
        size_in_bytes = num_elements * element_size
        aligned_size = math.ceil(size_in_bytes / self.alignment) * self.alignment

        if self.heap_offset + aligned_size > self.heap_size:
            raise MemoryError("Heap out of memory")

        start = self.heap_offset
        self.heap_offset += aligned_size

        sub_buffer = self.memory_pool[start : start + size_in_bytes].view(dtype)
        return sub_buffer.reshape((num_elements,))

    def _parse_size(self, size):
        """
        Parse size parameter and calculate number of elements.

        Args:
            size (tuple): Size specification (can be nested)

        Returns:
            tuple: (parsed_size, num_elements)
        """
        # Handle nested tuples/lists by flattening them recursively
        while len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = size[0]
        num_elements = math.prod(size)
        return size, num_elements

    def _throw_if_invalid_output_tensor(self, tensor: torch.Tensor, num_elements: int, dtype: torch.dtype):
        """
        Validate that an output tensor meets requirements.

        Args:
            tensor: The tensor to validate
            num_elements: Expected number of elements
            dtype: Expected data type

        Raises:
            RuntimeError: If validation fails
        """
        if not self._tensor_on_device(tensor):
            raise RuntimeError(
                f"The output tensor is not on the same device as the Iris instance. "
                f"The Iris instance is on device {self.device} but the output tensor is on device {tensor.device}"
            )
        if not self._on_symmetric_heap(tensor):
            raise RuntimeError(
                f"The output tensor is not on the symmetric heap. "
                f"The Iris instance is on heap base {self.heap_bases[self.cur_rank]} "
                f"but the output tensor is on heap base {tensor.data_ptr()}"
            )
        if tensor.numel() != num_elements:
            raise RuntimeError(f"The output tensor has {tensor.numel()} elements, but {num_elements} are required")
        if tensor.dtype != dtype:
            raise RuntimeError(f"The output tensor has dtype {tensor.dtype}, but {dtype} is required")

    def _throw_if_invalid_device(self, device):
        """
        Throw a RuntimeError if the requested device is not compatible with this Iris instance.

        Args:
            device: The requested device (can be string, torch.device, or None)

        Raises:
            RuntimeError: If the device is not compatible
        """
        if not self._is_valid_device(device):
            raise RuntimeError(
                f"Device mismatch: requested device {device} but Iris instance is on device {self.device}. "
                f"Iris only supports tensors on its own device."
            )

    def _apply_layout(self, tensor: torch.Tensor, layout: torch.layout) -> torch.Tensor:
        """
        Apply the requested layout to a tensor.

        Args:
            tensor: The tensor to modify
            layout: The desired layout

        Returns:
            Tensor with the requested layout
        """
        if layout == torch.strided:
            # Strided layout is the default - no changes needed
            return tensor
        else:
            # Only support strided layout for now
            raise ValueError(f"Layout {layout} not supported. Only torch.strided is currently supported.")

    def _tensor_on_device(self, tensor: torch.Tensor):
        """
        Check if a tensor is on the same device as this Iris instance.

        Args:
            tensor: The tensor to check

        Returns:
            bool: True if tensor is on compatible device
        """
        # Get the Iris device from memory_pool.device
        iris_device = self.get_device()
        tensor_device = tensor.device

        # For CUDA devices, check if they're compatible
        if tensor_device.type == "cuda" and iris_device.type == "cuda":
            if iris_device.index is None:
                return True
            return tensor_device.index == iris_device.index

        # For non-CUDA devices, they must be exactly equal
        return tensor_device == iris_device

    def _on_symmetric_heap(self, tensor: torch.Tensor):
        """
        Check if a tensor is allocated on the symmetric heap.

        Args:
            tensor: The tensor to check

        Returns:
            bool: True if tensor is on symmetric heap
        """
        # Special case for empty tensors - they might not have a valid data_ptr
        if tensor.numel() == 0:
            self.debug("Empty tensor detected, skipping heap check")
            return True

        # Convert CUDA pointer to integer for comparison
        tensor_ptr = int(tensor.data_ptr())
        heap_base = int(self.heap_bases[self.cur_rank])

        result = tensor_ptr >= heap_base and tensor_ptr < heap_base + self.heap_size

        return result

    def _is_valid_device(self, device) -> bool:
        """
        Check if the requested device is compatible with this Iris instance.

        Args:
            device: The requested device (can be string, torch.device, or None)

        Returns:
            bool: True if the device is compatible, False otherwise
        """
        if device is None:
            return True  # None means use default device

        # Convert device strings to torch.device objects for proper comparison
        requested_device = torch.device(device) if isinstance(device, str) else device
        iris_device = self.get_device()

        # Check if both are CUDA devices
        if requested_device.type == "cuda" and iris_device.type == "cuda":
            # Check if index matches or if requested is "cuda" (any index)
            if requested_device.index is None:
                return True
            else:
                return requested_device.index == iris_device.index

        # For non-CUDA devices, always return False
        return False

    def get_heap_bases(self):
        """
        Return the tensor of symmetric heap base addresses for all ranks.

        Returns:
            torch.Tensor: A 1D tensor of ``uint64`` heap base addresses of size ``num_ranks``
            on the Iris device.
        """
        return self.heap_bases

    def barrier(self, stream=None):
        """
        Synchronize all ranks and their CUDA devices.

        This first calls ``torch.cuda.synchronize()`` or ``stream.synchronize()`` to ensure the local GPU has
        finished all queued work, then performs a global distributed barrier so that all
        ranks reach the same point before proceeding.

        Args:
            stream: If stream is given: wait only for that stream before barrier.
                    If stream is None: legacy behavior (device-wide sync).
        """
        # Wait for all GPUs to finish work
        if stream is None:
            torch.cuda.synchronize()
        else:
            stream.synchronize()

        # Distributed barrier
        distributed_barrier()

    def get_device(self):
        """
        Get the underlying device where the Iris symmetric heap resides.

        Returns:
            torch.device: The CUDA device of Iris-managed memory.
        """
        return self.memory_pool.device

    def get_cu_count(self):
        """
        Get the number of compute units (CUs) for the current GPU.

        Returns:
            int: Number of compute units on this rank's GPU.
        """
        return get_cu_count(self.gpu_id)

    def get_rank(self):
        """
        Get this process's rank id in the distributed communicator.

        Returns:
            int: Zero-based rank id of the current process.
        """
        return self.cur_rank

    def get_num_ranks(self):
        """
        Get the total number of ranks in the distributed communicator.

        Returns:
            int: World size (number of ranks).
        """
        return self.num_ranks


class CCLBase:
    """
    Base Collective Communication Library (CCL) interface.

    Provides collective operations that can be called as methods on Iris instances.
    This base class contains common CCL operations shared by both Triton and Gluon backends.
    """

    def __init__(self, iris_instance):
        """
        Initialize CCL with a reference to the parent Iris instance.

        Args:
            iris_instance: The parent Iris instance (either Iris or IrisGluon)
        """
        self._iris = iris_instance

    def all_to_all(self, output_tensor, input_tensor, config=None, async_op=False):
        """
        All-to-all collective operation.

        Each rank sends a tensor chunk to each other rank and receives
        a tensor chunk from each other rank. Input/output tensors should have
        shape (M, N * world_size) where each chunk of N columns corresponds to one rank.

        Args:
            output_tensor: Output tensor of shape (M, N * world_size)
            input_tensor: Input tensor of shape (M, N * world_size)
            config: Config instance with kernel parameters (default: None).
                    If None, uses default Config values.
                    Set config.use_gluon=True to use Gluon implementation with traffic shaping.
            async_op: If False, performs a barrier at the end. If True, returns immediately.
                      Default: False.

        Example:
            >>> shmem = iris.iris()
            >>> shmem.ccl.all_to_all(output_tensor, input_tensor)

            >>> # Custom configuration
            >>> from iris.ccl import Config
            >>> config = Config(block_size_m=128, block_size_n=32)
            >>> shmem.ccl.all_to_all(output_tensor, input_tensor, config=config)
        """
        from iris.ccl.all_to_all import all_to_all as _all_to_all

        _all_to_all(output_tensor, input_tensor, self._iris, config=config, async_op=async_op)

    def all_gather(self, output_tensor, input_tensor, config=None, async_op=False):
        """
        All-gather collective operation.

        Each rank sends its input tensor to all ranks, and all ranks receive
        and concatenate all input tensors along dimension 0 (rows), matching
        torch.distributed.all_gather_into_tensor behavior.

        Args:
            output_tensor: Output tensor of shape (world_size * M, N) - will contain concatenated inputs
            input_tensor: Input tensor of shape (M, N) - local rank's data to send
            config: Config instance with kernel parameters (default: None).
                    If None, uses default Config values.
            async_op: If False, performs a barrier at the end. If True, returns immediately.
                      Default: False.

        Example:
            >>> shmem = iris.iris()
            >>> # Input: (M, N), Output: (world_size * M, N)
            >>> shmem.ccl.all_gather(output_tensor, input_tensor)

            >>> # Custom configuration
            >>> from iris.ccl import Config
            >>> config = Config(block_size_m=128, block_size_n=32)
            >>> shmem.ccl.all_gather(output_tensor, input_tensor, config=config)
        """
        from iris.ccl.all_gather import all_gather as _all_gather

        _all_gather(output_tensor, input_tensor, self._iris, config=config, async_op=async_op)

    def reduce_scatter(self, output_tensor, input_tensor, config=None, async_op=False):
        """
        Reduce-scatter collective operation.

        Each rank reduces its assigned tiles from all ranks' inputs and stores
        the result only to its own output tensor. This is similar to all-reduce
        but without broadcasting the result to all ranks.

        Args:
            output_tensor: Output tensor of shape (M, N) - will contain reduced tiles for this rank
            input_tensor: Input tensor of shape (M, N) - local rank's partial data
            config: Config instance with kernel parameters (default: None).
                    If None, uses default Config values.
                    Only supports reduce_scatter_variant="two_shot".
            async_op: If False, performs a barrier at the end. If True, returns immediately.
                      Default: False.

        Example:
            >>> shmem = iris.iris()
            >>> shmem.ccl.reduce_scatter(output_tensor, input_tensor)

            >>> # Custom configuration
            >>> from iris.ccl import Config
            >>> config = Config(reduce_scatter_variant="two_shot", all_reduce_distribution=1)
            >>> shmem.ccl.reduce_scatter(output_tensor, input_tensor, config=config)
        """
        from iris.ccl.reduce_scatter import reduce_scatter as _reduce_scatter

        _reduce_scatter(output_tensor, input_tensor, self._iris, config=config, async_op=async_op)
