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


# Import tensor operations for use in IrisBase methods
from iris._tensor_ops import create_zeros, create_ones, create_full, create_zeros_like


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

    def zeros(self, *size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
        """
        Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size.
        The tensor is allocated on the Iris symmetric heap.

        Args:
            *size (int...): a sequence of integers defining the shape of the output tensor.
                Can be a variable number of arguments or a collection like a list or tuple.

        Keyword Arguments:
            out (Tensor, optional): the output tensor.
            dtype (torch.dtype, optional): the desired data type of returned tensor.
                Default: if None, uses a global default (see torch.set_default_dtype()).
            layout (torch.layout, optional): the desired layout of returned Tensor.
                Default: torch.strided. Note: Iris tensors always use `torch.strided` regardless of this parameter.
            device (torch.device, optional): the desired device of returned tensor.
                Default: if None, uses the current device for the default tensor type.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor.
                Default: False.

        Returns:
            torch.Tensor: Zero-initialized tensor on the symmetric heap

        Example:
            >>> import iris  # or: from iris.experimental import iris_gluon
            >>> ctx = iris.Iris(1 << 20)  # or: ctx = iris_gluon.IrisGluon(1 << 20)
            >>> tensor = ctx.zeros(2, 3)
            >>> print(tensor.shape)  # torch.Size([2, 3])
            >>> print(tensor[0])  # tensor([0., 0., 0.], device='cuda:0')
        """
        return create_zeros(
            self, *size, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
        )

    def ones(self, *size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
        """
        Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size.
        The tensor is allocated on the Iris symmetric heap.

        Args:
            *size (int...): a sequence of integers defining the shape of the output tensor.
                Can be a variable number of arguments or a collection like a list or tuple.

        Keyword Arguments:
            out (Tensor, optional): the output tensor.
            dtype (torch.dtype, optional): the desired data type of returned tensor.
                Default: if None, uses a global default (see torch.set_default_dtype()).
            layout (torch.layout, optional): the desired layout of returned Tensor.
                Default: torch.strided. Note: Iris tensors always use `torch.strided` regardless of this parameter.
            device (torch.device, optional): the desired device of returned tensor.
                Default: if None, uses the current device for the default tensor type.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor.
                Default: False.

        Returns:
            torch.Tensor: Ones-initialized tensor on the symmetric heap

        Example:
            >>> import iris  # or: from iris.experimental import iris_gluon
            >>> ctx = iris.Iris(1 << 20)  # or: ctx = iris_gluon.IrisGluon(1 << 20)
            >>> tensor = ctx.ones(2, 3)
            >>> print(tensor.shape)  # torch.Size([2, 3])
            >>> print(tensor[0])  # tensor([1., 1., 1.], device='cuda:0')
        """
        return create_ones(self, *size, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    def full(self, size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
        """
        Creates a tensor of size size filled with fill_value. The tensor's dtype is inferred from fill_value.
        The tensor is allocated on the Iris symmetric heap.

        Args:
            size (int...): a list, tuple, or torch.Size of integers defining the shape of the output tensor.
            fill_value (Scalar): the value to fill the output tensor with.

        Keyword Arguments:
            out (Tensor, optional): the output tensor.
            dtype (torch.dtype, optional): the desired data type of returned tensor.
                Default: if None, uses a global default (see torch.set_default_dtype()).
            layout (torch.layout, optional): the desired layout of returned Tensor.
                Default: torch.strided. Note: Iris tensors always use `torch.strided` regardless of this parameter.
            device (torch.device, optional): the desired device of returned tensor.
                Default: if None, uses the current device for the default tensor type.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor.
                Default: False.

        Returns:
            torch.Tensor: Tensor filled with fill_value

        Example:
            >>> import iris  # or: from iris.experimental import iris_gluon
            >>> ctx = iris.Iris(1 << 20)  # or: ctx = iris_gluon.IrisGluon(1 << 20)
            >>> tensor = ctx.full((2, 3), 3.14)
            >>> print(tensor.shape)  # torch.Size([2, 3])
            >>> print(tensor[0])  # tensor([3.1400, 3.1400, 3.1400], device='cuda:0')
        """
        return create_full(
            self, size, fill_value, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
        )

    def zeros_like(
        self,
        input,
        *,
        dtype=None,
        layout=None,
        device=None,
        requires_grad=False,
        memory_format=torch.preserve_format,
    ):
        """
        Returns a tensor filled with the scalar value 0, with the same size as input,
        allocated on the Iris symmetric heap.

        Args:
            input (Tensor): the size of input will determine size of the output tensor.

        Keyword Arguments:
            dtype (torch.dtype, optional): the desired data type of returned Tensor.
                Default: if None, defaults to the dtype of input.
            layout (torch.layout, optional): the desired layout of returned tensor.
                Default: if None, defaults to the layout of input. Note: Iris tensors are always contiguous (strided).
            device (torch.device, optional): the desired device of returned tensor.
                Default: if None, defaults to the device of input. Must be compatible with this Iris instance.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor.
                Default: False.
            memory_format (torch.memory_format, optional): the desired memory format of returned Tensor.
                Default: torch.preserve_format.

        Returns:
            torch.Tensor: Zero-initialized tensor with same shape as input

        Example:
            >>> import iris  # or: from iris.experimental import iris_gluon
            >>> ctx = iris.Iris(1 << 20)  # or: ctx = iris_gluon.IrisGluon(1 << 20)
            >>> input_tensor = ctx.ones(2, 3)
            >>> zeros_tensor = ctx.zeros_like(input_tensor)
            >>> print(zeros_tensor.shape)  # torch.Size([2, 3])
        """
        return create_zeros_like(
            self,
            input,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
            memory_format=memory_format,
        )

    def arange(
        self, start=0, end=None, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False
    ):
        """
        Returns a 1-D tensor of size ⌈(end - start) / step⌉ with values from the interval [start, end)
        taken with common difference step beginning from start. The tensor is allocated on the symmetric heap.

        Note: When using floating-point dtypes (especially reduced precision types like bfloat16),
        the results may be affected by floating-point rounding behavior. Some values in the sequence
        might not be exactly representable in certain floating-point formats, which can lead to
        repeated values or unexpected rounding. For precise sequences, it is recommended to use
        integer dtypes instead of floating-point dtypes.

        Note that non-integer step is subject to floating point rounding errors when comparing
        against end; to avoid inconsistency, we advise subtracting a small epsilon from end in such cases.

        Args:
            start (Number, optional): the starting value for the set of points. Default: 0.
            end (Number): the ending value for the set of points
            step (Number, optional): the gap between each pair of adjacent points. Default: 1.
            out (Tensor, optional): the output tensor.
            dtype (torch.dtype, optional): the desired data type of returned tensor.
            layout (torch.layout, optional): the desired layout of returned Tensor. Default: torch.strided.
            device (torch.device, optional): the desired device of returned tensor.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

        Returns:
            torch.Tensor: 1-D tensor with evenly spaced values

        Example:
            >>> import iris  # or: from iris.experimental import iris_gluon
            >>> ctx = iris.Iris(1 << 20)  # or: ctx = iris_gluon.IrisGluon(1 << 20)
            >>> tensor = ctx.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
            >>> print(tensor.shape)  # torch.Size([5])
        """
        self.debug(f"arange: start = {start}, end = {end}, step = {step}, dtype = {dtype}, device = {device}")

        # Handle the case where only one argument is provided (end)
        if end is None:
            end = start
            start = 0

        # Validate inputs
        if step == 0:
            raise ValueError("step must be non-zero")

        # Validate step direction consistency
        if step > 0 and start >= end:
            raise ValueError(f"Invalid range: start >= end with positive step (start={start}, end={end}, step={step})")
        elif step < 0 and start <= end:
            raise ValueError(f"Invalid range: start <= end with negative step (start={start}, end={end}, step={step})")

        # Calculate the number of elements
        num_elements = math.ceil((end - start) / step)

        # Infer dtype if not provided
        if dtype is None:
            if any(isinstance(x, float) for x in [start, end, step]):
                dtype = torch.get_default_dtype()
            else:
                dtype = torch.int64

        # Use current device if none specified
        if device is None:
            device = self.device

        # Validate device compatibility with Iris
        self._throw_if_invalid_device(device)

        if out is not None:
            self._throw_if_invalid_output_tensor(out, num_elements, dtype)
            tensor = out
        else:
            tensor = self._allocate(num_elements=num_elements, dtype=dtype)

        target_device = tensor.device
        arange_tensor = torch.arange(start, end, step, dtype=dtype, device=target_device)

        tensor[:] = arange_tensor

        tensor = self._apply_layout(tensor, layout)

        if requires_grad:
            tensor.requires_grad_()

        return tensor

    def randn(
        self,
        *size,
        generator=None,
        out=None,
        dtype=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
        pin_memory=False,
    ):
        """
        Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1.
        The tensor is allocated on the Iris symmetric heap.

        Args:
            *size (int...): a sequence of integers defining the shape of the output tensor.

        Keyword Arguments:
            generator (torch.Generator, optional): a pseudorandom number generator for sampling
            out (Tensor, optional): the output tensor.
            dtype (torch.dtype, optional): the desired data type of returned tensor.
            layout (torch.layout, optional): the desired layout of returned Tensor. Default: torch.strided.
            device (torch.device, optional): the desired device of returned tensor.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.
            pin_memory (bool, optional): If set, returned tensor would be allocated in the pinned memory. Default: False.

        Returns:
            torch.Tensor: Tensor filled with random numbers from normal distribution

        Example:
            >>> import iris  # or: from iris.experimental import iris_gluon
            >>> ctx = iris.Iris(1 << 20)  # or: ctx = iris_gluon.IrisGluon(1 << 20)
            >>> tensor = ctx.randn(2, 3)
            >>> print(tensor.shape)  # torch.Size([2, 3])
        """
        self.debug(
            f"randn: size = {size}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}, pin_memory = {pin_memory}"
        )

        # Use global default dtype if None is provided
        if dtype is None:
            dtype = torch.get_default_dtype()

        # Use current device if none specified
        if device is None:
            device = self.device

        # Validate device compatibility with Iris
        self._throw_if_invalid_device(device)

        # Parse size and calculate number of elements
        size, num_elements = self._parse_size(size)

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self._throw_if_invalid_output_tensor(out, num_elements, dtype)
            random_data = torch.randn(num_elements, generator=generator, dtype=dtype, device=device, layout=layout)
            out.copy_(random_data)
            tensor = out.view(size)
        else:
            tensor = self._allocate(num_elements=num_elements, dtype=dtype)
            random_data = torch.randn(num_elements, generator=generator, dtype=dtype, device=device, layout=layout)
            tensor.copy_(random_data)
            tensor = tensor.reshape(size)

        # Apply the requested layout
        tensor = self._apply_layout(tensor, layout)

        # Set requires_grad if specified
        if requires_grad:
            tensor.requires_grad_()

        return tensor

    def uniform(self, size, low=0.0, high=1.0, dtype=torch.float):
        """
        Returns a tensor filled with random numbers from a uniform distribution, allocated on the Iris symmetric heap.

        Args:
            size (int or tuple of ints): the size of the output tensor.
            low (float, optional): the lower bound of the uniform distribution. Default: 0.0.
            high (float, optional): the upper bound of the uniform distribution. Default: 1.0.
            dtype (torch.dtype, optional): the desired data type of returned tensor. Default: torch.float.

        Returns:
            torch.Tensor: A tensor filled with random numbers from a uniform distribution.

        Example:
            >>> import iris  # or: from iris.experimental import iris_gluon
            >>> ctx = iris.Iris(1 << 20)  # or: ctx = iris_gluon.IrisGluon(1 << 20)
            >>> tensor = ctx.uniform((2, 3), low=0.0, high=1.0)
            >>> print(tensor.shape)  # torch.Size([2, 3])
        """
        self.debug(f"uniform: size = {size}, low = {low}, high = {high}, dtype = {dtype}")
        size, num_elements = self._parse_size(size)
        tensor = self._allocate(num_elements=num_elements, dtype=dtype)
        tensor.uniform_(low, high)
        return tensor.reshape(size)

    def empty(
        self,
        *size,
        out=None,
        dtype=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
        pin_memory=False,
        memory_format=torch.contiguous_format,
    ):
        """
        Returns a tensor filled with uninitialized data. The shape of the tensor is defined by the variable argument size.
        The tensor is allocated on the Iris symmetric heap.

        Args:
            *size (int...): a sequence of integers defining the shape of the output tensor.

        Keyword Arguments:
            out (Tensor, optional): the output tensor.
            dtype (torch.dtype, optional): the desired data type of returned tensor.
            layout (torch.layout, optional): the desired layout of returned Tensor. Default: torch.strided.
            device (torch.device, optional): the desired device of returned tensor.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.
            pin_memory (bool, optional): If set, returned tensor would be allocated in the pinned memory. Default: False.
            memory_format (torch.memory_format, optional): the desired memory format of returned Tensor. Default: torch.contiguous_format.

        Returns:
            torch.Tensor: Tensor with uninitialized data

        Example:
            >>> import iris  # or: from iris.experimental import iris_gluon
            >>> ctx = iris.Iris(1 << 20)  # or: ctx = iris_gluon.IrisGluon(1 << 20)
            >>> tensor = ctx.empty(2, 3)
            >>> print(tensor.shape)  # torch.Size([2, 3])
        """
        self.debug(
            f"empty: size = {size}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}, pin_memory = {pin_memory}"
        )

        # Use global default dtype if None is provided
        if dtype is None:
            dtype = torch.get_default_dtype()

        # Use current device if none specified
        if device is None:
            device = self.device

        # Validate device compatibility with Iris
        self._throw_if_invalid_device(device)

        # Parse size and calculate number of elements
        size, num_elements = self._parse_size(size)

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self._throw_if_invalid_output_tensor(out, num_elements, dtype)
            tensor = out.view(size)
        else:
            tensor = self._allocate(num_elements=num_elements, dtype=dtype)
            tensor = tensor.reshape(size)

        # Apply the requested layout
        tensor = self._apply_layout(tensor, layout)

        # Set requires_grad if specified
        if requires_grad:
            tensor.requires_grad_()

        return tensor

    def randint(
        self, *args, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False
    ):
        """
        Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).
        The shape of the tensor is defined by the variable argument size. The tensor is allocated on the Iris symmetric heap.

        Args:
            low (int, optional): Lowest integer to be drawn from the distribution. Default: 0.
            high (int): One above the highest integer to be drawn from the distribution.
            size (tuple): a tuple defining the shape of the output tensor.

        Keyword Arguments:
            generator (torch.Generator, optional): a pseudorandom number generator for sampling.
            out (Tensor, optional): the output tensor.
            dtype (torch.dtype, optional): if None, this function returns a tensor with dtype torch.int64.
            layout (torch.layout, optional): the desired layout of returned Tensor. Default: torch.strided.
            device (torch.device, optional): the desired device of returned tensor.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

        Returns:
            torch.Tensor: Tensor filled with random integers

        Example:
            >>> import iris  # or: from iris.experimental import iris_gluon
            >>> ctx = iris.Iris(1 << 20)  # or: ctx = iris_gluon.IrisGluon(1 << 20)
            >>> tensor = ctx.randint(0, 10, (2, 3))  # Random integers [0, 10)
            >>> print(tensor.shape)  # torch.Size([2, 3])
        """
        self.debug(f"randint: args = {args}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}")

        # Parse arguments to determine low, high, and size
        if len(args) == 2:
            high, size = args
            low = 0
        elif len(args) == 3:
            low, high, size = args
        else:
            raise ValueError(f"randint expects 2 or 3 positional arguments, got {len(args)}")

        # Use default dtype if None is provided
        if dtype is None:
            dtype = torch.int64

        # Use current device if none specified
        if device is None:
            device = self.device

        # Validate device compatibility with Iris
        self._throw_if_invalid_device(device)

        # Parse size and calculate number of elements
        size, num_elements = self._parse_size(size)

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self._throw_if_invalid_output_tensor(out, num_elements, dtype)
            tensor = out.view(size)
        else:
            tensor = self._allocate(num_elements=num_elements, dtype=dtype)
            tensor = tensor.reshape(size)

        # Generate random integers using PyTorch's randint
        target_device = device if device is not None else self.device

        # Handle generator parameter
        if generator is not None:
            torch.randint(low, high, size, generator=generator, out=tensor, dtype=dtype, device=target_device)
        else:
            torch.randint(low, high, size, out=tensor, dtype=dtype, device=target_device)

        # Apply the requested layout
        tensor = self._apply_layout(tensor, layout)

        # Set requires_grad if specified
        if requires_grad:
            tensor.requires_grad_()

        return tensor

    def linspace(self, start, end, steps, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
        """
        Creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.
        The tensor is allocated on the Iris symmetric heap.

        Args:
            start (float or Tensor): the starting value for the set of points.
            end (float or Tensor): the ending value for the set of points.
            steps (int): size of the constructed tensor.

        Keyword Arguments:
            out (Tensor, optional): the output tensor.
            dtype (torch.dtype, optional): the data type to perform the computation in.
            layout (torch.layout, optional): the desired layout of returned Tensor. Default: torch.strided.
            device (torch.device, optional): the desired device of returned tensor.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

        Returns:
            torch.Tensor: 1-D tensor with evenly spaced values from start to end

        Example:
            >>> import iris  # or: from iris.experimental import iris_gluon
            >>> ctx = iris.Iris(1 << 20)  # or: ctx = iris_gluon.IrisGluon(1 << 20)
            >>> tensor = ctx.linspace(0, 10, 5)  # [0, 2.5, 5, 7.5, 10]
            >>> print(tensor)
        """
        self.debug(
            f"linspace: start = {start}, end = {end}, steps = {steps}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}"
        )

        # Use global default dtype if None is provided
        if dtype is None:
            # Check if start or end are complex numbers
            start_is_complex = isinstance(start, complex) or (hasattr(start, "dtype") and torch.is_complex(start))
            end_is_complex = isinstance(end, complex) or (hasattr(end, "dtype") and torch.is_complex(end))

            if start_is_complex or end_is_complex:
                dtype = torch.complex64 if torch.get_default_dtype() == torch.float32 else torch.complex128
            else:
                dtype = torch.get_default_dtype()

        # Use current device if none specified
        if device is None:
            device = self.device

        # Validate device compatibility with Iris
        self._throw_if_invalid_device(device)

        # Parse steps and extract the integer value
        if isinstance(steps, (tuple, list)):
            if len(steps) == 1:
                steps_int = steps[0]
                if isinstance(steps_int, (tuple, list)):
                    steps_int = steps_int[0]
            else:
                size, num_elements = self._parse_size(steps)
                steps_int = num_elements
        else:
            steps_int = steps

        steps_int = int(steps_int)
        size = (steps_int,)
        num_elements = steps_int

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self._throw_if_invalid_output_tensor(out, num_elements, dtype)
            tensor = out.view(size)
        else:
            tensor = self._allocate(num_elements=num_elements, dtype=dtype)
            tensor = tensor.reshape(size)

        # Generate linspace using PyTorch's linspace
        target_device = device if device is not None else self.device
        torch.linspace(start, end, steps_int, out=tensor, dtype=dtype, device=target_device)

        # Apply the requested layout
        tensor = self._apply_layout(tensor, layout)

        # Set requires_grad if specified
        if requires_grad:
            tensor.requires_grad_()

        return tensor

    def rand(
        self,
        *size,
        generator=None,
        out=None,
        dtype=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
        pin_memory=False,
    ):
        """
        Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1).
        The tensor is allocated on the Iris symmetric heap.

        Args:
            *size (int...): a sequence of integers defining the shape of the output tensor.

        Keyword Arguments:
            generator (torch.Generator, optional): a pseudorandom number generator for sampling.
            out (Tensor, optional): the output tensor.
            dtype (torch.dtype, optional): the desired data type of returned tensor.
            layout (torch.layout, optional): the desired layout of returned Tensor. Default: torch.strided.
            device (torch.device, optional): the desired device of returned tensor.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.
            pin_memory (bool, optional): If set, returned tensor would be allocated in the pinned memory. Default: False.

        Returns:
            torch.Tensor: Tensor filled with random values in [0, 1)

        Example:
            >>> import iris  # or: from iris.experimental import iris_gluon
            >>> ctx = iris.Iris(1 << 20)  # or: ctx = iris_gluon.IrisGluon(1 << 20)
            >>> tensor = ctx.rand(2, 3)  # Random values in [0, 1)
            >>> print(tensor.shape)  # torch.Size([2, 3])
        """
        self.debug(
            f"rand: size = {size}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}, pin_memory = {pin_memory}"
        )

        # Use global default dtype if None is provided
        if dtype is None:
            dtype = torch.get_default_dtype()

        # Use current device if none specified
        if device is None:
            device = self.device

        # Validate device compatibility with Iris
        self._throw_if_invalid_device(device)

        # Parse size and calculate number of elements
        size, num_elements = self._parse_size(size)

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self._throw_if_invalid_output_tensor(out, num_elements, dtype)
            tensor = out.view(size)
        else:
            tensor = self._allocate(num_elements=num_elements, dtype=dtype)
            tensor = tensor.reshape(size)

        # Generate random numbers using PyTorch's rand
        if generator is not None:
            torch.rand(size, generator=generator, out=tensor, dtype=dtype, device=device)
        else:
            torch.rand(size, out=tensor, dtype=dtype, device=device)

        # Apply the requested layout
        tensor = self._apply_layout(tensor, layout)

        # Set requires_grad if specified
        if requires_grad:
            tensor.requires_grad_()

        return tensor

    def get_device_context(self):
        """
        Get the device context tensor for kernels.

        Returns a tensor encoding: `[cur_rank, num_ranks, heap_base_0, heap_base_1, ...]`

        This method is useful for both Gluon kernels and future Triton backends that
        utilize aggregates for passing context information.

        Returns:
            torch.Tensor: Encoded context data as int64 tensor on device

        Example:
            >>> import iris  # or: from iris.experimental import iris_gluon
            >>> ctx = iris.Iris(1 << 20)  # or: ctx = iris_gluon.IrisGluon(1 << 20)
            >>> context_tensor = ctx.get_device_context()
            >>>
            >>> @gluon.jit
            >>> def kernel(IrisDeviceCtx: gl.constexpr, context_tensor):
            >>>     ctx = IrisDeviceCtx.initialize(context_tensor)
            >>>     data = ctx.load(buffer, 1)
        """
        # Convert heap_bases to a list for concatenation
        heap_bases_list = self.heap_bases.tolist()

        # Create context tensor: [cur_rank, num_ranks, heap_base_0, heap_base_1, ...]
        context_data = [self.cur_rank, self.num_ranks] + heap_bases_list
        context_tensor = torch.tensor(context_data, dtype=torch.int64, device=self.device)

        return context_tensor

    def get_backend(self):
        """
        Legacy method for backward compatibility.
        Use get_device_context() for kernel context.

        Returns:
            torch.Tensor: Device context tensor
        """
        return self.get_device_context()


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
