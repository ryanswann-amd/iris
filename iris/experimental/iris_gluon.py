# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Iris Gluon: Gluon-based Multi-GPU Communication Framework

This module provides a Gluon-based implementation of Iris that uses the
`@aggregate` decorator with Gluon's `@gluon.jit` to encapsulate the Iris backend
struct, eliminating the need to pass `heap_bases` around manually.

Key Features:
- Uses Gluon's `@gluon.jit` decorator for device-side methods
- Encapsulates `heap_bases` and rank info in `IrisDeviceCtx` aggregate
- Provides same functionality as original Iris with improved ergonomics

Example:
    >>> import iris.iris_gluon as iris_gl
    >>> ctx = iris_gl.iris(heap_size=2**30)  # 1GB heap
    >>> context_tensor = ctx.get_device_context()  # Get context tensor
    >>>
    >>> @gluon.jit
    >>> def kernel(IrisDeviceCtx: gl.constexpr, context_tensor):
    >>>     ctx = IrisDeviceCtx.initialize(context_tensor)
    >>>     data = ctx.load(buffer, 1)
"""

from triton.language.core import _aggregate as aggregate

# Import Gluon - if this fails, you need to update Triton to a version with Gluon support
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
except ImportError as e:
    raise ImportError(
        "Gluon is not available in your Triton installation. "
        "Please update Triton to a version with Gluon support to use iris.experimental.iris_gluon. "
        "You can install the latest Triton with: pip install --upgrade triton"
    ) from e

import triton.language as tl

from iris._common import IrisBase, CCLBase
import torch


@aggregate
class IrisDeviceCtx:
    """
    Gluon device-side context that decodes the tensor from Iris.get_device_context().

    This aggregate encapsulates the `heap_bases` pointer and provides
    device-side methods for memory operations and atomics using Gluon.

    Attributes:
        cur_rank: Current rank ID
        num_ranks: Total number of ranks
        heap_bases: Pointer to array of heap base addresses for all ranks
    """

    cur_rank: gl.tensor
    num_ranks: gl.tensor
    heap_bases: gl.tensor

    def __init__(self, cur_rank, num_ranks, heap_bases):
        self.cur_rank = cur_rank
        self.num_ranks = num_ranks
        self.heap_bases = heap_bases

    @staticmethod
    @gluon.jit
    def initialize(context_tensor):
        """
        Initialize `IrisDeviceCtx` from the encoded tensor.

        The context tensor has the format: `[cur_rank, num_ranks, heap_base_0, heap_base_1, ...]`

        Args:
            context_tensor: Pointer to encoded context data

        Returns:
            `IrisDeviceCtx`: Initialized device context
        """
        # Decode the tensor: [cur_rank, num_ranks, heap_base_0, heap_base_1, ...]
        cur_rank = gl.load(context_tensor + 0)
        num_ranks = gl.load(context_tensor + 1)

        # Extract heap bases (from index 2 onwards)
        heap_bases = context_tensor + 2  # Offset pointer to start at heap bases

        return IrisDeviceCtx(cur_rank, num_ranks, heap_bases)

    @gluon.jit
    def _translate(self, ptr, from_rank, to_rank):
        """
        Internal function to translate a pointer from one rank's address space to another.

        Args:
            ptr: Pointer in the `from_rank`'s address space
            from_rank: Source rank ID
            to_rank: Target rank ID

        Returns:
            Translated pointer in the `to_rank`'s address space
        """
        from_base = gl.load(self.heap_bases + from_rank)
        to_base = gl.load(self.heap_bases + to_rank)
        # convert to int to compute difference
        ptr_int = tl.cast(ptr, gl.uint64)
        # Find the offset from from_rank heap
        offset = ptr_int - from_base
        # Byte cast for byte offset addition
        to_base_byte = tl.cast(to_base, gl.pointer_type(gl.int8))
        # Find the offset into the to_rank heap
        translated_ptr_byte = to_base_byte + offset
        # Cast to_base back to pointer type
        translated_ptr = tl.cast(translated_ptr_byte, ptr.dtype)

        # Optimization to vectorize the load/store - similar to iris.py
        # This enables the compiler to generate dwordx4 or wider loads
        # Note: Gluon uses scalar multiples, not 2D tuples like Triton
        # ptr = gl.max_contiguous(gl.multiple_of(ptr, 64), 64)
        # translated_ptr = gl.max_contiguous(gl.multiple_of(translated_ptr, 64), 64)

        return translated_ptr

    @gluon.jit
    def load(self, pointer, from_rank, mask=None):
        """
        Loads a value from the specified rank's memory location to the current rank.

        Args:
            pointer: Pointer in the `from_rank`'s address space
            from_rank: The rank ID from which to read the data
            mask: Optional mask for conditional loading

        Returns:
            The loaded value from the target memory location

        Example:
            >>> # Load from rank 1 to current rank
            >>> data = ctx.load(buffer + offsets, 1, mask=mask)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, from_rank)
        result = gl.load(translated_ptr, mask=mask)
        return result

    @gluon.jit
    def store(self, pointer, value, to_rank, mask=None):
        """
        Writes data from the current rank to the specified rank's memory location.

        Args:
            pointer: Pointer in the current rank's address space
            value: The value to store
            to_rank: The rank ID to which the data will be written
            mask: Optional mask for conditional storing

        Example:
            >>> # Store from current rank to rank 1
            >>> ctx.store(buffer + offsets, values, 1, mask=mask)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        gl.store(translated_ptr, value, mask=mask)

    @gluon.jit
    def get(self, from_ptr, to_ptr, from_rank, mask=None):
        """
        Copies data from the specified rank's memory to the current rank's local memory.

        Args:
            from_ptr: Pointer to remote memory in `from_rank`'s address space
            to_ptr: Pointer to local memory in current rank
            from_rank: The rank ID from which to read the data
            mask: Optional mask for conditional operations

        Example:
            >>> # Copy from rank 1 to current rank's local memory
            >>> ctx.get(remote_ptr + offsets, local_ptr + offsets, 1, mask=mask)
        """
        translated_from_ptr = self._translate(from_ptr, self.cur_rank, from_rank)
        data = gl.load(translated_from_ptr, mask=mask)
        gl.store(to_ptr, data, mask=mask)

    @gluon.jit
    def put(self, from_ptr, to_ptr, to_rank, mask=None):
        """
        Copies data from the current rank's local memory to the specified rank's memory.

        Args:
            from_ptr: Pointer to local memory in current rank
            to_ptr: Pointer to remote memory in `to_rank`'s address space
            to_rank: The rank ID to which the data will be written
            mask: Optional mask for conditional operations

        Example:
            >>> # Copy from current rank's local memory to rank 1
            >>> ctx.put(local_ptr + offsets, remote_ptr + offsets, 1, mask=mask)
        """
        translated_to_ptr = self._translate(to_ptr, self.cur_rank, to_rank)
        data = gl.load(from_ptr, mask=mask)
        gl.store(translated_to_ptr, data, mask=mask)

    @gluon.jit
    def copy(self, src_ptr, dst_ptr, from_rank, to_rank, mask=None):
        """
        Copies data from the specified rank's memory into the destination rank's memory.

        This function performs the transfer by translating `src_ptr` from the `from_rank`'s address
        space to the `to_rank`'s address space, performing a masked load from the translated
        source, and storing the loaded data to `dst_ptr` in the `to_rank` memory location.
        If `from_rank` and `to_rank` are the same, this function performs a local copy operation.
        It is undefined behaviour if neither `from_rank` nor `to_rank` is the `cur_rank`.

        Args:
            src_ptr: Pointer in the `from_rank`'s local memory from which to read data
            dst_ptr: Pointer in the `to_rank`'s local memory where the data will be written
            from_rank: The rank ID that owns `src_ptr` (source rank)
            to_rank: The rank ID that will receive the data (destination rank)
            mask: Optional mask for conditional operations

        Example:
            >>> # Copy from rank 1 to rank 0 (current rank must be either 1 or 0)
            >>> ctx.copy(remote_ptr + offsets, local_ptr + offsets, 1, 0, mask=mask)
        """
        cur_base = gl.load(self.heap_bases + self.cur_rank)
        from_base = gl.load(self.heap_bases + from_rank)
        to_base = gl.load(self.heap_bases + to_rank)

        src_ptr_int = tl.cast(src_ptr, gl.uint64)
        src_offset = src_ptr_int - cur_base

        dst_ptr_int = tl.cast(dst_ptr, gl.uint64)
        dst_offset = dst_ptr_int - cur_base

        from_base_byte = tl.cast(from_base, gl.pointer_type(gl.int8))
        to_base_byte = tl.cast(to_base, gl.pointer_type(gl.int8))

        translated_src = tl.cast(from_base_byte + src_offset, src_ptr.dtype)
        translated_dst = tl.cast(to_base_byte + dst_offset, src_ptr.dtype)

        data = gl.load(translated_src, mask=mask)
        gl.store(translated_dst, data, mask=mask)

    @gluon.jit
    def atomic_add(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic add at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to add
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically add to rank 1's memory
            >>> old_val = ctx.atomic_add(buffer, 5, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_add(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_sub(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Atomically subtracts data from the specified rank's memory location.

        Args:
            pointer: Pointer in the current rank's address space
            val: The value to subtract
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically subtract from rank 1's memory
            >>> old_val = ctx.atomic_sub(buffer, 3, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_sub(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_cas(self, pointer, cmp, val, to_rank, sem=None, scope=None):
        """
        Atomically compares and exchanges the specified rank's memory location.

        Args:
            pointer: Pointer in the current rank's address space
            cmp: The expected value to compare
            val: The new value to write if comparison succeeds
            to_rank: The rank ID to which the atomic operation will be performed
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Compare-and-swap on rank 1's memory
            >>> old_val = ctx.atomic_cas(flag + pid, 0, 1, 1, sem="release", scope="sys")
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_cas(translated_ptr, cmp, val, sem=sem, scope=scope)

    @gluon.jit
    def atomic_xchg(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic exchange at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to exchange
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Exchange value with rank 1's memory
            >>> old_val = ctx.atomic_xchg(buffer, 99, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_xchg(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_xor(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic xor at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to xor
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically XOR with rank 1's memory
            >>> old_val = ctx.atomic_xor(buffer, 0xFF, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_xor(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_and(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic and at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to and
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically AND with rank 1's memory
            >>> old_val = ctx.atomic_and(buffer, 0x0F, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_and(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_or(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic or at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to or
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically OR with rank 1's memory
            >>> old_val = ctx.atomic_or(buffer, 0xF0, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_or(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_min(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic min at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to compare and potentially store
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically compute minimum with rank 1's memory
            >>> old_val = ctx.atomic_min(buffer, 10, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_min(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_max(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic max at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to compare and potentially store
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically compute maximum with rank 1's memory
            >>> old_val = ctx.atomic_max(buffer, 100, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_max(translated_ptr, val, mask=mask, sem=sem, scope=scope)


class IrisGluon(IrisBase):
    """
    Gluon-based Iris class for multi-GPU communication and memory management.

    This class provides the same functionality as the original Iris class but
    uses Gluon's `@aggregate` decorator to encapsulate the backend state.

    Args:
        heap_size (int): Size of the symmetric heap in bytes. Default: 1GB (2^30)

    Example:
        >>> ctx = iris_gluon.iris(heap_size=2**31)  # 2GB heap
        >>> backend = ctx.get_backend()  # Get Gluon aggregate
        >>> tensor = ctx.zeros(1000, 1000, dtype=torch.float32)
    """

    def __init__(self, heap_size=1 << 30):
        # Initialize base class
        super().__init__(heap_size)

        # Initialize CCL interface
        self.ccl = CCLBase(self)

    def get_device_context(self):
        """
        Get the device context tensor for Gluon kernels.

        Returns a tensor encoding: `[cur_rank, num_ranks, heap_base_0, heap_base_1, ...]`

        Returns:
            torch.Tensor: Encoded context data as int64 tensor on device

        Example:
            >>> ctx = iris_gluon.iris()
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
        Use get_device_context() for Gluon kernels.

        Returns:
            torch.Tensor: Device context tensor
        """
        return self.get_device_context()


def iris(heap_size=1 << 30):
    """
    Create and return a Gluon-based Iris instance with the specified heap size.
    Args:
        heap_size (int): Size of the heap in bytes. Defaults to 1GB.
    Returns:
        IrisGluon: An initialized Gluon-based Iris instance
    Example:
        >>> import iris.iris_gluon as iris_gl
        >>> ctx = iris_gl.iris(2**30)  # 1GB heap
        >>> backend = ctx.get_backend()
        >>> tensor = ctx.zeros(1024, 1024)
    """
    return IrisGluon(heap_size)
