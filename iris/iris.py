# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Iris: Multi-GPU Communication and Memory Management Framework

Iris is a high-performance framework that enables seamless multi-GPU programming in Triton,
enabling fine-grained communication and compute overlap natively in Triton
across multiple GPUs with SHMEM-like Remote Memory Access (RMA) capabilities.

Key Features:
- Symmetric heap management across multiple GPUs
- High-performance atomic operations (add, cas, xchg, xor, and, or, min, max)
- Efficient load/store operations with rank-to-rank communication
- Memory allocation and deallocation utilities
- Built-in logging with rank information
- PyTorch distributed integration for distributed computing

Example:
    >>> import iris
    >>> ctx = iris.iris(heap_size=2**30)  # 1GB heap
    >>> tensor = ctx.zeros(1024, 1024, dtype=torch.float32)
"""

import triton
import triton.language as tl

from iris._common import IrisBase, CCLBase


class Iris(IrisBase):
    """
    Main Iris class for multi-GPU communication and memory management.

    This class provides a unified interface for distributed GPU operations including
    memory allocation, atomic operations, and inter-rank communication.

    Args:
        heap_size (int): Size of the symmetric heap in bytes. Default: 1GB (2^30)

    Example:
        >>> ctx = iris.iris(heap_size=2**31)  # 2GB heap
        >>> print(f"Rank {ctx.cur_rank} of {ctx.num_ranks}") # Rank 0 of 1
        >>> tensor = ctx.zeros(1000, 1000, dtype=torch.float32)
    """

    def __init__(self, heap_size=1 << 30):
        # Initialize base class
        super().__init__(heap_size)

        # Initialize CCL interface
        self.ccl = self.CCL(self)

    def __deallocate(self, pointer):
        pass

    class CCL(CCLBase):
        """
        Collective Communication Library (CCL) interface for Iris.

        Extends CCLBase with Triton-specific all_reduce operations.
        Provides collective operations that can be called as methods on the Iris instance.
        Example usage:
            >>> shmem = iris.iris()
            >>> shmem.ccl.all_to_all(output_tensor, input_tensor)
        """

        def all_reduce_preamble(self, output_tensor, input_tensor, config=None, workspace=None):
            """
            Prepare reusable workspace for all-reduce.

            Args:
                output_tensor: Output tensor that will receive the reduced data.
                input_tensor: Input tensor providing the local contribution.
                config: Optional Config describing variant parameters.
                workspace: Optional existing workspace to update/reuse.

            Returns:
                Workspace object that can be passed to ``all_reduce``.
            """
            from iris.ccl.all_reduce import all_reduce_preamble as _all_reduce_preamble

            return _all_reduce_preamble(
                output_tensor,
                input_tensor,
                self._iris,
                config=config,
                workspace=workspace,
            )

        def all_reduce(self, output_tensor, input_tensor, config=None, async_op=False, workspace=None):
            """
            All-reduce collective operation.

            Each rank has a local input tensor, and all ranks compute the sum of all
            input tensors. The result is written to output_tensor on all ranks.

            Args:
                output_tensor: Output tensor of shape (M, N) - will contain sum of all inputs
                input_tensor: Input tensor of shape (M, N) - local rank's partial data
                config: Config instance with kernel parameters (default: None).
                        If None, uses default Config values.
                        Set config.all_reduce_variant to choose variant: "atomic", "ring", or "two_shot"
                async_op: If False, performs a barrier at the end. If True, returns immediately.
                          Default: False.
                workspace: Optional workspace prepared by ``all_reduce_preamble`` to
                           reuse internal buffers across invocations.

            Example:
                >>> shmem = iris.iris()
                >>> shmem.ccl.all_reduce(output_tensor, input_tensor)

                >>> # Custom configuration with ring variant
                >>> from iris.ccl import Config
                >>> config = Config(all_reduce_variant="ring")
                >>> shmem.ccl.all_reduce(output_tensor, input_tensor, config=config)

                >>> # Two-shot variant with block distribution
                >>> config = Config(all_reduce_variant="two_shot", all_reduce_distribution=1)
                >>> shmem.ccl.all_reduce(output_tensor, input_tensor, config=config)

                >>> # Async operation (no barrier)
                >>> shmem.ccl.all_reduce(output_tensor, input_tensor, async_op=True)
            """
            from iris.ccl.all_reduce import all_reduce as _all_reduce

            return _all_reduce(
                output_tensor,
                input_tensor,
                self._iris,
                config=config,
                async_op=async_op,
                workspace=workspace,
            )


@triton.jit
def __translate(ptr, from_rank, to_rank, heap_bases):
    from_base = tl.load(heap_bases + from_rank)
    to_base = tl.load(heap_bases + to_rank)
    # convert to int to compute difference
    ptr_int = tl.cast(ptr, tl.uint64)
    # Find the offset from from_rank heap
    offset = ptr_int - from_base
    # Byte cast for byte offset addition
    to_base_byte = tl.cast(to_base, tl.pointer_type(tl.int8))
    # Find the offset into the to_rank heap
    translated_ptr_byte = to_base_byte + offset
    # Cast to_base back to pointer type
    translated_ptr = tl.cast(translated_ptr_byte, ptr.dtype)

    # Optimization to vectorize the load/store
    # We can't do this in general because we don't know the shape of the tensor or block sizes
    # ptr = tl.max_contiguous(tl.multiple_of(ptr, (16, 16)), (16, 32))

    # 0 You can use this if your block sizes are multiples of 32.
    # Largest vectorized load instruction is dwordx4 (128-bits)
    # translated_ptr = tl.multiple_of(translated_ptr, (32, 32))
    # translated_ptr = tl.max_contiguous(translated_ptr, (1, 32))

    # ptr = tl.max_contiguous(tl.multiple_of(ptr, 512), 512)
    # translated_ptr = tl.max_contiguous(tl.multiple_of(translated_ptr, 512), 512)
    return translated_ptr


@triton.jit
def load(pointer, to_rank, from_rank, heap_bases, mask=None):
    """
    Loads a value from the specified rank's memory location.

    This function performs a memory read operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and loading
    data from the target memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local load operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the pointer will be translated. Must be the current rank where the pointer is local.
        from_rank (int): The rank ID from which to read the data.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not load the data at address pointer[idx]. Defaults to None.

    Returns:
        Block: The loaded value from the target memory location.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Load data from rank 1's memory into the current rank
        >>>     cur_rank = 0      # Current rank
        >>>     remote_rank = 1   # Remote rank to load from
        >>>     data = iris.load(ptr, cur_rank, remote_rank, heap_bases)
        >>>     return data
    """
    translated_ptr = __translate(pointer, to_rank, from_rank, heap_bases)
    result = tl.load(translated_ptr, mask=mask)
    return result


@triton.jit
def store(pointer, value, from_rank, to_rank, heap_bases, mask=None):
    """
    Writes data to the specified rank's memory location.

    This function performs a memory write operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and storing
    the provided data to the target memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local store operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        value (Block): The tensor of elements to be stored.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the data will be written.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not store the data at address pointer[idx]. Defaults to None.

    Returns:
        None

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Store value 42 into rank 1's heap from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     value = 42
        >>>     iris.store(ptr, value, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    tl.store(translated_ptr, value, mask=mask)


@triton.jit
def copy(src_ptr, dst_ptr, from_rank, to_rank, cur_rank, heap_bases, mask=None):
    """
    Copies data from the specified rank's memory into the destination rank's memory.
    This function performs the transfer by translating `src_ptr` from the `from_rank`'s address
    space to the `to_rank`'s address space, performing a masked load from the translated
    source, and storing the loaded data to `dst_ptr` in the `to_rank` memory location.
    If `from_rank` and `to_rank` are the same, this function performs a local copy operation.
    It is undefined behaviour if neither `from_rank` nor `to_rank` is the `cur_rank`.

    Args:
        src_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the `from_rank`'s local memory from which to read data.
        dst_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the `to_rank`'s local memory where the data will be written.
        from_rank (int): The rank ID that owns `src_ptr` (source rank).
        to_rank (int): The rank ID that will receive the data (destination rank).
        cur_rank (int): The rank ID issuing the copy operation. Must be either `from_rank` or `to_rank`.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not load from the translated src_ptr[idx] and do not store to dst_ptr[idx]. Defaults to None.

    Returns:
        None

    Example:
        >>> @triton.jit
        >>> def kernel(remote_ptr, local_ptr, heap_bases):
        >>>     from_rank = 1
        >>>     to_rank = 0
        >>>     iris.copy(remote_ptr, local_ptr, from_rank, to_rank, to_rank, heap_bases)
    """

    cur_base = tl.load(heap_bases + cur_rank)

    from_base = tl.load(heap_bases + from_rank)
    to_base = tl.load(heap_bases + to_rank)

    src_ptr_int = tl.cast(src_ptr, tl.uint64)
    src_offset = src_ptr_int - cur_base

    dst_ptr_int = tl.cast(dst_ptr, tl.uint64)
    dst_offset = dst_ptr_int - cur_base

    from_base_byte = tl.cast(from_base, tl.pointer_type(tl.int8))
    to_base_byte = tl.cast(to_base, tl.pointer_type(tl.int8))

    translated_src = tl.cast(from_base_byte + src_offset, src_ptr.dtype)
    translated_dst = tl.cast(to_base_byte + dst_offset, src_ptr.dtype)

    data = tl.load(translated_src, mask=mask)
    tl.store(translated_dst, data, mask=mask)


@triton.jit
def get(from_ptr, to_ptr, from_rank, to_rank, heap_bases, mask=None):
    """
    Copies data from the specified rank's memory to the current rank's local memory.

    This function performs a memory read operation by translating the `from_ptr`
    from the current rank's address space to the `from_rank`'s address space, loading data
    from the `from_rank` memory location, and storing it to the local `to_ptr`.
    If the `from_rank` is the same as the current rank, this function performs a local copy operation.

    Args:
        from_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that will be translated to the `from_rank`'s address space. Must be the current rank where the pointer is local.
        to_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's local memory where the data will be stored.
        from_rank (int): The `from_rank` ID from which to read the data.
        to_rank (int): The current rank ID where the data will be stored.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not load the data at address from_ptr[idx] and do not store to to_ptr[idx]. Defaults to None.

    Returns:
        None

    Example:
        >>> @triton.jit
        >>> def kernel(remote_ptr, local_ptr, heap_bases):
        >>>     from_rank = 1
        >>>     to_rank = 0
        >>>     iris.get(remote_ptr, local_ptr, from_rank, to_rank, heap_bases)
    """
    translated_from_ptr = __translate(from_ptr, from_rank, to_rank, heap_bases)

    data = tl.load(translated_from_ptr, mask=mask)

    tl.store(to_ptr, data, mask=mask)


@triton.jit
def put(from_ptr, to_ptr, from_rank, to_rank, heap_bases, mask=None):
    """
    Copies data from the current rank's local memory to the specified rank's memory.
    This function performs a memory write operation by loading data from the current
    rank's `from_ptr`, translating the `to_ptr` from the current rank's address
    space to the `to_rank`'s address space, and storing the data to the `to_rank` memory location.
    If the `to_rank` is the same as the current rank, this function performs a local copy operation.

    Args:
        from_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's local memory from which to read data.
        to_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        from_rank (int): The current rank ID from which to read the data.
        to_rank (int): The `to_rank` ID to which the data will be written.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not load the data at address from_ptr[idx] and do not store to to_ptr[idx]. Defaults to None.

    Returns:
        None

    Example:
        >>> @triton.jit
        >>> def kernel(local_ptr, remote_ptr, heap_bases):
        >>>     from_rank = 0
        >>>     to_rank = 1
        >>>     iris.put(local_ptr, remote_ptr, from_rank, to_rank, heap_bases)
    """
    translated_to_ptr = __translate(to_ptr, from_rank, to_rank, heap_bases)

    data = tl.load(from_ptr, mask=mask)

    tl.store(translated_to_ptr, data, mask=mask)


@triton.jit
def atomic_add(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
    """
    Performs an atomic add at the specified rank's memory location.

    This function performs an atomic addition operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    adding the provided data to the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic addition operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically add 5 to rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     increment = 5
        >>>     old_val = iris.atomic_add(ptr, increment, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_add(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_sub(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
    """
    Atomically subtracts data from the specified rank's memory location.

    This function performs an atomic subtraction operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
        subtracting the provided data from the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic subtraction operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block): The tensor of elements to be subtracted atomically.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". Defaults to "acq_rel".
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). Defaults to "gpu".

    Returns:
        Block: The value at the memory location before the atomic subtraction.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically subtract 3 from rank 2's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 2   # Remote rank (destination)
        >>>     decrement = 3
        >>>     old_val = iris.atomic_sub(ptr, decrement, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_sub(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_cas(pointer, cmp, val, from_rank, to_rank, heap_bases, sem=None, scope=None):
    """
    Atomically compares and exchanges the specified rank's memory location.

    This function performs an atomic compare-and-swap operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    comparing the current value with the expected value, then writing the new value if they match.
    If the `from_rank` and `to_rank` are the same, this function performs a local atomic compare-and-swap operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        cmp (Block): The expected value to be compared with the current value at the memory location.
        val (Block): The new value to be written if the compare succeeds.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". Defaults to "acq_rel".
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). Defaults to "gpu".

    Returns:
        Block: The value contained at the memory location before the atomic operation attempt.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Compare-and-swap on rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     expected = 0
        >>>     new_val = 42
        >>>     old_val = iris.atomic_cas(ptr, expected, new_val, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_cas(translated_ptr, cmp, val, sem=sem, scope=scope)


@triton.jit
def atomic_xchg(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
    """
    Performs an atomic exchange at the specified rank's memory location.

    This function performs an atomic exchange operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    exchanging the current value with the provided new value. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic exchange operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Exchange value with rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     new_value = 99
        >>>     old_val = iris.atomic_xchg(ptr, new_value, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_xchg(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_xor(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
    """
    Performs an atomic xor at the specified rank's memory location.

    This function performs an atomic xor operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    xoring the provided data to the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic xor operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically XOR with rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     mask_val = 0xFF
        >>>     old_val = iris.atomic_xor(ptr, mask_val, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_xor(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_and(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
    """
    Performs an atomic and at the specified rank's memory location.

    This function performs an atomic and operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    anding the provided data to the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic and operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically AND with rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     mask_val = 0x0F
        >>>     old_val = iris.atomic_and(ptr, mask_val, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_and(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_or(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
    """
    Performs an atomic or at the specified rank's memory location.

    This function performs an atomic or operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    oring the provided data to the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic or operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically OR with rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     mask_val = 0xF0
        >>>     old_val = iris.atomic_or(ptr, mask_val, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_or(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_min(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
    """
    Performs an atomic min at the specified rank's memory location.

    This function performs an atomic min operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    performing the min on the provided data to the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic min operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically find minimum with rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     new_val = 10
        >>>     old_val = iris.atomic_min(ptr, new_val, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_min(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_max(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
    """
    Performs an atomic max at the specified rank's memory location.

    This function performs an atomic max operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    performing the max on the provided data to the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic max operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically find maximum with rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     new_val = 100
        >>>     old_val = iris.atomic_max(ptr, new_val, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_max(translated_ptr, val, mask=mask, sem=sem, scope=scope)


def iris(heap_size=1 << 30):
    """
    Create and return an Iris instance with the specified heap size.

    Args:
        heap_size (int): Size of the heap in bytes. Defaults to 1GB.

    Returns:
        Iris: An initialized Iris instance.

    Example:
        >>> import iris
        >>> iris_ctx = iris.iris(2**30)  # 1GB heap
        >>> tensor = iris_ctx.zeros(1024, 1024)
    """
    return Iris(heap_size)
