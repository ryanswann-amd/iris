# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
HIP Runtime Integration Module

This module provides low-level HIP runtime integration for AMD GPUs,
offering Python bindings to essential HIP runtime functions through ctypes.
It enables device management, memory operations, and inter-process communication
for multi-GPU programming.

Key Features:
- Device enumeration and management
- IPC (Inter-Process Communication) memory handles
- Device attribute queries (compute units, architecture, XCC count)
- Fine-grained and coarse-grained memory allocation
- ROCm version detection

Example:
    >>> import iris.hip as hip
    >>> num_devices = hip.count_devices()
    >>> hip.set_device(0)
    >>> cu_count = hip.get_cu_count()
"""

import ctypes
import numpy as np
import sys
import torch

rt_path = "libamdhip64.so"
hip_runtime = ctypes.cdll.LoadLibrary(rt_path)


def hip_try(err):
    """
    Check HIP error codes and raise RuntimeError if an error occurred.

    Args:
        err (int): HIP error code returned from a HIP runtime function.

    Raises:
        RuntimeError: If err is non-zero, with a descriptive error message.

    Example:
        >>> hip_try(0)  # No error, returns silently
        >>> hip_try(1)  # Raises RuntimeError with HIP error message
    """
    if err != 0:
        hip_runtime.hipGetErrorString.restype = ctypes.c_char_p
        error_string = hip_runtime.hipGetErrorString(ctypes.c_int(err)).decode("utf-8")
        raise RuntimeError(f"HIP error code {err}: {error_string}")


class hipIpcMemHandle_t(ctypes.Structure):
    """
    HIP IPC (Inter-Process Communication) memory handle structure.

    This structure represents an opaque handle used for sharing memory
    between processes on different GPUs. The handle contains 64 bytes
    of reserved data that uniquely identifies the shared memory region.

    Attributes:
        reserved (ctypes.c_char * 64): Reserved bytes containing the handle data.

    Example:
        >>> handle = hipIpcMemHandle_t()
        >>> # Use with get_ipc_handle and open_ipc_handle
    """

    _fields_ = [("reserved", ctypes.c_char * 64)]


def open_ipc_handle(ipc_handle_data, rank):
    """
    Open an IPC memory handle to access shared memory from another process.

    This function takes an IPC memory handle (obtained via get_ipc_handle) and
    opens it to allow the current process to access the shared memory region.
    The memory is opened with lazy peer access enabled.

    Args:
        ipc_handle_data (numpy.ndarray): A 64-element uint8 numpy array containing
            the IPC handle data.
        rank (int): The rank ID of the process opening the handle (used for logging/debugging).

    Returns:
        int: The pointer value (as Python int) to the opened shared memory.

    Raises:
        ValueError: If ipc_handle_data is not a 64-element uint8 numpy array.
        TypeError: If ipc_handle_data is not a numpy.ndarray.
        RuntimeError: If the HIP runtime call fails.

    Example:
        >>> # On process with rank 1, get the handle from process 0
        >>> ipc_data = all_ipc_handles[0]  # From distributed communication
        >>> ptr = open_ipc_handle(ipc_data, rank=1)
    """
    ptr = ctypes.c_void_p()
    hipIpcMemLazyEnablePeerAccess = ctypes.c_uint(1)
    hip_runtime.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        hipIpcMemHandle_t,
        ctypes.c_uint,
    ]
    if isinstance(ipc_handle_data, np.ndarray):
        if ipc_handle_data.dtype != np.uint8 or ipc_handle_data.size != 64:
            raise ValueError("ipc_handle_data must be a 64-element uint8 numpy array")
        ipc_handle_bytes = ipc_handle_data.tobytes()
        ipc_handle_data = (ctypes.c_char * 64).from_buffer_copy(ipc_handle_bytes)
    else:
        raise TypeError("ipc_handle_data must be a numpy.ndarray of dtype uint8 with 64 elements")

    raw_memory = ctypes.create_string_buffer(64)
    ctypes.memset(raw_memory, 0x00, 64)
    ipc_handle_struct = hipIpcMemHandle_t.from_buffer(raw_memory)
    ipc_handle_data_bytes = bytes(ipc_handle_data)
    ctypes.memmove(raw_memory, ipc_handle_data_bytes, 64)

    hip_try(
        hip_runtime.hipIpcOpenMemHandle(
            ctypes.byref(ptr),
            ipc_handle_struct,
            hipIpcMemLazyEnablePeerAccess,
        )
    )

    return ptr.value


def get_ipc_handle(ptr, rank):
    """
    Get an IPC memory handle for a memory pointer to share with other processes.

    This function creates an IPC handle that can be shared with other processes
    to allow them to access the memory pointed to by ptr.

    Args:
        ptr (ctypes.c_void_p): Pointer to the memory region to share.
        rank (int): The rank ID of the process creating the handle (used for logging/debugging).

    Returns:
        hipIpcMemHandle_t: An IPC memory handle that can be shared with other processes.

    Raises:
        RuntimeError: If the HIP runtime call fails.

    Example:
        >>> import ctypes
        >>> heap_ptr = ctypes.c_void_p(tensor.data_ptr())
        >>> handle = get_ipc_handle(heap_ptr, rank=0)
    """
    ipc_handle = hipIpcMemHandle_t()
    hip_try(hip_runtime.hipIpcGetMemHandle(ctypes.byref(ipc_handle), ptr))
    return ipc_handle


def count_devices():
    """
    Get the number of available HIP devices (GPUs).

    Returns:
        int: The number of HIP-capable devices available on the system.

    Raises:
        RuntimeError: If the HIP runtime call fails.

    Example:
        >>> num_gpus = count_devices()
        >>> print(f"Found {num_gpus} GPU(s)")
    """
    device_count = ctypes.c_int()
    hip_try(hip_runtime.hipGetDeviceCount(ctypes.byref(device_count)))
    return device_count.value


def set_device(gpu_id):
    """
    Set the current HIP device for subsequent operations.

    Args:
        gpu_id (int): The device ID to set as the current device (0-indexed).

    Raises:
        RuntimeError: If the HIP runtime call fails or the device ID is invalid.

    Example:
        >>> set_device(0)  # Use GPU 0
        >>> set_device(1)  # Switch to GPU 1
    """
    hip_try(hip_runtime.hipSetDevice(gpu_id))


def get_device_id():
    """
    Get the currently active HIP device ID.

    Returns:
        int: The ID of the currently active HIP device.

    Raises:
        RuntimeError: If the HIP runtime call fails.

    Example:
        >>> current_device = get_device_id()
        >>> print(f"Using GPU {current_device}")
    """
    device_id = ctypes.c_int()
    hip_try(hip_runtime.hipGetDevice(ctypes.byref(device_id)))
    return device_id.value


def get_cu_count(device_id=None):
    """
    Get the number of compute units (CUs) for a HIP device.

    Args:
        device_id (int, optional): The device ID to query. If None, uses the current device.

    Returns:
        int: The number of compute units on the specified device.

    Raises:
        RuntimeError: If the HIP runtime call fails.

    Example:
        >>> cu_count = get_cu_count()  # Current device
        >>> cu_count_gpu1 = get_cu_count(device_id=1)  # Specific device
    """
    if device_id is None:
        device_id = get_device_id()

    hipDeviceAttributeMultiprocessorCount = 63
    cu_count = ctypes.c_int()

    hip_try(hip_runtime.hipDeviceGetAttribute(ctypes.byref(cu_count), hipDeviceAttributeMultiprocessorCount, device_id))

    return cu_count.value


def get_rocm_version():
    """
    Get the installed ROCm version.

    Returns:
        tuple: A tuple of (major, minor) version numbers as integers.

    Raises:
        FileNotFoundError: If the ROCm version file is not found.
        IndexError: If the version file format is unexpected.

    Example:
        >>> major, minor = get_rocm_version()
        >>> print(f"ROCm version: {major}.{minor}")
    """
    major, minor = -1, -1
    with open("/opt/rocm/.info/version", "r") as version_file:
        version = version_file.readline().strip()
        major = int(version.split(".")[0])
        minor = int(version.split(".")[1])
    return (major, minor)


def get_wall_clock_rate(device_id):
    """
    Get the wall clock rate (GPU clock frequency) for a HIP device.

    Args:
        device_id (int): The device ID to query.

    Returns:
        int: The wall clock rate in kHz.

    Raises:
        RuntimeError: If the HIP runtime call fails.

    Example:
        >>> clock_rate = get_wall_clock_rate(0)
        >>> print(f"GPU clock rate: {clock_rate} kHz")
    """
    hipDeviceAttributeWallClockRate = 10017
    wall_clock_rate = ctypes.c_int()
    status = hip_runtime.hipDeviceGetAttribute(
        ctypes.byref(wall_clock_rate), hipDeviceAttributeWallClockRate, device_id
    )
    hip_try(status)
    return wall_clock_rate.value


def get_num_xcc(device_id=None):
    """
    Get the number of XCCs (Compute Dies) for a HIP device.

    XCC (eXtended Compute Complex) refers to the compute dies in MI300 series GPUs.
    For ROCm versions before 7.0, returns a default value of 8.

    Args:
        device_id (int, optional): The device ID to query. If None, uses the current device.

    Returns:
        int: The number of XCCs on the device.

    Raises:
        RuntimeError: If the HIP runtime call fails.

    Example:
        >>> xcc_count = get_num_xcc()
        >>> print(f"Number of XCCs: {xcc_count}")
    """
    if device_id is None:
        device_id = get_device_id()
    rocm_major, _ = get_rocm_version()
    if rocm_major < 7:
        return 8
    hipDeviceAttributeNumberOfXccs = 10018
    xcc_count = ctypes.c_int()
    hip_try(hip_runtime.hipDeviceGetAttribute(ctypes.byref(xcc_count), hipDeviceAttributeNumberOfXccs, device_id))
    return xcc_count.value
