# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
GPU Array wrapper with __cuda_array_interface__ support
"""

class GpuArrayWrapper:
    """Wrapper that exposes __cuda_array_interface__ for vmem allocations"""
    
    def __init__(self, data_ptr, shape, dtype_str, device_id=0):
        """
        Args:
            data_ptr: Integer pointer to GPU memory
            shape: Tuple of dimensions
            dtype_str: String like 'float32', 'int64', etc.
            device_id: GPU device ID
        """
        self._data_ptr = data_ptr
        self._shape = shape if isinstance(shape, tuple) else (shape,)
        self._dtype_str = dtype_str
        self._device_id = device_id
        
        # Map dtype to typestr for CAI (CUDA Array Interface v3).
        # Note: torch bfloat16 is not representable directly via CAI typestr in this build;
        # we handle bfloat16 by exporting as uint16 (<u2) and viewing as bfloat16 in Python.
        self._dtype_map = {
            'bool': '|b1',
            'int8': '|i1',
            'uint8': '|u1',
            'int16': '<i2',
            'uint16': '<u2',
            'float32': '<f4',
            'float64': '<f8',
            'float16': '<f2',
            'int32': '<i4',
            'int64': '<i8',
            'uint32': '<u4',
            'uint64': '<u8',
            # bfloat16 exported as uint16 (see note above)
            'bfloat16': '<u2',
            'bf16': '<u2',
        }
    
    @property
    def __cuda_array_interface__(self):
        """CUDA Array Interface v3"""
        return {
            'version': 3,
            'shape': self._shape,
            'typestr': self._dtype_map[self._dtype_str],
            'data': (self._data_ptr, False),  # (ptr, readonly)
            'strides': None,  # C-contiguous
            'descr': [('', self._dtype_map[self._dtype_str])],
        }
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def dtype(self):
        return self._dtype_str
    
    @property
    def device(self):
        return self._device_id
