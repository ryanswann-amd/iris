# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Common utilities for matmul wrappers in Iris GEMM examples.

This module provides shared helper functions and a mixin class that can be used
to reduce code duplication across matmul wrapper implementations.
"""

import torch
from examples.common.utils import is_triton_interpret_set


class MatmulDebugMixin:
    """
    Mixin class providing debug functionality for matmul wrappers.
    
    This can be mixed into torch.autograd.Function subclasses to add
    standardized debug flag management and register/spill tracking.
    
    Usage:
        class matmul(MatmulDebugMixin, torch.autograd.Function):
            # ...your implementation...
            pass
    """
    
    _debug = False
    _registers = None
    _spills = None
    
    @classmethod
    def set_debug(cls, debug: bool):
        """Enable or disable debug mode for register/spill tracking."""
        cls._debug = debug
        # Initialize streamk attributes for backward compatibility with some examples
        if not hasattr(cls, 'streamk_registers'):
            cls.streamk_registers = 0
            cls.streamk_spills = 0
    
    @classmethod
    def get_matmul_registers(cls):
        """Get the number of registers used by the kernel (debug mode only)."""
        if cls._debug:
            # Support both naming conventions
            if cls._registers is not None:
                return cls._registers
            elif hasattr(cls, 'streamk_registers'):
                return cls.streamk_registers
            return 0
        else:
            raise RuntimeError("Debug mode is not enabled. Call set_debug(True) first.")
    
    @classmethod
    def get_matmul_spills(cls):
        """Get the number of register spills in the kernel (debug mode only)."""
        if cls._debug:
            # Support both naming conventions
            if cls._spills is not None:
                return cls._spills
            elif hasattr(cls, 'streamk_spills'):
                return cls.streamk_spills
            return 0
        else:
            raise RuntimeError("Debug mode is not enabled. Call set_debug(True) first.")
    
    @classmethod
    def _track_debug_info(cls, kernel_result):
        """
        Track register and spill information from kernel execution.
        
        Call this after kernel invocation to store debug info if debug mode is enabled.
        
        Args:
            kernel_result: The kernel object returned from kernel invocation
        """
        if cls._debug and not is_triton_interpret_set():
            cls._registers = kernel_result.n_regs
            cls._spills = kernel_result.n_spills
            # Also update streamk_ attributes if they exist
            if hasattr(cls, 'streamk_registers'):
                cls.streamk_registers = kernel_result.n_regs
                cls.streamk_spills = kernel_result.n_spills
