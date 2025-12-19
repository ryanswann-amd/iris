# Matmul Wrapper Consolidation

## Overview

The matmul wrapper files across different GEMM examples have been consolidated to use a common `MatmulDebugMixin` class. This reduces code duplication and makes it easier to maintain debug functionality across all examples.

## Changes Made

### Before
Each matmul_wrapper.py file (9 files total, ~1,541 lines) contained duplicated code for:
- Debug flag management (`set_debug`, `get_matmul_registers`, `get_matmul_spills`)
- Register/spill tracking logic  
- Class attributes (`_debug`, `_registers`, `_spills`)

### After  
- Created `examples/common/matmul_helpers.py` with `MatmulDebugMixin` class
- All 9 matmul_wrapper.py files now inherit from this mixin
- **Net reduction: 84 lines (5.5%)** after adding the helper file

## How to Use

### For Existing Examples

No changes needed - the matmul wrappers work exactly as before:

```python
from matmul_wrapper import matmul

# Enable debug mode
matmul.set_debug(True)

# Run matmul
result = matmul.apply(a, b, c, ...)

# Get debug info
registers = matmul.get_matmul_registers()
spills = matmul.get_matmul_spills()
```

### For New Examples

When creating a new GEMM example, use this pattern:

```python
# examples/XX_my_new_example/matmul_wrapper.py
import torch
import triton

from my_kernel import my_gemm_kernel
from examples.common.matmul_helpers import MatmulDebugMixin
import iris

gemm_kernel = my_gemm_kernel


class matmul(MatmulDebugMixin, torch.autograd.Function):
    _num_xcds = iris.hip.get_num_xcc()

    @staticmethod
    def _call(a, b, c, ...):
        # Your kernel invocation logic here
        #...
        
        kk = gemm_kernel[(grid_size,)](...)
        
        # Track debug info (replaces manual register/spill tracking)
        matmul._track_debug_info(kk)
        
        return c
    
    @staticmethod
    def forward(ctx, a, b, c, ...):
        return matmul._call(a, b, c, ...)
```

## Benefits

1. **Reduced Duplication**: Common debug functionality is now in one place
2. **Easier Maintenance**: Bug fixes and improvements only need to be made once  
3. **Consistent API**: All matmul wrappers behave identically
4. **Simpler New Examples**: Less boilerplate code to write

## Implementation Details

The `MatmulDebugMixin` provides:
- `set_debug(debug: bool)` - Enable/disable debug mode
- `get_matmul_registers()` - Get register count (debug mode only)
- `get_matmul_spills()` - Get spill count (debug mode only)
- `_track_debug_info(kernel_result)` - Internal method to track register/spill info

The mixin supports both naming conventions:
- `_registers`/`_spills` attributes
- `streamk_registers`/`streamk_spills` attributes (for backward compatibility)

## Files Modified

- `examples/common/matmul_helpers.py` (new file)
- `examples/07_gemm_all_scatter/matmul_wrapper.py`
- `examples/08_gemm_all_reduce_atomics/matmul_wrapper.py`
- `examples/09_gemm_one_shot_all_reduce/matmul_wrapper.py`
- `examples/10_gemm_all_scatter_wg_specialization/matmul_wrapper.py`
- `examples/11_gemm_all_scatter_producer_consumer/matmul_wrapper.py`
- `examples/12_gemm_all_scatter_bulk_synchronous/matmul_wrapper.py`
- `examples/15_gemm_all_reduce_ring_based/matmul_wrapper.py`
- `examples/20_gemm_all_scatter_independent/matmul_wrapper.py`
- `examples/21_gemm_one_shot_all_reduce_independent/matmul_wrapper.py`
