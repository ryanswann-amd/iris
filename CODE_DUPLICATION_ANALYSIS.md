# Code Duplication Analysis and Improvement Suggestions

**Date:** December 19, 2025  
**Repository:** ROCm/iris  
**Objective:** Identify code duplication and suggest improvements to reduce code size

---

## Executive Summary

This analysis identified significant code duplication across multiple areas of the Iris codebase, with opportunities to reduce code size by an estimated **30-40%** in affected areas. The main areas of duplication are:

1. **Example benchmark files** (~3,400 lines total, 85-97% similarity)
2. **Example matmul_wrapper files** (~1,600 lines total, 94-100% similarity)
3. **Test files for atomic operations** (~16 test files, 86-88% similarity)
4. **Test files for tensor creation** (~6 test files, 61-72% similarity)
5. **Core atomic operations in iris.py** (~9 functions with nearly identical structure)
6. **Tensor creation methods in iris.py** (~10 methods with highly repetitive boilerplate)

---

## 1. Example Benchmark Files Duplication

### Findings

**Total Files:** 10 benchmark.py files in examples/  
**Total Lines:** ~3,400 lines  
**Similarity:** 85-97% between related examples

#### High Similarity Pairs:
- `11_gemm_all_scatter_producer_consumer` ↔ `12_gemm_all_scatter_bulk_synchronous`: **97.2%**
- `10_gemm_all_scatter_wg_specialization` ↔ `07_gemm_all_scatter`: **91.5%**
- `08_gemm_all_reduce_atomics` ↔ `09_gemm_one_shot_all_reduce`: **86.8%**
- `20_gemm_all_scatter_independent` ↔ `21_gemm_one_shot_all_reduce_independent`: **86.1%**

### Duplicated Code Patterns

All benchmark files share:
1. **Import statements** (95% identical)
2. **parse_args() function** (80-90% identical with minor parameter variations)
3. **_worker() function setup** (90-95% identical)
4. **Distributed initialization** (100% identical)
5. **Iris initialization** (100% identical)
6. **SM/CU count detection logic** (70-80% identical)
7. **Datatype parsing** (100% identical)
8. **Validation and benchmarking logic** (60-70% identical)
9. **JSON logging setup** (90-100% identical)
10. **Main function structure** (85-95% identical)

### Improvement Suggestions

**Priority: HIGH**

1. **Create a common benchmark base class/module:**
   ```python
   # examples/common/benchmark_base.py
   class BenchmarkBase:
       def __init__(self, heap_size):
           self.shmem = iris.iris(heap_size)
           self.rank = self.shmem.get_rank()
           self.world_size = self.shmem.get_num_ranks()
       
       def parse_common_args(self):
           # Common argument parsing
           pass
       
       def setup_distributed(self, local_rank, world_size, init_url):
           # Common distributed setup
           pass
       
       def detect_compute_units(self):
           # Common CU detection logic
           pass
       
       def parse_datatype(self, dtype_str):
           # Common datatype parsing
           pass
   ```

2. **Extract common worker function:**
   - Create `examples/common/worker_utils.py` with reusable worker setup
   - Each example only needs to override example-specific logic

3. **Standardize argument parsing:**
   - Create base parser with common arguments
   - Each example extends with example-specific arguments

**Estimated Reduction:** 2,000-2,500 lines (60-75% of duplicated code)

---

## 2. Example Matmul Wrapper Files Duplication

### Findings

**Total Files:** 9 matmul_wrapper.py files  
**Total Lines:** ~1,600 lines  
**Similarity:** 94-100% between related examples

#### High Similarity Pairs:
- `20_gemm_all_scatter_independent` ↔ `12_gemm_all_scatter_bulk_synchronous`: **100%**
- `20_gemm_all_scatter_independent` ↔ `11_gemm_all_scatter_producer_consumer`: **97.6%**
- `11_gemm_all_scatter_producer_consumer` ↔ `12_gemm_all_scatter_bulk_synchronous`: **97.6%**
- `10_gemm_all_scatter_wg_specialization` ↔ `07_gemm_all_scatter`: **96.0%**

### Duplicated Code Patterns

All matmul_wrapper files share:
1. **Class structure** (matmul as torch.autograd.Function)
2. **Debug flag management** (100% identical)
3. **Register/spills getter methods** (100% identical)
4. **_call() method structure** (90-95% identical)
5. **Kernel invocation setup** (85-90% identical)
6. **Forward() method** (95-100% identical)

### Key Differences
Only the imported kernel function varies:
- `from gemm_all_scatter import persistent_gemm_all_scatter`
- `from gemm_all_reduce_atomics import persistent_gemm_all_reduce`
- etc.

### Improvement Suggestions

**Priority: HIGH**

1. **Create a unified matmul wrapper:**
   ```python
   # examples/common/matmul_wrapper.py
   class MatmulWrapper(torch.autograd.Function):
       def __init__(self, kernel_func):
           self.kernel = kernel_func
           self._debug = False
           # ... common implementation
       
       @staticmethod
       def _call(kernel, a, b, c, ...):
           # Common implementation
           kk = kernel[(num_sms,)](...)
           return c
   ```

2. **Each example only needs:**
   ```python
   # examples/07_gemm_all_scatter/matmul_wrapper.py
   from examples.common.matmul_wrapper import MatmulWrapper
   from gemm_all_scatter import persistent_gemm_all_scatter
   
   matmul = MatmulWrapper(persistent_gemm_all_scatter)
   ```

**Estimated Reduction:** 1,400-1,500 lines (90-95% of duplicated code)

---

## 3. Atomic Operation Test Files Duplication

### Findings

**Total Files:** 16 test files (8 Gluon + 8 Triton)  
**Similarity:**
- Gluon atomic tests: **88.1%** similar
- Triton atomic tests: **86.5%** similar

**Test files:**
- `test_atomic_add_gluon.py` / `test_atomic_add_triton.py`
- `test_atomic_and_gluon.py` / `test_atomic_and_triton.py`
- `test_atomic_cas_gluon.py` / `test_atomic_cas_triton.py`
- `test_atomic_max_gluon.py` / `test_atomic_max_triton.py`
- `test_atomic_min_gluon.py` / `test_atomic_min_triton.py`
- `test_atomic_or_gluon.py` / `test_atomic_or_triton.py`
- `test_atomic_xchg_gluon.py` / `test_atomic_xchg_triton.py`
- `test_atomic_xor_gluon.py` / `test_atomic_xor_triton.py`

### Duplicated Code Patterns

All atomic test files share:
1. **Kernel structure** (95% identical, only operation name differs)
2. **Test function structure** (90% identical)
3. **Parametrize decorators** (90-100% identical for most operations)
4. **Setup code** (100% identical)
5. **Validation logic** (80-90% identical)

### Improvement Suggestions

**Priority: MEDIUM**

1. **Create a parameterized test framework:**
   ```python
   # tests/unittests/test_atomic_operations.py
   import pytest
   
   ATOMIC_OPS = ['add', 'and', 'or', 'xor', 'min', 'max', 'xchg', 'cas']
   
   @pytest.mark.parametrize("operation", ATOMIC_OPS)
   @pytest.mark.parametrize("backend", ["gluon", "triton"])
   def test_atomic_operation(operation, backend, dtype, sem, scope, BLOCK_SIZE):
       # Unified test logic that handles all atomic operations
       # Select appropriate kernel and validation based on operation
       pass
   ```

2. **Create atomic test utilities:**
   ```python
   # tests/unittests/atomic_test_utils.py
   def create_atomic_kernel(operation, backend):
       # Factory function to create kernel for any atomic operation
       pass
   
   def validate_atomic_result(operation, initial, num_ranks):
       # Common validation logic
       pass
   ```

**Estimated Reduction:** 8-10 test files can be consolidated into 1-2 files

---

## 4. Tensor Creation Test Files Duplication

### Findings

**Total Files:** 6 major tensor creation test files  
**Similarity:** 61-72% between files

**Files:**
- `test_zeros.py` ↔ `test_ones.py`: **72.4%**
- `test_zeros.py` ↔ `test_empty.py`: **69.9%**
- `test_ones.py` ↔ `test_empty.py`: **69.2%**
- `test_randn.py` ↔ `test_rand.py`: **67.2%**

### Duplicated Code Patterns

All tensor creation test files share:
1. **Test structure** (parametrize decorators, test functions)
2. **Size testing patterns** (scalar, 1D, 2D, 3D, etc.)
3. **Dtype testing** (fp16, fp32, bf16, int32, etc.)
4. **Device testing**
5. **Layout testing**
6. **Out parameter testing**
7. **requires_grad testing**
8. **Error handling tests**

### Improvement Suggestions

**Priority: MEDIUM**

1. **Create a base test class:**
   ```python
   # tests/unittests/tensor_creation_base.py
   class TensorCreationTestBase:
       def run_creation_test(self, method_name, *args, **kwargs):
           # Common test logic for all creation methods
           pass
       
       def validate_tensor(self, tensor, expected_shape, expected_dtype):
           # Common validation
           pass
   ```

2. **Parameterize by method:**
   ```python
   # tests/unittests/test_tensor_creation.py
   @pytest.mark.parametrize("method", ["zeros", "ones", "randn", "rand", "empty"])
   def test_tensor_creation_basic(method, size, dtype):
       # Unified test for all creation methods
       pass
   ```

**Estimated Reduction:** 6 test files → 2 test files (with shared base)

---

## 5. Atomic Operations in iris.py

### Findings

**Total Functions:** 9 atomic operations  
**Lines per function:** ~33 lines (excluding docstrings)  
**Total Lines:** ~300 lines for atomic operations

**Functions:**
- `atomic_add`, `atomic_sub`, `atomic_xchg`
- `atomic_xor`, `atomic_and`, `atomic_or`
- `atomic_min`, `atomic_max`, `atomic_cas`

### Duplicated Pattern

**Every atomic function follows this identical pattern:**
```python
@triton.jit
def atomic_<op>(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
    """Docstring"""
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_<op>(translated_ptr, val, mask=mask, sem=sem, scope=scope)
```

**Only difference:** The triton atomic operation name (`tl.atomic_add`, `tl.atomic_xor`, etc.)

### Improvement Suggestions

**Priority: LOW (readability vs. reduction tradeoff)**

1. **Create a factory function:**
   ```python
   def _create_atomic_op(op_name):
       """Factory to create atomic operation wrappers."""
       def atomic_op(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
           translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
           triton_op = getattr(tl, f'atomic_{op_name}')
           return triton_op(translated_ptr, val, mask=mask, sem=sem, scope=scope)
       return atomic_op
   
   # Generate all atomic operations
   for op in ['add', 'sub', 'xchg', 'xor', 'and', 'or', 'min', 'max', 'cas']:
       globals()[f'atomic_{op}'] = triton.jit(_create_atomic_op(op))
   ```

2. **Keep current approach but add comment:**
   - The explicit functions provide better IDE support and documentation
   - Duplication is minimal and highly regular
   - **Recommendation:** Keep as-is for maintainability

**Estimated Reduction:** ~200 lines if consolidated, but NOT RECOMMENDED due to:
- Loss of individual docstrings
- Reduced IDE support
- Harder to debug
- Reduced code clarity

---

## 6. Tensor Creation Methods in iris.py

### Findings

**Total Methods:** 10 tensor creation methods  
**Common patterns:** All 10 methods share 10 common code patterns

**Methods:**
- `zeros`, `ones`, `empty`, `full`
- `randn`, `rand`, `randint`
- `linspace`, `arange`, `uniform`, `zeros_like`

### Duplicated Code Patterns

**Every method follows this structure (with minor variations):**

```python
def <method>(self, *size, **kwargs):
    # 1. Debug logging
    self.debug(f"<method>: ...")
    
    # 2. Default dtype/device handling
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = self.device
    
    # 3. Device validation
    self.__throw_if_invalid_device(device)
    
    # 4. Size parsing
    size, num_elements = self.__parse_size(size)
    
    # 5. Output tensor validation (if provided)
    if out is not None:
        self.__throw_if_invalid_output_tensor(out, num_elements, dtype)
        tensor = out.view(size)
    else:
        # 6. Memory allocation
        tensor = self.__allocate(num_elements=num_elements, dtype=dtype)
        # 7. Data initialization (method-specific)
        tensor.fill_(value) / tensor.zero_() / etc.
        # 8. Reshaping
        tensor = tensor.reshape(size)
    
    # 9. Layout application
    tensor = self.__apply_layout(tensor, layout)
    
    # 10. requires_grad handling
    if requires_grad:
        tensor.requires_grad_()
    
    return tensor
```

### Improvement Suggestions

**Priority: MEDIUM**

1. **Extract common boilerplate into a base method:**
   ```python
   def _create_tensor_base(self, size, dtype=None, device=None, out=None, 
                           layout=torch.strided, requires_grad=False,
                           initializer=None, **init_kwargs):
       """Base method for all tensor creation functions.
       
       Args:
           initializer: Function to initialize the tensor (e.g., lambda t: t.zero_())
       """
       # Common boilerplate (steps 1-6, 8-10)
       self.debug(f"Creating tensor: size={size}, dtype={dtype}")
       
       if dtype is None:
           dtype = torch.get_default_dtype()
       if device is None:
           device = self.device
       
       self.__throw_if_invalid_device(device)
       size, num_elements = self.__parse_size(size)
       
       if out is not None:
           self.__throw_if_invalid_output_tensor(out, num_elements, dtype)
           tensor = out.view(size)
       else:
           tensor = self.__allocate(num_elements=num_elements, dtype=dtype)
           tensor = tensor.reshape(size)
       
       # Initialize (method-specific step 7)
       if initializer:
           initializer(tensor, **init_kwargs)
       
       tensor = self.__apply_layout(tensor, layout)
       
       if requires_grad:
           tensor.requires_grad_()
       
       return tensor
   ```

2. **Simplify tensor creation methods:**
   ```python
   def zeros(self, *size, **kwargs):
       """Returns a tensor filled with zeros."""
       return self._create_tensor_base(
           size, 
           initializer=lambda t: t.zero_(),
           **kwargs
       )
   
   def ones(self, *size, **kwargs):
       """Returns a tensor filled with ones."""
       return self._create_tensor_base(
           size,
           initializer=lambda t: t.fill_(1),
           **kwargs
       )
   
   def randn(self, *size, **kwargs):
       """Returns a tensor filled with random normal values."""
       def init_randn(tensor, generator=None, device=None, dtype=None):
           random_data = torch.randn(
               tensor.numel(), 
               generator=generator, 
               dtype=dtype, 
               device=device
           )
           tensor.copy_(random_data)
       
       return self._create_tensor_base(
           size,
           initializer=init_randn,
           **kwargs
       )
   ```

3. **Benefits:**
   - Reduces ~150-200 lines of boilerplate code
   - Ensures consistency across all methods
   - Easier to maintain and update
   - Centralized error handling

**Estimated Reduction:** 150-200 lines (10-15% of current implementation)

---

## 7. Additional Duplication Patterns

### Copy/Get/Put Operations in iris.py

**Functions:** `copy`, `get`, `put`, `load`, `store`  
**Pattern:** All follow similar pointer translation → operation pattern

**Suggestion:** Already well-factored with `__translate` helper. No changes needed.

### Logging Methods in iris.py

**Methods:** `debug`, `info`, `warning`, `error`  
**Pattern:** All call `_log_with_rank` with different log levels

**Current implementation:** Already optimized with single helper method. No changes needed.

---

## Summary of Recommendations

| Category | Priority | Files Affected | Estimated Reduction | Implementation Effort |
|----------|----------|----------------|---------------------|----------------------|
| Example Benchmarks | HIGH | 10 files | 2,000-2,500 lines | Medium |
| Matmul Wrappers | HIGH | 9 files | 1,400-1,500 lines | Low |
| Atomic Tests | MEDIUM | 16 files | 8-10 files → 1-2 files | Medium |
| Tensor Tests | MEDIUM | 6 files | 6 files → 2 files | Medium |
| Tensor Creation Methods | MEDIUM | iris.py | 150-200 lines | Medium |
| Atomic Operations | LOW | iris.py | Not recommended | N/A |

### Total Estimated Code Reduction

**Conservative estimate:** 4,000-5,000 lines of code (30-35% of duplicated code)  
**Optimistic estimate:** 5,000-6,000 lines of code (35-40% of duplicated code)

---

## Implementation Priority Order

### Phase 1 (High Priority - Quick Wins)
1. **Matmul Wrappers** - 90% duplication, simple refactoring
2. **Example Benchmarks** - Large impact, moderate effort

### Phase 2 (Medium Priority - Test Consolidation)
3. **Atomic Operation Tests** - Improves test maintainability
4. **Tensor Creation Tests** - Reduces test code duplication

### Phase 3 (Medium Priority - Core Library)
5. **Tensor Creation Methods** - Improves core library maintainability

### Not Recommended
6. **Atomic Operations in iris.py** - Readability/documentation trade-off not worth it

---

## Maintenance Benefits

Beyond code size reduction, these improvements provide:

1. **Easier Updates:** Changes to common patterns only need to be made once
2. **Consistency:** Ensures all examples/tests follow the same patterns
3. **Reduced Bugs:** Less code means fewer places for bugs to hide
4. **Better Onboarding:** New contributors have clear patterns to follow
5. **Faster Development:** New examples/tests can be created by extending base classes

---

## Conclusion

The Iris codebase has significant opportunities for code size reduction through:
- **Abstraction of common patterns** (benchmarks, wrappers, tests)
- **Base class/utility creation** (shared functionality)
- **Consolidation of highly similar files** (tests, wrappers)

The recommendations prioritize high-impact, low-effort changes first, with estimated reductions of 30-40% in affected areas. Implementation should be phased to minimize disruption and allow for testing at each stage.
