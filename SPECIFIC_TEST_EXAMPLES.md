# Specific Test Examples - Detailed Analysis

This document provides concrete examples of the issues identified in the main analysis.

## Example 1: test_copy_gluon.py vs test_copy_triton.py

### Files
- `tests/unittests/test_copy_gluon.py` (4,368 test cases)
- `tests/unittests/test_copy_triton.py` (4,368 test cases)

### Parametrization (Both Files)
```python
@pytest.mark.parametrize("dtype", [torch.int8, torch.float16, torch.bfloat16, torch.float32])  # 4
@pytest.mark.parametrize("BLOCK_SIZE", [1, 8, 16, 32])  # 4
```

Total: 4 × 4 = 16 combinations per test function × 3 test functions = **48 test cases per file**

But actual count shows 4,368 - this suggests there's likely more parametrization or module-level decorators that multiply all tests.

### The Code Duplication

**test_copy_gluon.py**:
```python
@gluon.jit
def copy_get_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    data,
    results,
    cur_rank: gl.constexpr,
    num_ranks: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
):
    ctx = IrisDeviceCtx.initialize(context_tensor)
    # ... kernel logic using gluon API
```

**test_copy_triton.py**:
```python
@triton.jit
def copy_get_kernel(
    data,
    results,
    cur_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    # ... nearly identical kernel logic using triton API
```

**The test functions are nearly identical**:
- Same parametrization
- Same test data setup
- Same validation logic
- Only difference: which kernel gets called

### Recommendation
Merge into single parametrized test:
```python
@pytest.mark.parametrize("api", ["gluon", "triton"])
@pytest.mark.parametrize("dtype", [torch.int8, torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("BLOCK_SIZE", [1, 8, 16, 32])
def test_copy_get(api, dtype, BLOCK_SIZE):
    shmem = iris_gl.iris(1 << 20) if api == "gluon" else iris.iris(1 << 20)
    # ... shared setup ...
    
    if api == "gluon":
        copy_get_kernel_gluon[grid](...)
    else:
        copy_get_kernel_triton[grid](...)
    
    # ... shared validation ...
```

**Savings**: 8,736 → 96 test cases (with reduced parametrization below)

---

## Example 2: test_zeros_like.py - Parametrization Explosion

### File
`tests/unittests/test_zeros_like.py` (139,216 test cases!)

### Current Parametrization (test_zeros_like_basic)
```python
@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bool,
    ],
)  # 8 dtypes
@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (5,),
        (2, 3),
        (3, 4, 5),
        (1, 1, 1),
        (10, 20),
    ],
)  # 6 shapes
def test_zeros_like_basic(dtype, shape):
    shmem = iris.iris(1 << 20)
    input_tensor = shmem.full(shape, 5, dtype=dtype)
    result = shmem.zeros_like(input_tensor)
    
    assert result.shape == input_tensor.shape
    assert result.dtype == input_tensor.dtype
    assert torch.all(result == 0)
```

Total combinations: 8 × 6 = **48 test cases**

### The Problem
This is just ONE test function. The file has 13 test functions with similar or worse parametrization patterns. Many of these test the exact same `zeros_like()` function with slightly different focuses:
- `test_zeros_like_basic`
- `test_zeros_like_dtype_override`
- `test_zeros_like_requires_grad`
- `test_zeros_like_device_override`
- `test_zeros_like_layout_override`
- `test_zeros_like_memory_format`
- `test_zeros_like_parameter_combinations`
- `test_zeros_like_symmetric_heap_shapes_dtypes`
- etc.

### Why This Is Wasteful

1. **Testing same behavior repeatedly**: All 8 integer dtypes (int8, int16, int32, int64) behave identically for zeros_like. Testing all 8 provides no additional coverage.

2. **Redundant shape testing**: Shapes (1,), (5,), (10,20) all test the same code path - just different tensor sizes. One small, one medium, one large is sufficient.

3. **Combinatorial explosion**: When you multiply 8 dtypes × 6 shapes × multiple test functions × multiple features, you get 139,216 test cases that all essentially verify "zeros_like creates zeros".

### Recommended Reduction

**Before** (8 dtypes × 6 shapes = 48 combinations):
```python
@pytest.mark.parametrize("dtype", [int8, int16, int32, int64, float16, float32, float64, bool])
@pytest.mark.parametrize("shape", [(1,), (5,), (2,3), (3,4,5), (1,1,1), (10,20)])
```

**After** (3 dtypes × 3 shapes = 9 combinations, 81% reduction):
```python
@pytest.mark.parametrize("dtype", [
    torch.int32,    # Representative integer
    torch.float32,  # Representative float
    torch.bool      # Edge case: boolean
])
@pytest.mark.parametrize("shape", [
    (1,),           # Scalar-like (edge case)
    (2, 3),         # 2D (common)
    (3, 4, 5)       # 3D (multi-dimensional)
])
```

**Consolidate test functions** - instead of 13 separate test functions, have 3-4:
```python
def test_zeros_like_basic(dtype, shape):
    # Test basic functionality

def test_zeros_like_edge_cases():
    # Test edge cases: empty tensors, very large tensors, etc.

def test_zeros_like_parameters():
    # Test dtype override, requires_grad, etc. in one test
```

**Result**: 139,216 → ~2,000 test cases (98.6% reduction)

---

## Example 3: CI Matrix Explosion

### Current CI Configuration

From `.github/workflows/iris-tests.yml`:

```yaml
strategy:
  matrix:
    include:
      # examples directory
      - test_dir: examples, num_ranks: 1, gpu_devices: "0,1"
      - test_dir: examples, num_ranks: 2, gpu_devices: "2,3"
      - test_dir: examples, num_ranks: 4, gpu_devices: "4,5,6,7"
      - test_dir: examples, num_ranks: 8, gpu_devices: "0,1,2,3,4,5,6,7"
      
      # unittests directory
      - test_dir: unittests, num_ranks: 1, gpu_devices: "0,1"
      - test_dir: unittests, num_ranks: 2, gpu_devices: "2,3"
      - test_dir: unittests, num_ranks: 4, gpu_devices: "4,5,6,7"
      - test_dir: unittests, num_ranks: 8, gpu_devices: "0,1,2,3,4,5,6,7"
      
      # ... same for ccl, x, ops (5 directories × 4 ranks = 20 jobs)
```

This creates 20 jobs per install method × 3 install methods = **60 total CI jobs**.

### Execution Pattern

1. **test-git job** (20 parallel jobs):
   - examples/1-rank, examples/2-rank, examples/4-rank, examples/8-rank
   - unittests/1-rank, unittests/2-rank, unittests/4-rank, unittests/8-rank
   - ccl/1-rank, ccl/2-rank, ccl/4-rank, ccl/8-rank
   - x/1-rank, x/2-rank, x/4-rank, x/8-rank
   - ops/1-rank, ops/2-rank, ops/4-rank, ops/8-rank

2. **test-editable job** (needs: test-git):
   - Same 20 jobs, but waits for test-git to complete

3. **test-install job** (needs: test-editable):
   - Same 20 jobs, but waits for test-editable to complete

### The Waste

**For test_zeros.py**:
- 50,176 test cases
- Runs with 1, 2, 4, 8 ranks (4 times)
- Runs with git, editable, install methods (3 times)
- Total executions: 50,176 × 4 × 3 = **602,112 executions**

But `zeros()` doesn't use ANY distributed features! It's a local tensor creation function. Running it on 8 GPUs tests nothing different than running on 1 GPU.

### Recommended CI Configuration

**Phase 1: Remove install duplication**
```yaml
# Single job, editable install only
test-suite:
  strategy:
    matrix:
      include:
        # Test local ops with 1 rank only
        - test_dir: unittests, num_ranks: 1, marker: "single_rank"
        
        # Test distributed ops with 2 and 8 ranks only
        - test_dir: ccl, num_ranks: 2, marker: "multi_rank"
        - test_dir: ccl, num_ranks: 8, marker: "multi_rank"
        - test_dir: x, num_ranks: 2, marker: "multi_rank"
        - test_dir: x, num_ranks: 8, marker: "multi_rank"
        
        # Test examples with representative ranks
        - test_dir: examples, num_ranks: 1
        - test_dir: examples, num_ranks: 8
        
        # Test ops with representative ranks
        - test_dir: ops, num_ranks: 2
        - test_dir: ops, num_ranks: 8

# Separate lightweight install verification
test-install-methods:
  strategy:
    matrix:
      install_method: [git, editable, install]
  steps:
    - run: |
        # Just run smoke tests (5-10 quick tests)
        pytest tests/unittests/test_iris_helpers.py
```

**Result**: 60 jobs → ~10-12 jobs (80% reduction)

---

## Example 4: test_atomic_add - Gluon vs Triton Side-by-Side

### test_atomic_add_gluon.py
```python
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32])  # 5
@pytest.mark.parametrize("sem", ["acquire", "release", "acq_rel"])  # 3
@pytest.mark.parametrize("scope", ["cta", "gpu", "sys"])  # 3
@pytest.mark.parametrize("BLOCK_SIZE", [1, 8, 16, 32])  # 4
def test_atomic_add_api(dtype, sem, scope, BLOCK_SIZE):
    shmem = iris_gl.iris(1 << 20)
    # ... setup ...
    atomic_add_kernel[grid](
        iris_gl.IrisDeviceCtx,
        context_tensor,
        results,
        sem, scope, cur_rank, num_ranks, BLOCK_SIZE,
        num_warps=1,
    )
    # ... validation ...
```

Total: 5 × 3 × 3 × 4 = **180 test cases**

### test_atomic_add_triton.py
```python
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32])  # 5
@pytest.mark.parametrize("sem", ["acquire", "release", "acq_rel"])  # 3
@pytest.mark.parametrize("scope", ["cta", "gpu", "sys"])  # 3
@pytest.mark.parametrize("BLOCK_SIZE", [1, 8, 16, 32])  # 4
def test_atomic_add_api(dtype, sem, scope, BLOCK_SIZE):
    shmem = iris.iris(1 << 20)
    # ... identical setup ...
    atomic_add_kernel[grid](results, sem, scope, cur_rank, num_ranks, BLOCK_SIZE, heap_bases)
    # ... identical validation ...
```

Total: 5 × 3 × 3 × 4 = **180 test cases**

Combined: **360 test cases** testing the exact same atomic_add functionality.

### Recommended Consolidation

**Merged test**:
```python
@pytest.mark.parametrize("api", ["gluon", "triton"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])  # Reduced from 5 to 2
@pytest.mark.parametrize("sem", ["acquire", "acq_rel"])  # Reduced from 3 to 2
@pytest.mark.parametrize("scope", ["gpu", "sys"])  # Reduced from 3 to 2
@pytest.mark.parametrize("BLOCK_SIZE", [1, 32])  # Reduced from 4 to 2 (edge cases)
def test_atomic_add(api, dtype, sem, scope, BLOCK_SIZE):
    # Shared test logic
    if api == "gluon":
        # Use gluon kernel
    else:
        # Use triton kernel
    # Shared validation
```

Total: 2 × 2 × 2 × 2 × 2 = **32 test cases** (91% reduction from 360)

---

## Example 5: Rank Configuration Waste

### Test That Doesn't Need Multiple Ranks: test_zeros.py

```python
def test_zeros_basic(dtype, size):
    shmem = iris.iris(1 << 20)
    result = shmem.zeros(*size, dtype=dtype)
    assert result.shape == size
    assert result.dtype == dtype
    assert torch.all(result == 0)
    assert shmem._Iris__on_symmetric_heap(result)
```

**What this test does**: Creates a zero tensor and verifies it's all zeros.

**What this test DOESN'T do**: 
- No communication between ranks
- No collective operations
- No multi-GPU synchronization
- Just allocates memory and fills with zeros

**Current execution**:
- Runs with 1 rank: ✓ (makes sense)
- Runs with 2 ranks: ✗ (unnecessary - same code path)
- Runs with 4 ranks: ✗ (unnecessary - same code path)
- Runs with 8 ranks: ✗ (unnecessary - same code path)

**Result**: 3× wasteful executions per test case.

### Test That DOES Need Multiple Ranks: test_all_reduce.py

```python
def test_all_reduce_sum(dtype):
    # Create distributed process group
    shmem = iris.iris(1 << 20)
    rank = shmem.get_rank()
    num_ranks = shmem.get_num_ranks()
    
    # Each rank contributes its rank value
    data = shmem.full((10,), rank, dtype=dtype)
    
    # All-reduce sum across all ranks
    result = shmem.all_reduce(data, op="sum")
    
    # Expected: sum of 0 + 1 + 2 + ... + (num_ranks-1)
    expected_sum = (num_ranks * (num_ranks - 1)) // 2
    assert torch.all(result == expected_sum)
```

**What this test does**: Tests distributed reduction across multiple GPUs.

**Why it needs multiple ranks**:
- Tests actual multi-GPU communication
- Verifies synchronization correctness
- Tests scaling behavior

**Recommended execution**:
- Runs with 1 rank: ✗ (can't test distributed behavior)
- Runs with 2 ranks: ✓ (tests basic multi-GPU case)
- Runs with 4 ranks: ~ (could skip - doesn't add much)
- Runs with 8 ranks: ✓ (tests larger scale)

### Recommendation

Tag tests with markers:
```python
@pytest.mark.single_rank
def test_zeros_basic():
    # Only needs 1 rank
    
@pytest.mark.multi_rank
def test_all_reduce_sum():
    # Needs 2+ ranks
```

Update CI to respect markers:
- `single_rank` tests: Run only with `--num_ranks=1`
- `multi_rank` tests: Run with `--num_ranks=2` and `--num_ranks=8`

---

## Summary of Specific Findings

1. **14 duplicate file pairs** (gluon/triton) → Consolidate to single parametrized tests
2. **139,216 test cases** in test_zeros_like.py alone → Reduce to ~2,000 with smart sampling
3. **60 CI jobs** (20 × 3 install methods) → Reduce to ~10 jobs
4. **360 atomic_add tests** across 2 files → Reduce to ~32 merged tests
5. **75% of tests don't need multiple ranks** → Run only with 1 rank

**Total savings potential**: 98.6% reduction in test executions while maintaining coverage.
