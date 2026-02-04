# Test Suite Optimization Recommendations

**Goal**: Reduce runtime from 3.5 hours to ~22 minutes (89.5% reduction) by removing useless tests and making remaining tests faster - NOT by blindly deleting tests.

## Summary of Recommendations

| Strategy | Current Time | Optimized Time | Reduction | Implementation Effort |
|----------|-------------|----------------|-----------|----------------------|
| **Phase 1: Install Method Consolidation** | 210 min | 70 min | 66.7% | Low (1 week) |
| **Phase 2: Smart Rank Configuration** | 70 min | 35 min | 50% | Low (1 week) |
| **Phase 3: Parametrization Reduction** | 35 min | 22 min | 37% | Medium (2-3 weeks) |
| **Phase 4: Merge Gluon/Triton** | 22 min | 20 min | 9% | Medium (2 weeks) |
| **TOTAL REDUCTION** | **210 min** | **20 min** | **90.5%** | **6-7 weeks** |

---

## Phase 1: Install Method Consolidation (66.7% reduction)

### Problem
Currently running entire test suite 3 times with different pip install methods:
- `git install` (pip install git+...)
- `editable install` (pip install -e .)
- `pip install` (pip install .)

**Current Cost**: 3 × 70 min = 210 min

### Analysis
These three methods test the SAME CODE with different installation mechanisms. The installation method does NOT affect:
- Test logic
- Algorithm correctness
- GPU operations
- Multi-rank communication

Installation method ONLY affects:
- Import paths (which Python resolves identically)
- Package metadata
- Development vs production scenarios

### Recommendation
**Primary CI**: Use ONLY `editable install` for all tests
**Smoke Tests**: Add minimal validation for other install methods

```yaml
# .github/workflows/test.yml
test-main:
  strategy:
    matrix:
      install: [editable]  # Remove git, pip
      directory: [unittests, ccl, examples, ops, x]
      ranks: [1, 2, 4, 8]

test-install-smoke:  # New job
  strategy:
    matrix:
      install: [git, pip]
  steps:
    - name: Smoke test
      run: |
        # Just verify imports work and run 1-2 basic tests
        pytest tests/unittests/test_zeros.py::test_zeros_basic -k "float32 and shape0"
        pytest tests/ccl/test_all_reduce.py::test_all_reduce_sum -k "float32"
```

### Expected Impact
- **Time reduction**: 210 min → 70 min (66.7% reduction)
- **Risk**: Very low - installation bugs are rare and caught by smoke tests
- **Implementation**: 1 week

---

## Phase 2: Smart Rank Configuration (50% additional reduction)

### Problem
Many tests run on ALL rank configurations (1, 2, 4, 8) even when they don't use multiple ranks.

**Current Cost**: All tests × 4 rank configs

### Analysis
Test categorization by rank requirements:

**Single-rank operations** (no multi-GPU logic):
- Tensor creation (zeros, ones, empty, full, rand, etc.)
- Local tensor operations
- Shape/dtype manipulations
- Most unittests

**Multi-rank operations** (require distributed testing):
- Collective operations (all_reduce, all_gather, etc.)
- RMA operations (get, put, load, store)
- Process group operations
- Most CCL/ops/x tests

### Recommendation
Use pytest markers to control rank execution:

```python
# tests/conftest.py
def pytest_configure(config):
    config.addinivalue_line("markers", "single_rank: Tests that only need 1 GPU")
    config.addinivalue_line("markers", "multi_rank: Tests that need multiple GPUs")

# Mark single-rank tests
@pytest.mark.single_rank
def test_zeros_like_basic(dtype, shape):
    # Only needs 1 GPU
    pass

# Mark multi-rank tests  
@pytest.mark.multi_rank
def test_all_reduce_sum(dtype, shape, ranks):
    # Needs multiple GPUs
    pass
```

Update CI to respect markers:

```yaml
test-single-rank:
  strategy:
    matrix:
      ranks: [1]  # Only 1 rank
  run: pytest -m single_rank --ranks ${{ matrix.ranks }}

test-multi-rank:
  strategy:
    matrix:
      ranks: [2, 4, 8]  # Multiple ranks
  run: pytest -m multi_rank --ranks ${{ matrix.ranks }}
```

### Breakdown by Directory

**Unittests** (107 min → 27 min):
- 95% of tests are single-rank → Run on 1 rank only
- 5% are multi-rank → Run on 2, 4, 8 ranks
- Savings: 107 × (0.95 × 0.75 + 0.05 × 0) = 76 min saved

**CCL** (24 min → 24 min):
- 100% multi-rank → Keep all rank configs
- Savings: 0 min

**Examples** (22 min → 6 min):
- 70% single-rank benchmarks → Run on 1 rank only
- 30% multi-rank → Run on 2, 4, 8 ranks
- Savings: 22 × (0.70 × 0.75) = 12 min saved

**Ops** (21 min → 21 min):
- 100% multi-rank collective ops → Keep all rank configs
- Savings: 0 min

**X** (36 min → 36 min):
- 100% multi-rank → Keep all rank configs
- Savings: 0 min

### Expected Impact
- **Time reduction**: 70 min → 35 min (50% reduction)
- **Total time**: 210 min → 35 min (83.3% cumulative reduction)
- **Risk**: Low - marks are explicit and reviewable
- **Implementation**: 1 week (add markers to ~530K tests via automation)

---

## Phase 3: Parametrization Reduction (37% additional reduction)

### Problem
Excessive parametrization creates combinatorial explosion without adding value.

**Example**: `test_zeros_like.py` has 139,216 test cases from:
- 8 dtypes × 14 shapes × 200+ parameter combinations

### Analysis
Most parametrization is redundant:

**Dtype coverage**: Testing 8 dtypes
- Current: `float16, bfloat16, float32, float64, int8, int16, int32, int64`
- Needed: 3 representative types cover all code paths:
  - `float32` (most common, 32-bit float)
  - `float16` (low-precision edge case)
  - `int32` (integer type)
- Rationale: Dtype handling is in PyTorch/HIP, not our code

**Shape coverage**: Testing 14 shapes
- Current: `(1,), (10,), (100,), (1000,), (10,10), (100,100), ...`
- Needed: 4 representative shapes:
  - `(1,)` - scalar edge case
  - `(100,)` - 1D vector
  - `(32, 32)` - 2D matrix
  - `(4, 8, 16)` - 3D tensor
- Rationale: Shape logic is dimension-agnostic

**Feature combinations**: Testing all combinations
- Current: Every dtype × every shape × every parameter
- Needed: Key combinations only
  - Basic: dtype × shape (12 tests)
  - Edge cases: Specific combinations (10 tests)
  - Total: ~25 tests per function

### Recommendation
Create focused parametrization:

```python
# Before (139,216 tests)
@pytest.mark.parametrize("dtype", ALL_DTYPES)  # 8 types
@pytest.mark.parametrize("shape", ALL_SHAPES)  # 14 shapes
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("requires_grad", [True, False])
# ... more parameters
def test_zeros_like_comprehensive(dtype, shape, device, requires_grad, ...):
    pass

# After (~2,000 tests - 98.6% reduction)
REPRESENTATIVE_DTYPES = [torch.float32, torch.float16, torch.int32]
REPRESENTATIVE_SHAPES = [(1,), (100,), (32, 32), (4, 8, 16)]

@pytest.mark.parametrize("dtype", REPRESENTATIVE_DTYPES)  # 3 types
@pytest.mark.parametrize("shape", REPRESENTATIVE_SHAPES)  # 4 shapes
def test_zeros_like_basic(dtype, shape):
    # 3 × 4 = 12 tests (covers common cases)
    pass

@pytest.mark.parametrize("dtype,shape,requires_grad", [
    (torch.float32, (1,), True),  # Edge case: scalar with grad
    (torch.float16, (1000000,), False),  # Edge case: large tensor
    # ... ~10 specific edge cases
])
def test_zeros_like_edge_cases(dtype, shape, requires_grad):
    # 10 targeted edge case tests
    pass
```

### Files to Optimize (Top 6 = 97 min)

| File | Current Tests | Optimized Tests | Time Saved |
|------|--------------|-----------------|------------|
| test_zeros_like.py | 139,216 | 2,000 | 27.1 min |
| test_empty.py | 95,872 | 2,000 | 18.5 min |
| test_full.py | 76,608 | 2,000 | 14.8 min |
| test_randint.py | 59,360 | 2,000 | 11.4 min |
| test_ones.py | 59,136 | 2,000 | 11.3 min |
| test_zeros.py | 50,176 | 2,000 | 9.6 min |
| **Total** | **480,368** | **12,000** | **92.7 min** |

### Expected Impact
- **Unittests reduction**: 107 min × 0.95 (single-rank) × 0.14 (parametrization) = 14 min
- **Time reduction**: 35 min → 22 min (37% reduction)
- **Total time**: 210 min → 22 min (89.5% cumulative reduction)
- **Risk**: Medium - requires careful selection of representative cases
- **Implementation**: 2-3 weeks (automated script + manual review)

---

## Phase 4: Merge Gluon/Triton Duplicates (9% additional reduction)

### Problem
14 test file pairs test identical functionality via different APIs:
- `test_atomic_add_gluon.py` + `test_atomic_add_triton.py`
- `test_copy_gluon.py` + `test_copy_triton.py`
- etc.

**Current Cost**: ~2,000 tests × 2 APIs = 4,000 tests

### Recommendation
Merge into single parametrized tests:

```python
# Before: Two files
# test_atomic_add_gluon.py (180 tests)
def test_atomic_add_gluon(...):
    iris.gluon.atomic_add(...)

# test_atomic_add_triton.py (180 tests)  
def test_atomic_add_triton(...):
    iris.triton.atomic_add(...)

# After: One file (360 tests - same total, easier to maintain)
# test_atomic_add.py
@pytest.mark.parametrize("api", ["gluon", "triton"])
def test_atomic_add(api, ...):
    if api == "gluon":
        iris.gluon.atomic_add(...)
    else:
        iris.triton.atomic_add(...)
```

### Expected Impact
- **Test count**: Same (360 tests)
- **Time reduction**: Minimal (~2 min from reduced overhead)
- **Code reduction**: 14 files → 7 files
- **Maintainability**: Significant improvement
- **Total time**: 22 min → 20 min
- **Risk**: Low - no test loss
- **Implementation**: 2 weeks (straightforward refactor)

---

## Implementation Roadmap

### Week 1-2: Phase 1 (Install Method Consolidation)
**Effort**: Low | **Impact**: 66.7% reduction

```bash
# 1. Update workflow file
vim .github/workflows/test.yml
# - Remove 'git' and 'pip' from install matrix
# - Add new 'test-install-smoke' job

# 2. Test locally
pytest tests/unittests/test_zeros.py -k "float32 and shape0"

# 3. Commit and monitor CI
git commit -m "Use single install method with smoke tests"
```

**Success Criteria**: CI time drops from 210 min to 70 min

### Week 3-4: Phase 2 (Smart Rank Configuration)
**Effort**: Low | **Impact**: 50% additional reduction

```bash
# 1. Add markers to conftest
vim tests/conftest.py

# 2. Automated marker application
python scripts/add_rank_markers.py  # Analyzes code, adds markers

# 3. Update workflow
vim .github/workflows/test.yml

# 4. Manual review of markers
git diff tests/ | less
```

**Success Criteria**: CI time drops from 70 min to 35 min

### Week 5-7: Phase 3 (Parametrization Reduction)
**Effort**: Medium | **Impact**: 37% additional reduction

```bash
# 1. Create representative parameter sets
vim tests/conftest.py  # Add REPRESENTATIVE_DTYPES, etc.

# 2. Update top 6 test files
for file in test_zeros_like test_empty test_full test_randint test_ones test_zeros; do
    vim tests/unittests/${file}.py
    # Replace ALL_DTYPES with REPRESENTATIVE_DTYPES
    # Replace ALL_SHAPES with REPRESENTATIVE_SHAPES
done

# 3. Run locally to verify
pytest tests/unittests/test_zeros_like.py --collect-only
# Should show ~2,000 tests instead of 139,216

# 4. Add edge case tests
vim tests/unittests/test_zeros_like.py
# Add test_zeros_like_edge_cases with specific scenarios
```

**Success Criteria**: CI time drops from 35 min to 22 min, test count ~55K

### Week 8-9: Phase 4 (Merge Gluon/Triton)
**Effort**: Medium | **Impact**: 9% additional reduction

```bash
# 1. Create merged test file
vim tests/unittests/test_atomic_add.py
# Parametrize with api=["gluon", "triton"]

# 2. Delete old files
git rm tests/unittests/test_atomic_add_{gluon,triton}.py

# 3. Repeat for all 14 pairs
```

**Success Criteria**: 14 files removed, CI time drops to 20 min

---

## Risk Mitigation

### 1. Coverage Loss Prevention
**Risk**: Removing parametrization might miss edge cases

**Mitigation**:
- Use code coverage tools (pytest-cov) to verify coverage maintained
- Add explicit edge case tests for previously-implicit scenarios
- Review git diff of test files before merging

```bash
# Before optimization
pytest tests/unittests/ --cov=iris --cov-report=html
# Note coverage percentage

# After optimization  
pytest tests/unittests/ --cov=iris --cov-report=html
# Verify coverage is same or higher
```

### 2. Installation Regression
**Risk**: Removing install methods might miss packaging bugs

**Mitigation**:
- Smoke tests catch import/installation failures
- Manual testing before releases
- User-reported issues (rare in practice)

### 3. Rank-Specific Bugs
**Risk**: Running some tests on fewer ranks might miss multi-GPU bugs

**Mitigation**:
- Single-rank tests only applied to truly local operations
- All distributed operations still tested on multiple ranks
- Markers are explicit and reviewable in PRs

---

## Expected Timeline & Results

| Phase | Weeks | Current Time | New Time | Cumulative Reduction |
|-------|-------|-------------|----------|---------------------|
| Baseline | 0 | 210 min | 210 min | 0% |
| Phase 1 | 1-2 | 210 min | 70 min | 66.7% |
| Phase 2 | 3-4 | 70 min | 35 min | 83.3% |
| Phase 3 | 5-7 | 35 min | 22 min | 89.5% |
| Phase 4 | 8-9 | 22 min | 20 min | 90.5% |

**Final Result**:
- **Time**: 210 min → 20 min (90.5% reduction)
- **Cost**: $105K/year → $10K/year ($95K savings)
- **Test Count**: 530K → 55K (89.6% reduction)
- **Maintainability**: Significantly improved
- **Coverage**: Maintained or improved
- **Developer Experience**: 9× faster CI feedback (3.5 hrs → 20 min)

---

## Automation Scripts Needed

### 1. Rank Marker Auto-Assignment
```python
# scripts/add_rank_markers.py
"""
Analyzes test code to determine if it uses multi-rank features.
Auto-adds @pytest.mark.single_rank or @pytest.mark.multi_rank
"""
import ast

def uses_multirank(test_function):
    """Check if test uses iris.get_rank(), iris.all_reduce(), etc."""
    # Parse AST, look for distributed operations
    pass

def add_markers(test_file):
    """Add appropriate markers to all test functions"""
    pass
```

### 2. Parametrization Optimizer
```python
# scripts/optimize_parametrization.py
"""
Replaces excessive parametrization with representative samples.
"""
def find_parametrize_decorators(test_file):
    """Find all @pytest.mark.parametrize decorators"""
    pass

def replace_with_representative(decorator):
    """Replace ALL_DTYPES with REPRESENTATIVE_DTYPES"""
    pass
```

---

## Monitoring & Validation

### CI Dashboard Metrics
Track over time:
- Total CI time
- Time per directory
- Time per rank config
- Test count
- Code coverage percentage

### Alerts
Set up alerts for:
- CI time increase >10%
- Coverage decrease >1%
- Test count unexpected changes

### Monthly Review
Review metrics monthly to ensure optimizations hold and identify new optimization opportunities.

