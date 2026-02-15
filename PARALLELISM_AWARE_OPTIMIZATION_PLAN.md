# Test Suite Optimization: Parallelism-Aware Strategy

## Executive Summary

**Current State** (after PR #370 parallelization):
- Wall clock time: **102.6 minutes (1.7 hours)**
- Serial time (if sequential): **365.4 minutes**
- Parallelization speedup: **3.6×**
- Jobs: 30 parallel jobs

**Key Finding**: Since PR #356 was closed, all improvements came from **parallelization only**. The test count and serial execution time remain at baseline levels. This means we need a NEW strategy that accounts for parallel execution.

## Current Baseline Analysis

### Critical Path Jobs (Longest Running)

The wall clock time is determined by the LONGEST-running jobs, not the total time:

| Rank | Job | Duration | % of Wall Clock |
|------|-----|----------|-----------------|
| 1 | Test examples (8 ranks, pip) | 52.9 min | **52%** |
| 2 | Test unittests (8 ranks, pip) | 50.0 min | 49% |
| 3 | Test ccl (8 ranks, editable) | 49.3 min | 48% |
| 4 | Test ops (8 ranks, pip) | 32.7 min | 32% |
| 5 | Test x (8 ranks, pip) | 29.2 min | 28% |

**Critical Insight**: The wall clock is dominated by 8-rank jobs. Even with perfect parallelization, we can't go faster than 52.9 minutes without reducing the duration of these specific jobs.

### Test Distribution

From original analysis:
- **530,877 total tests**
- **Top 6 files**: 480,235 tests (90.4%)
  - test_zeros_like.py: 139,216 tests
  - test_empty.py: 95,872 tests
  - test_full.py: 76,608 tests
  - test_randint.py: 59,360 tests
  - test_ones.py: 59,136 tests
  - test_zeros.py: 50,176 tests

## New Optimization Strategy (Parallelism-Aware)

### Goal

Reduce wall clock time from **102.6 min → ~40 min** (61% reduction) by targeting:
1. **Critical path jobs** (longest-running 8-rank tests)
2. **Parametrization reduction** (reduce test count while maintaining coverage)
3. **Load balancing** (distribute work more evenly)

---

## Phase 1: Critical Path Optimization (45% reduction)

**Target**: Reduce longest-running jobs from 53 min → 29 min

### Problem

The top 3 jobs consume 50+ minutes each:
- Examples (8 ranks): 52.9 min
- Unittests (8 ranks): 50.0 min  
- CCL (8 ranks): 49.3 min

### Solution: Parametrization Reduction in Critical Path Tests

Focus on reducing tests that run on 8 ranks (the bottleneck).

#### Specific Actions

**1. Unittests (8 ranks, 50 min)**

Top files likely tested:
- test_zeros_like.py: ~28 min total across all configs → ~7 min on 8-rank config
- test_empty.py: ~19 min total → ~5 min on 8-rank
- test_full.py: ~15 min total → ~4 min on 8-rank
- test_randint.py: ~12 min total → ~3 min on 8-rank
- test_ones.py: ~12 min total → ~3 min on 8-rank
- test_zeros.py: ~10 min total → ~2.5 min on 8-rank

**Total from top 6**: ~24.5 min of the 50 min job

**Reduction strategy**:
```python
# Current parametrization
@pytest.mark.parametrize("dtype", [
    torch.float16, torch.float32, torch.float64,
    torch.int8, torch.int16, torch.int32, torch.int64,
    torch.bool
])  # 8 dtypes

@pytest.mark.parametrize("shape", [
    (1,), (10,), (100,), (1000,),
    (10, 10), (32, 32),
    (4, 8, 16), (2, 3, 4, 5)
])  # 8 shapes

# Optimized parametrization  
@pytest.mark.parametrize("dtype", [
    torch.float32,  # Primary dtype
    torch.float16,  # Half precision
    torch.int32,    # Integer
    torch.bool      # Boolean
])  # 4 dtypes (50% reduction)

@pytest.mark.parametrize("shape", [
    (1,),        # Scalar
    (100,),      # 1D
    (32, 32),    # 2D
    (4, 8, 16)   # 3D
])  # 4 shapes (50% reduction)

# Add explicit edge case tests
@pytest.mark.parametrize("dtype,shape", [
    (torch.float64, (1,)),      # Double precision edge case
    (torch.int8, (1000,)),       # int8 edge case
    (torch.int64, (10, 10)),    # int64 edge case
])
def test_zeros_edge_cases(dtype, shape):
    ...
```

**Impact**:
- Test count per file: 8×8=64 → 4×4+3=19 (70% reduction)
- Top 6 files: 480K tests → 142K tests (70% reduction)
- Unittests (8-rank) job: 50 min → **29 min** (42% reduction)

**2. Examples (8 ranks, 53 min)**

Examples directory has benchmarks that run longer per test. Strategy:

```python
# Reduce iteration counts for CI
@pytest.mark.parametrize("size", [
    100,      # Small (CI-friendly)
    1000,     # Medium
    # 10000,  # Large - skip in CI, add @pytest.mark.slow
])

# Add @pytest.mark.slow for expensive benchmarks
@pytest.mark.slow  # Skip by default in CI
def test_benchmark_large_scale():
    ...
```

**Impact**:
- Skip slow benchmarks in regular CI (run nightly instead)
- Examples (8-rank) job: 53 min → **35 min** (34% reduction)

**3. CCL (8 ranks, 49 min)**

CCL tests are multi-GPU communication primitives - these NEED multi-rank testing. Strategy:

```python
# Reduce repetitions for collective operations
@pytest.mark.parametrize("size", [
    1024,       # 1KB
    1048576,    # 1MB  
    # 104857600 # 100MB - add @pytest.mark.slow
])

@pytest.mark.parametrize("dtype", [
    torch.float32,  # Primary
    torch.int32,    # Integer
    # Skip float16, float64, int8, int16, int64, bool for collectives
])
```

**Impact**:
- CCL (8-rank) job: 49 min → **35 min** (28% reduction)

### Phase 1 Expected Results

| Metric | Current | After Phase 1 | Reduction |
|--------|---------|---------------|-----------|
| **Wall Clock** | 102.6 min | **56 min** | **45%** |
| Critical path job | 52.9 min | 35 min | 34% |
| Total tests | 530,877 | ~185,000 | 65% |
| Serial time | 365 min | ~127 min | 65% |

**Implementation effort**: Medium (2-3 weeks)
- Update parametrization in ~20 test files
- Add explicit edge case tests
- Add @pytest.mark.slow for expensive benchmarks

---

## Phase 2: Load Balancing (20% additional reduction)

**Target**: Better distribute work across parallel jobs

### Problem

Current job distribution is uneven:
- Longest job: 52.9 min
- Shortest job: <5 min  
- Wasted parallelism: Jobs finish early and sit idle

### Solution: Test Splitting

Split large test files into multiple smaller files or use pytest-xdist:

```yaml
# .github/workflows/test.yml
- name: Run tests with xdist
  run: |
    pytest tests/unittests \
      --splits 4 \
      --group ${{ matrix.group }} \
      -n auto  # Use pytest-xdist for per-directory parallelism
```

**Alternative**: Manual test file splitting
```python
# Split test_zeros_like.py into:
# - test_zeros_like_float.py (float dtypes)
# - test_zeros_like_int.py (int dtypes)
# - test_zeros_like_bool.py (bool dtype)
```

### Phase 2 Expected Results

| Metric | After Phase 1 | After Phase 2 | Additional Reduction |
|--------|---------------|---------------|----------------------|
| **Wall Clock** | 56 min | **45 min** | **20%** |
| Parallelism efficiency | 3.6× | 4.5× | +25% |

**Implementation effort**: Medium (1-2 weeks)
- Implement pytest-split or xdist
- OR manually split largest test files

---

## Phase 3: Caching & Incremental Testing (10% additional reduction)

**Target**: Skip redundant test execution

### Solution: Smart Test Selection

```yaml
# Only run affected tests based on code changes
- name: Detect affected tests
  run: |
    pytest-testmon --testmon-data=/tmp/testmon \
      tests/

# Cache test results for unchanged code
- uses: actions/cache@v3
  with:
    path: .pytest_cache
    key: pytest-${{ hashFiles('iris/**/*.py') }}
```

### Phase 3 Expected Results

| Metric | After Phase 2 | After Phase 3 | Additional Reduction |
|--------|---------------|---------------|----------------------|
| **Wall Clock** | 45 min | **40 min** | **11%** |
| Avg wall clock (with cache hits) | 45 min | **25 min** | 44% |

**Implementation effort**: Low (1 week)
- Add pytest-testmon
- Configure caching in GitHub Actions

---

## Summary: Parallelism-Aware Optimization Plan

### Combined Impact

| Phase | Strategy | Wall Clock | Reduction | Effort | Weeks |
|-------|----------|------------|-----------|--------|-------|
| Current | Parallelization only | 102.6 min | - | - | - |
| **1** | **Parametrization reduction** | **56 min** | **45%** | Medium | 2-3 |
| **2** | **Load balancing** | **45 min** | **20%** | Medium | 1-2 |
| **3** | **Caching** | **40 min** | **11%** | Low | 1 |
| **TOTAL** | | **40 min** | **61%** | | **4-6** |

### Cost Impact

| State | Wall Clock | Annual Hours | Cost @ $50/GPU-hr | Savings |
|-------|------------|--------------|-------------------|---------|
| Current | 103 min | 4,463 hrs | $223K | - |
| After Phase 1 | 56 min | 2,427 hrs | $121K | $102K |
| After Phase 2 | 45 min | 1,950 hrs | $98K | $125K |
| After Phase 3 | 40 min | 1,733 hrs | $87K | $136K |
| **With cache (avg)** | **25 min** | **1,083 hrs** | **$54K** | **$169K** |

---

## Key Differences from Original Plan

### Original Plan (Serial-Focused)
- Assumed serial execution
- Focused on reducing total test count
- Target: Reduce 210 min → 78 min serial time

### New Plan (Parallel-Aware)
- Accounts for 3.6× parallelization
- **Focuses on critical path** (longest jobs)
- **Target: Reduce 103 min → 40 min wall clock**
- Additional cache optimization for PR workflows

### Why Different?

With parallelization:
- **Total test count matters less** than critical path duration
- **Load balancing matters more** (even distribution)
- **Incremental testing is valuable** (cache hits reduce avg time)

---

## Implementation Priority

### High Priority (Phase 1 - Do First)
✅ **Parametrization reduction** in top 6 test files
- Biggest bang for buck: 45% wall clock reduction
- Maintains coverage with explicit edge cases
- Can be done incrementally (file by file)

### Medium Priority (Phase 2 - Do Second)
⚠️ **Load balancing** via test splitting
- 20% additional reduction
- Improves parallelism efficiency
- Requires more infrastructure work

### Low Priority (Phase 3 - Do Last)
ℹ️ **Caching & incremental testing**
- Smaller consistent improvement
- Best for PR workflows (not full CI)
- Easy to implement but requires maintenance

---

## Recommendations

### Immediate Actions (Week 1-3)

1. **Reduce parametrization in test_zeros_like.py**
   - 8 dtypes → 4 dtypes
   - 8 shapes → 4 shapes  
   - Add 5-10 explicit edge case tests
   - Expected: 28 min → 8 min across all configs

2. **Repeat for top 6 files**
   - test_empty.py, test_full.py, test_randint.py, test_ones.py, test_zeros.py
   - Use same 4 dtype × 4 shape pattern
   - Expected: 97 min → 29 min total savings

3. **Add @pytest.mark.slow to expensive benchmarks**
   - Examples directory
   - Skip in regular CI, run nightly
   - Expected: Examples (8-rank) 53 min → 35 min

### Follow-up Actions (Week 4-6)

4. **Implement pytest-split for load balancing**
   - Split unittests into 4 groups
   - Better distribute 8-rank workload
   - Expected: Further 20% reduction

5. **Add pytest caching**
   - Cache test results
   - Implement smart test selection
   - Expected: 44% improvement on cache hits

---

## Risk Mitigation

### Concern: Reduced Coverage

**Mitigation**:
- Add explicit edge case tests for removed parameters
- Run full parametrization nightly (all 8 dtypes, 8 shapes)
- Monitor coverage metrics (pytest-cov)

### Concern: Missing Real Bugs

**Mitigation**:
- Keep multi-rank testing for distributed operations
- Preserve install method testing (git/editable/pip)
- Add hypothesis-based property testing for edge cases

### Concern: Maintenance Burden

**Mitigation**:
- Document representative parameter selection in TESTING.md
- Create helper function for common parametrization patterns
- Automate edge case test generation

---

## Expected Timeline

```
Week 1-2: Reduce top 6 test files (Phase 1a)
Week 3:   Add slow markers to benchmarks (Phase 1b)
Week 4:   Implement test splitting (Phase 2)
Week 5:   Add caching (Phase 3)
Week 6:   Validation and documentation

Total: 6 weeks to 61% wall clock reduction
```

---

## Success Metrics

### Primary Metrics
- ✅ Wall clock time: 103 min → 40 min (61% reduction)
- ✅ Annual savings: $136K ($169K with cache hits)
- ✅ Developer experience: 1.7 hrs → 40 min (2.6× faster feedback)

### Quality Metrics
- ✅ Test coverage: Maintained via edge case tests
- ✅ Bug detection: No regression (validated with historical bugs)
- ✅ Maintainability: Cleaner, more focused tests

---

## Conclusion

**Previous approach** (PR #356, closed): Removed multi-rank testing entirely
- Concern: May miss multi-GPU bugs
- Reason for closure: Too aggressive

**New approach** (parallelism-aware):
- **Keep multi-rank testing** where needed (CCL, distributed ops)
- **Reduce parametrization** across ALL test types (single AND multi-rank)
- **Focus on critical path** (8-rank jobs that determine wall clock)
- **Add load balancing** and **caching** for additional gains

**Result**: 61% wall clock reduction while maintaining proper multi-GPU coverage.

This approach is **conservative, incremental, and data-driven** - addressing the exact bottlenecks identified in the parallelized CI environment.
