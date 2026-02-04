# Test Suite Optimization - Phase 1

## Overview

This document describes the Phase 1 test suite optimization implemented to reduce CI time by ~30% (from 210 minutes to 147 minutes).

## Background

Analysis revealed that the original test suite was running **every test** on **all 4 rank configurations** (1, 2, 4, 8 ranks), which was wasteful. While multi-rank validation is essential for distributed features (symmetric heap allocation, cross-rank operations), many tests only validate tensor properties (shape, dtype, values) and don't require multi-rank execution.

### Original Test Execution
- **3 install methods** × **5 test directories** × **4 rank configs** = **60 CI jobs**
- Each job runs all tests in a directory
- Total multi-rank test runs: **6.37M**

### Optimized Test Execution
- **Same CI matrix structure** (no workflow changes)
- Tests are filtered automatically by pytest markers
- Single-rank tests skip execution when NUM_RANKS > 1
- Total multi-rank test runs: **3.98M** (37.5% reduction)

## Implementation

### 1. Pytest Markers

Two new markers were added in `pytest.ini`:

- **`@pytest.mark.single_rank`**: Tests that validate tensor properties (shape, dtype, values)
  - These tests only need to run on **1 rank**
  - Examples: `test_zeros`, `test_ones`, `test_rand`, `test_full`, `test_empty`
  
- **`@pytest.mark.multi_rank_required`**: Tests that validate distributed behavior
  - These tests must run on **all rank configurations** (1, 2, 4, 8)
  - Examples: `test_get_*`, `test_put_*`, `test_load_*`, `test_store_*`, `test_all_reduce`, `test_all_gather`

### 2. Test Classification

Tests were classified into three categories:

| Category | Count | Runs on Ranks | Examples |
|----------|-------|---------------|----------|
| `single_rank` | 10 files | 1 only | zeros, ones, rand, empty, full, arange, linspace, randint, randn, zeros_like |
| `multi_rank_required` | 47 files | 1, 2, 4, 8 | get, put, load, store, atomic_*, broadcast, copy, all_reduce, all_gather, all_to_all |
| Unmarked | 4 files | 1, 2, 4, 8 | logging, dmabuf_apis, get_num_xcc, iris_helpers |

### 3. Automated Marker Assignment

A Python script `scripts/assign_test_markers.py` was created to automate the marker assignment process:

```bash
# Preview changes (dry run)
python scripts/assign_test_markers.py --dry-run --test-dir tests

# Apply markers
python scripts/assign_test_markers.py --test-dir tests
```

The script:
- Classifies tests based on their functionality
- Adds `pytestmark = pytest.mark.<marker>` to test files
- Preserves backward compatibility for unmarked tests

### 4. Test Filtering

The `.github/scripts/run_tests.sh` script was minimally modified to skip `single_rank` tests when running with multiple ranks:

```bash
# Skip single_rank tests when running with multiple ranks
MARKER_ARG=""
if [ "$NUM_RANKS" -gt 1 ]; then
    MARKER_ARG="-m 'not single_rank'"
fi
```

This approach:
- Requires minimal changes to CI infrastructure
- Uses pytest's built-in marker filtering
- Automatically skips single_rank tests on multi-rank configurations
- Preserves the existing CI workflow structure

## Adding New Tests

When adding new tests, follow these guidelines:

### Single-rank Tests
Use `@pytest.mark.single_rank` for tests that:
- Validate tensor properties (shape, dtype, values)
- Test tensor creation functions (zeros, ones, rand, etc.)
- Don't involve cross-rank communication
- Can verify correctness on a single rank

Example:
```python
import pytest
import iris

pytestmark = pytest.mark.single_rank

def test_zeros():
    shmem = iris.iris(1 << 20)
    result = shmem.zeros(2, 3, dtype=torch.float32)
    assert result.shape == (2, 3)
    assert result.dtype == torch.float32
```

### Multi-rank Tests
Use `@pytest.mark.multi_rank_required` for tests that:
- Validate distributed behavior
- Test cross-rank operations (get, put, load, store)
- Test collective operations (all_reduce, all_gather, all_to_all)
- Test atomic operations across ranks
- Require symmetric heap visibility validation

Example:
```python
import pytest
import iris

pytestmark = pytest.mark.multi_rank_required

def test_all_reduce():
    shmem = iris.iris(1 << 20)
    # Test requires multiple ranks to validate reduction
    input_tensor = shmem.ones(10, dtype=torch.float32) * shmem.get_rank()
    output = shmem.ccl.all_reduce(input_tensor)
    # Validation logic...
```

### Unmarked Tests
Leave tests unmarked if:
- They test infrastructure/utilities (logging, helpers)
- Classification is unclear
- Backward compatibility is preferred

## Running Tests Locally

### Run all tests
```bash
pytest tests/
```

### Run only single-rank tests
```bash
pytest tests/ -m single_rank
```

### Run only multi-rank tests
```bash
pytest tests/ -m multi_rank_required
```

### Run unmarked tests
```bash
pytest tests/ -m "not single_rank and not multi_rank_required"
```

### Run with specific rank count
```bash
python tests/run_tests_distributed.py --num_ranks 4 tests/ccl/test_all_reduce.py -m multi_rank_required
```

## Expected Impact

### Time Savings
- **Previous CI time**: ~210 minutes
- **New CI time**: ~147 minutes
- **Reduction**: 63 minutes (30%)

### Test Execution Reduction
- **Previous multi-rank test runs**: 6.37M
- **New multi-rank test runs**: 3.98M
- **Reduction**: 2.39M test runs (37.5%)

### Key Metrics
- **Test count**: Unchanged (530,877 tests)
- **Coverage**: No reduction - all tests still run at least once
- **Quality**: No degradation - multi-rank tests still validated on all configs

## Future Optimizations (Phase 2+)

Potential future optimizations include:
1. **Parameterization reduction**: Reduce parameter combinations for single-rank tests
2. **Test parallelization**: Run independent tests in parallel
3. **Caching**: Cache build artifacts between jobs
4. **Smart test selection**: Skip tests unaffected by code changes

## References

- Issue: [Implement test suite optimization](https://github.com/ROCm/iris/issues/XXX)
- PR: [Test Suite Optimization - Phase 1](https://github.com/ROCm/iris/pull/XXX)
- Analysis: See PRs #353 and #354 for detailed analysis

## Troubleshooting

### Marker not recognized
Ensure `pytest.ini` is present in the repository root with the marker definitions.

### Tests not filtered correctly
1. Check that the marker is added to the test file
2. Verify the marker syntax: `pytestmark = pytest.mark.<marker>`
3. Check that the CI workflow passes the marker parameter correctly

### CI failures after optimization
1. Check that multi-rank tests have `multi_rank_required` marker
2. Verify that single-rank tests don't depend on multi-rank execution
3. Review test logs to identify which rank configuration failed
