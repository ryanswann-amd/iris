# Test Suite Review - Data Tables

This document contains all the raw data and detailed breakdowns from the analysis.

## Table 1: Test Distribution by Directory

| Directory | Test Files | Test Functions | Base Test Cases | CI Executions (×12) | % of Total |
|-----------|-----------|---------------|-----------------|---------------------|------------|
| unittests | 42 | 227 | 530,399 | 6,364,788 | 99.91% |
| ccl | 5 | 13 | 309 | 3,708 | 0.06% |
| examples | 5 | 8 | 146 | 1,752 | 0.03% |
| x | 5 | 7 | 13 | 156 | <0.01% |
| ops | 4 | 6 | 10 | 120 | <0.01% |
| **TOTAL** | **61** | **261** | **530,877** | **6,370,524** | **100%** |

*CI Executions = Base Test Cases × 4 ranks × 3 install methods*

---

## Table 2: Top 30 Test Files by Test Count

| Rank | Test File | Test Cases | CI Executions | % of Total |
|------|-----------|-----------|---------------|------------|
| 1 | test_zeros_like.py | 139,216 | 1,670,592 | 26.22% |
| 2 | test_empty.py | 95,872 | 1,150,464 | 18.06% |
| 3 | test_full.py | 76,608 | 919,296 | 14.43% |
| 4 | test_randint.py | 59,360 | 712,320 | 11.18% |
| 5 | test_ones.py | 59,136 | 709,632 | 11.14% |
| 6 | test_zeros.py | 50,176 | 602,112 | 9.45% |
| 7 | test_randn.py | 17,724 | 212,688 | 3.34% |
| 8 | test_rand.py | 17,724 | 212,688 | 3.34% |
| 9 | test_copy_gluon.py | 4,368 | 52,416 | 0.82% |
| 10 | test_copy_triton.py | 4,368 | 52,416 | 0.82% |
| 11 | test_linspace.py | 3,840 | 46,080 | 0.72% |
| 12 | test_arange.py | 609 | 7,308 | 0.11% |
| 13 | test_atomic_add_gluon.py | 180 | 2,160 | 0.03% |
| 14 | test_atomic_add_triton.py | 180 | 2,160 | 0.03% |
| 15 | test_process_groups.py | 282 | 3,384 | 0.05% |
| 16 | test_atomic_and_gluon.py | 72 | 864 | 0.01% |
| 17 | test_atomic_and_triton.py | 72 | 864 | 0.01% |
| 18 | test_atomic_max_gluon.py | 72 | 864 | 0.01% |
| 19 | test_atomic_max_triton.py | 72 | 864 | 0.01% |
| 20 | test_atomic_min_gluon.py | 72 | 864 | 0.01% |
| 21 | test_atomic_min_triton.py | 72 | 864 | 0.01% |
| 22 | test_atomic_or_gluon.py | 72 | 864 | 0.01% |
| 23 | test_atomic_or_triton.py | 72 | 864 | 0.01% |
| 24 | test_atomic_xor_gluon.py | 72 | 864 | 0.01% |
| 25 | test_atomic_xor_triton.py | 72 | 864 | 0.01% |
| 26 | test_message_passing.py | 72 | 864 | 0.01% |
| 27 | test_atomic_add_bench.py | 42 | 504 | 0.01% |
| 28 | test_broadcast_gluon.py | 33 | 396 | 0.01% |
| 29 | test_broadcast_triton.py | 33 | 396 | 0.01% |
| 30 | test_atomic_cas_gluon.py | 27 | 324 | 0.01% |

**Top 10 files**: 524,552 tests = **98.8% of all tests**

---

## Table 3: Duplicate Gluon/Triton Test Pairs

| Base Name | Gluon File | Triton File | Tests Each | Combined Tests | Waste Factor |
|-----------|-----------|-------------|-----------|---------------|--------------|
| test_copy | test_copy_gluon.py | test_copy_triton.py | 4,368 | 8,736 | 2× |
| test_atomic_add | test_atomic_add_gluon.py | test_atomic_add_triton.py | 180 | 360 | 2× |
| test_atomic_and | test_atomic_and_gluon.py | test_atomic_and_triton.py | 72 | 144 | 2× |
| test_atomic_max | test_atomic_max_gluon.py | test_atomic_max_triton.py | 72 | 144 | 2× |
| test_atomic_min | test_atomic_min_gluon.py | test_atomic_min_triton.py | 72 | 144 | 2× |
| test_atomic_or | test_atomic_or_gluon.py | test_atomic_or_triton.py | 72 | 144 | 2× |
| test_atomic_xor | test_atomic_xor_gluon.py | test_atomic_xor_triton.py | 72 | 144 | 2× |
| test_broadcast | test_broadcast_gluon.py | test_broadcast_triton.py | 33 | 66 | 2× |
| test_atomic_cas | test_atomic_cas_gluon.py | test_atomic_cas_triton.py | 27 | 54 | 2× |
| test_atomic_xchg | test_atomic_xchg_gluon.py | test_atomic_xchg_triton.py | 27 | 54 | 2× |
| test_get | test_get_gluon.py | test_get_triton.py | 16 | 32 | 2× |
| test_load | test_load_gluon.py | test_load_triton.py | 16 | 32 | 2× |
| test_put | test_put_gluon.py | test_put_triton.py | 16 | 32 | 2× |
| test_store | test_store_gluon.py | test_store_triton.py | 16 | 32 | 2× |
| **TOTAL** | **14 files** | **14 files** | **~5,000** | **~10,000** | **2×** |

---

## Table 4: CI Matrix Configuration

### Current Configuration (60 jobs)

| Install Method | Test Directory | Rank Config | Total Jobs |
|---------------|----------------|-------------|-----------|
| git | examples, unittests, ccl, x, ops | 1, 2, 4, 8 | 20 |
| editable | examples, unittests, ccl, x, ops | 1, 2, 4, 8 | 20 |
| install | examples, unittests, ccl, x, ops | 1, 2, 4, 8 | 20 |
| **TOTAL** | **5 directories** | **4 configs** | **60** |

### Execution Pattern (Sequential Waterfall)

```
test-git (20 jobs)
    ↓ (waits for completion)
test-editable (20 jobs)
    ↓ (waits for completion)
test-install (20 jobs)
```

**Estimated wall-clock time**: 60 min × 3 = 180 minutes

---

## Table 5: Parametrization Analysis - test_zeros_like.py

| Test Function | Combinations | Description |
|--------------|--------------|-------------|
| test_zeros_like_basic | 48 | 8 dtypes × 6 shapes |
| test_zeros_like_dtype_override | 48 | Override dtype parameter |
| test_zeros_like_requires_grad | 2 | requires_grad True/False |
| test_zeros_like_device_override | 1 | Device override test |
| test_zeros_like_layout_override | 1 | Layout override test |
| test_zeros_like_memory_format | 1 | Memory format test |
| test_channels_last_format_shape_preservation | 1 | Channels last format |
| test_zeros_like_pytorch_equivalence | 1 | PyTorch equivalence |
| test_zeros_like_edge_cases | 1 | Edge cases |
| test_zeros_like_parameter_combinations | ~10,000+ | Multiple parameter combinations |
| test_zeros_like_symmetric_heap_shapes_dtypes | ~10,000+ | Symmetric heap tests |
| test_zeros_like_symmetric_heap_dtype_override | ~50,000+ | Heap dtype override |
| test_zeros_like_symmetric_heap_other_params | ~50,000+ | Other heap params |
| **TOTAL** | **139,216** | All combinations |

*Note: Actual parametrization appears to have module-level or fixture-based multiplication that creates the massive numbers*

---

## Table 6: Savings Breakdown by Optimization

| Optimization | Current | After | Reduction | % Savings |
|-------------|---------|-------|-----------|-----------|
| **Base Test Count** |
| Reduce parametrization | 530,877 | 65,000 | 465,877 | 87.8% |
| Merge gluon/triton | 65,000 | 55,000 | 10,000 | 15.4% |
| **Final base count** | **530,877** | **55,000** | **475,877** | **89.6%** |
| | | | | |
| **CI Executions** |
| Remove install duplication | 6,370,524 | 2,123,508 | 4,247,016 | 66.7% |
| Smart rank configs | 2,123,508 | ~350,000 | ~1,773,508 | 83.5% |
| Apply test count reduction | ~350,000 | ~88,000 | ~262,000 | 74.9% |
| **Final execution count** | **6,370,524** | **~88,000** | **~6,282,524** | **98.6%** |
| | | | | |
| **CI Time (minutes)** |
| Remove sequential dependency | 180 | 60 | 120 | 66.7% |
| Reduce test count | 60 | 8-12 | 48-52 | 80-87% |
| **Final CI time** | **180** | **8-12** | **168-172** | **93-95%** |

---

## Table 7: Estimated Cost Savings

### Assumptions
- Self-hosted AMD GPUs (8× MI300X per CI run)
- Average PR triggers 2-3 CI runs
- ~500 PRs per year
- GPU time cost: $3/hour per GPU
- Current CI time: 4 hours per run
- Optimized CI time: 0.2 hours per run

### Current Annual Cost
```
500 PRs × 2.5 runs × 4 hours × 8 GPUs × $3/hour = $120,000/year
```

### After Optimization
```
500 PRs × 2.5 runs × 0.2 hours × 8 GPUs × $3/hour = $6,000/year
```

### Savings
```
$120,000 - $6,000 = $114,000/year (95% reduction)
```

---

## Table 8: Implementation Effort Estimates

| Phase | Optimization | Effort | Impact | Priority |
|-------|-------------|--------|--------|----------|
| 1 | Remove install duplication | 1-2 weeks | 67% execution reduction | HIGH |
| 1 | Smart rank configs | 1-2 weeks | 60-70% reduction | HIGH |
| 2 | Merge gluon/triton tests | 2-3 weeks | 10K fewer tests | MEDIUM |
| 3 | Reduce parametrization | 3-4 weeks | 87% test reduction | HIGH |
| 4 | Structural improvements | Ongoing | Maintainability | LOW |

**Total implementation time**: 8-12 weeks for Phases 1-3

---

## Table 9: Risk Assessment

| Risk | Likelihood | Impact | Mitigation | Risk Level |
|------|-----------|--------|------------|-----------|
| Reduced coverage | Medium | High | Use representative samples, monitor coverage | Medium |
| Breaking changes during refactor | Low | Medium | Incremental changes, thorough testing | Low |
| Team resistance | Medium | Low | Data-driven analysis, gradual rollout | Low |
| Performance regression | Low | Low | Keep performance benchmarks separate | Low |
| False sense of security | Low | Medium | Add property-based testing | Low |

**Overall risk level**: LOW-MEDIUM (with proper mitigation)

---

## Table 10: Recommended Parametrization Reductions

### Tensor Creation Tests (zeros, ones, empty, full, etc.)

| Parameter | Current Values | Recommended Values | Reduction |
|-----------|---------------|-------------------|-----------|
| dtype | 8 (int8, int16, int32, int64, float16, float32, float64, bool) | 3 (int32, float32, bool) | 62.5% |
| shape/size | 6-14 various shapes | 3-4 (scalar-like, 2D, 3D, large) | 50-70% |
| requires_grad | 2 (True, False) | 2 (keep both) | 0% |
| device | 2-5 various | 1-2 (default, edge case) | 50-75% |

**Combined reduction**: ~90% fewer combinations

### Atomic Operations Tests

| Parameter | Current Values | Recommended Values | Reduction |
|-----------|---------------|-------------------|-----------|
| dtype | 5 (int32, int64, float16, bfloat16, float32) | 2 (int32, float32) | 60% |
| sem | 3 (acquire, release, acq_rel) | 2 (acquire, acq_rel) | 33% |
| scope | 3 (cta, gpu, sys) | 2 (gpu, sys) | 33% |
| BLOCK_SIZE | 4 (1, 8, 16, 32) | 2 (1, 32) | 50% |

**Combined reduction**: ~85% fewer combinations

---

## Table 11: Test Categorization by Rank Requirements

### Single-Rank Tests (Run with --num_ranks=1 only)
- test_zeros.py, test_ones.py, test_empty.py, test_full.py
- test_zeros_like.py, test_ones_like.py, test_empty_like.py
- test_rand.py, test_randn.py, test_randint.py
- test_arange.py, test_linspace.py
- test_logging.py, test_iris_helpers.py, test_dmabuf_apis.py
- test_get_num_xcc.py

**Total**: ~520,000 test cases → Run 1× instead of 4× = **75% savings**

### Multi-Rank Tests (Run with --num_ranks=2 and --num_ranks=8)
- test_all_reduce.py, test_all_gather.py, test_all_to_all.py
- test_reduce_scatter.py, test_gather.py
- test_process_groups.py
- test_matmul_all_reduce.py, test_matmul_all_gather.py
- test_matmul_reduce_scatter.py, test_all_gather_matmul.py

**Total**: ~10,000 test cases → Run 2× instead of 4× = **50% savings**

### Rank-Scaling Tests (Keep all ranks: 1, 2, 4, 8)
- Performance benchmarks in examples/ directory
- Scaling validation tests

**Total**: ~100 test cases → Run 4× = **No change**

---

## Summary Statistics

### Current State
- **Test files**: 61
- **Test functions**: 261
- **Base test cases**: 530,877
- **CI executions**: 6,370,524
- **CI jobs**: 60
- **CI time**: 180-240 minutes
- **Annual cost**: ~$120,000

### Target State (After All Optimizations)
- **Test files**: ~45 (consolidate duplicates)
- **Test functions**: ~200 (consolidate similar tests)
- **Base test cases**: ~55,000
- **CI executions**: ~88,000
- **CI jobs**: ~10-12
- **CI time**: 8-12 minutes
- **Annual cost**: ~$6,000

### Reductions
- **Test count**: 89.6% reduction
- **Execution count**: 98.6% reduction
- **CI time**: 93-95% reduction
- **Cost**: 95% reduction
