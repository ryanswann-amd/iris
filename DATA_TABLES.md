# Test Suite Review - Data Tables (With Wall Clock Timing Analysis)

This document contains all the raw data and detailed breakdowns from the analysis, now enhanced with **wall clock time estimates** based on test type complexity.

## Executive Timing Summary

**Total Wall Clock Time Estimates**:
- **Single run (1 rank, 1 install)**: 7.8 hours
- **Full CI matrix (4 ranks × 3 installs)**: 94.1 hours (~4 days if sequential)
- **With parallelization (realistic)**: 20-24 hours per CI run

**Time Distribution by Directory**:
- **Unittests**: 93.3 hrs (99.4% of total time)
- **CCL**: 31.4 min (0.5%)
- **Examples**: 16.0 min (0.3%)
- **Ops + X**: <1 min (<0.1%)

**Top Time Consumers** (Top 6 = 88% of total time):
1. test_zeros_like.py: 23.2 hrs (24.7%)
2. test_empty.py: 16.0 hrs (17.0%)
3. test_full.py: 12.8 hrs (13.6%)
4. test_randint.py: 9.9 hrs (10.5%)
5. test_ones.py: 9.9 hrs (10.5%)
6. test_zeros.py: 8.4 hrs (8.9%)

**Critical Insight**: Even though tensor creation tests are individually fast (50ms each), the massive parametrization creates 520K test cases that consume 86.9 hours of CI time - **92% of total CI execution time**.

**Optimization Impact**:
- **Before**: 94.1 hours
- **After**: 38 minutes
- **Reduction**: 99.3%

---

## Table 1: Test Distribution by Directory (With Timing Estimates)

| Directory | Test Files | Base Test Cases | Est. Time (1 rank) | Est. Time (CI: 4 ranks × 3 methods) | % of Total Time | % of Total Tests |
|-----------|-----------|-----------------|-------------------|-------------------------------------|----------------|-----------------|
| **unittests** | 42 | 530,399 | **7.8 hrs** | **93.3 hrs** | **99.4%** | 99.91% |
| ccl | 5 | 309 | 2.6 min | 31.4 min | 0.5% | 0.06% |
| examples | 5 | 146 | 1.3 min | 16.0 min | 0.3% | 0.03% |
| x | 5 | 13 | 0.03 min | 0.3 min | <0.01% | <0.01% |
| ops | 4 | 10 | 0.02 min | 0.2 min | <0.01% | <0.01% |
| **TOTAL** | **61** | **530,877** | **7.8 hrs** | **94.1 hrs** | **100%** | **100%** |

*Note: Timing estimates based on test type complexity (tensor creation: 50ms, atomic ops: 100ms, RMA: 200ms, collective: 500ms, benchmarks: 1s per test). Actual CI times may vary based on hardware, system load, and parallelization.*

*CI total assumes sequential execution of all rank configs and install methods (not parallelized across different test directories).*

---

## Table 2: Top 30 Test Files by Wall Clock Time

| Rank | Test File | Test Cases | Type | Est. Time (1 rank) | Est. CI Time (×12) | % of Total Time |
|------|-----------|-----------|------|-------------------|-------------------|----------------|
| 1 | **test_zeros_like.py** | 139,216 | tensor | **116.0 min** | **23.2 hrs** | **24.7%** |
| 2 | **test_empty.py** | 95,872 | tensor | **79.9 min** | **16.0 hrs** | **17.0%** |
| 3 | **test_full.py** | 76,608 | tensor | **63.8 min** | **12.8 hrs** | **13.6%** |
| 4 | **test_randint.py** | 59,360 | tensor | **49.5 min** | **9.9 hrs** | **10.5%** |
| 5 | **test_ones.py** | 59,136 | tensor | **49.3 min** | **9.9 hrs** | **10.5%** |
| 6 | **test_zeros.py** | 50,176 | tensor | **41.8 min** | **8.4 hrs** | **8.9%** |
| 7 | **test_randn.py** | 17,724 | tensor | **14.8 min** | **3.0 hrs** | **3.2%** |
| 8 | **test_rand.py** | 17,724 | tensor | **14.8 min** | **3.0 hrs** | **3.2%** |
| 9 | **test_copy_gluon.py** | 4,368 | rma | **14.6 min** | **2.9 hrs** | **3.1%** |
| 10 | **test_copy_triton.py** | 4,368 | rma | **14.6 min** | **2.9 hrs** | **3.1%** |
| 11 | test_linspace.py | 3,840 | tensor | 3.2 min | 38.4 min | 0.7% |
| 12 | test_process_groups.py | 282 | collective | 2.4 min | 28.2 min | 0.5% |
| 13 | test_message_passing.py | 72 | benchmark | 1.2 min | 14.4 min | 0.3% |
| 14 | test_arange.py | 609 | tensor | 0.5 min | 6.1 min | 0.1% |
| 15 | test_atomic_add_gluon.py | 180 | atomic | 0.3 min | 3.6 min | 0.1% |
| 16 | test_atomic_add_triton.py | 180 | atomic | 0.3 min | 3.6 min | 0.1% |
| 17 | test_atomic_add_bench.py | 42 | benchmark | 0.7 min | 8.4 min | 0.1% |
| 18 | test_flash_decode.py | 8 | benchmark | 0.1 min | 1.6 min | <0.1% |
| 19 | test_all_load_bench.py | 16 | benchmark | 0.3 min | 3.2 min | <0.1% |
| 20 | test_load_bench.py | 8 | benchmark | 0.1 min | 1.6 min | <0.1% |
| 21 | test_all_reduce.py | 18 | collective | 0.2 min | 1.8 min | <0.1% |
| 22 | test_atomic_and_gluon.py | 72 | atomic | 0.1 min | 1.4 min | <0.1% |
| 23 | test_atomic_and_triton.py | 72 | atomic | 0.1 min | 1.4 min | <0.1% |
| 24 | test_atomic_max_gluon.py | 72 | atomic | 0.1 min | 1.4 min | <0.1% |
| 25 | test_atomic_max_triton.py | 72 | atomic | 0.1 min | 1.4 min | <0.1% |
| 26 | test_atomic_min_gluon.py | 72 | atomic | 0.1 min | 1.4 min | <0.1% |
| 27 | test_atomic_min_triton.py | 72 | atomic | 0.1 min | 1.4 min | <0.1% |
| 28 | test_atomic_or_gluon.py | 72 | atomic | 0.1 min | 1.4 min | <0.1% |
| 29 | test_atomic_or_triton.py | 72 | atomic | 0.1 min | 1.4 min | <0.1% |
| 30 | test_atomic_xor_gluon.py | 72 | atomic | 0.1 min | 1.4 min | <0.1% |

**Top 10 files by time**: ~445 min (7.4 hrs) = **94.7% of total test time**

**Key Finding**: The massive parametrization in tensor creation tests (test_zeros_like, test_empty, etc.) not only creates the most test cases but also consumes the most wall clock time. Even though each test is fast (~50ms), the sheer volume (139K+ tests in test_zeros_like alone) results in nearly 2 hours of execution time per rank configuration.

---

## Table 2A: Wall Clock Time Analysis by Test Type

| Test Type | Test Count | Time/Test | Total Time (1 rank) | % of Total Time | Example Tests |
|-----------|-----------|-----------|---------------------|----------------|---------------|
| **Tensor Creation** | **520,265** | **50ms** | **7.2 hrs** | **92.3%** | zeros, ones, empty, full, rand, etc. |
| RMA Operations | 8,800 | 200ms | 29.3 min | 6.2% | copy, get, put, load, store |
| Collective Ops | 322 | 500ms | 2.7 min | 0.6% | all_reduce, all_gather, process_groups |
| Atomic Ops | 1,260 | 100ms | 2.1 min | 0.4% | atomic_add, atomic_and, atomic_max |
| Benchmarks | 146 | 1s | 2.4 min | 0.5% | load_bench, flash_decode, message_passing |
| **TOTAL** | **530,793** | **-** | **7.8 hrs** | **100%** | **All tests** |

**Key Insight**: Tensor creation tests dominate both test count (98%) AND wall clock time (92%). These are also the tests with the most redundant parametrization.

**Impact on CI**: With 4 ranks × 3 install methods, tensor creation tests alone consume **86.9 hours** of the **94.1 hour** total CI time.

---

## Table 2B: Critical Time Consumers - Detailed Breakdown

### Unittests Directory (7.8 hrs per rank, 93.3 hrs total CI time)

| Category | Files | Tests | Time (1 rank) | CI Time (×12) | % of CI |
|----------|-------|-------|---------------|---------------|---------|
| **Tensor creation (zeros/ones/empty/full)** | 4 | 420,272 | **5.9 hrs** | **70.2 hrs** | **74.6%** |
| Random generation (rand/randn/randint) | 3 | 94,808 | 1.3 hrs | 15.8 hrs | 16.8% |
| RMA operations (copy/get/put/load/store) | 14 | 8,800 | 29.3 min | 5.9 hrs | 6.2% |
| Atomic operations | 17 | 1,260 | 2.1 min | 25.2 min | 0.4% |
| Other (linspace, arange, helpers) | 4 | 5,309 | 5.4 min | 1.1 hrs | 1.2% |
| **Subtotal** | **42** | **530,449** | **7.8 hrs** | **93.3 hrs** | **99.1%** |

### CCL Directory (2.6 min per rank, 31.4 min total CI time)
| Test File | Tests | Time (1 rank) | CI Time (×12) |
|-----------|-------|---------------|---------------|
| test_process_groups.py | 282 | 2.4 min | 28.2 min |
| test_all_reduce.py | 18 | 9 sec | 1.8 min |
| Others (all_gather, all_to_all) | 19 | 10 sec | 2.0 min |
| **Subtotal** | **319** | **2.6 min** | **31.4 min** |

### Examples Directory (1.3 min per rank, 16.0 min total CI time)
| Test File | Tests | Time (1 rank) | CI Time (×12) |
|-----------|-------|---------------|---------------|
| test_message_passing.py | 72 | 1.2 min | 14.4 min |
| test_atomic_add_bench.py | 42 | 42 sec | 8.4 min |
| Others (flash_decode, load_bench) | 32 | 32 sec | 6.4 min |
| **Subtotal** | **146** | **1.3 min** | **16.0 min** |

### Ops Directory (<1 min per rank)
Small set of collective operation tests, negligible time impact.

### X Directory (<1 min per rank)
Small set of simplified collective tests, negligible time impact.



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

## Table 6: Savings Breakdown by Optimization (With Timing)

| Optimization | Current | After | Reduction | % Savings |
|-------------|---------|-------|-----------|-----------|
| **Base Test Count** |
| Reduce parametrization | 530,877 | 65,000 | 465,877 | 87.8% |
| Merge gluon/triton | 65,000 | 55,000 | 10,000 | 15.4% |
| **Final base count** | **530,877** | **55,000** | **475,877** | **89.6%** |
| | | | | |
| **Wall Clock Time (1 rank, 1 install)** |
| Reduce parametrization | 7.8 hrs | 57 min | 6.9 hrs | 87.8% |
| Merge gluon/triton | 57 min | 48 min | 9 min | 15.8% |
| **Final time (1 rank)** | **7.8 hrs** | **48 min** | **6.9 hrs** | **89.7%** |
| | | | | |
| **CI Total Time** |
| Single install method | 94.1 hrs | 31.4 hrs | 62.7 hrs | 66.6% |
| Smart rank configs | 31.4 hrs | 5.2 hrs | 26.2 hrs | 83.4% |
| Apply test reduction | 5.2 hrs | 38 min | 4.4 hrs | 87.8% |
| **Final CI time** | **94.1 hrs** | **38 min** | **93.5 hrs** | **99.3%** |
| | | | | |

**Time Savings Summary**:
- **Current**: 94.1 hours per CI run (full matrix)
- **Optimized**: 38 minutes per CI run  
- **Reduction**: 99.3% (from ~4 days to ~40 minutes)

**Key Drivers**:
1. Reducing parametrization: 87.8% time reduction (the biggest win)
2. Single install method: 66.6% execution reduction
3. Smart rank configs: 83.4% additional reduction



---

## Table 7: Estimated Cost Savings (Based on Wall Clock Time)

### Assumptions
- Self-hosted AMD GPUs (8× MI300X per CI run)
- Average PR triggers 2-3 CI runs
- ~500 PRs per year
- GPU time cost: $3/hour per GPU
- **Current CI time: 94.1 hours per run** (based on wall clock estimates)
- **Optimized CI time: 0.6 hours (38 min) per run**

### Current Annual Cost
```
Wall clock time per run: 94.1 hours
But with parallelization across 5 test directories, estimated actual time: ~20-24 hours
Conservative estimate: 20 hours per CI run

500 PRs × 2.5 runs × 20 hours × 8 GPUs × $3/hour = $600,000/year
```

### After Optimization
```
Wall clock time per run: 0.6 hours (38 min)
With some parallelization: ~15-20 min actual runtime

500 PRs × 2.5 runs × 0.33 hours × 8 GPUs × $3/hour = $9,900/year
```

### Savings
```
$600,000 - $9,900 = $590,100/year (98.4% reduction)
```

**Note**: The original estimate of $120K/year assumed significant parallelization. With timing data showing 94.1 hours of sequential test execution, the actual cost is likely much higher if tests run sequentially or with limited parallelization. The savings could be even more substantial than originally estimated.

**Conservative Estimate** (assuming current CI is already highly parallelized):
```
Current: $120,000/year (4-hour runs with good parallelization)
Optimized: $6,000/year (0.33-hour runs)
Savings: $114,000/year (95% reduction)
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

### Current State (Based on Wall Clock Timing)
- **Test files**: 61
- **Test functions**: 261
- **Base test cases**: 530,877
- **Wall clock time (1 rank)**: 7.8 hours
- **Wall clock time (CI full matrix)**: 94.1 hours
- **Estimated CI time (with parallelization)**: 20-24 hours
- **CI jobs**: 60
- **Annual cost**: ~$120,000 - $600,000 (depending on parallelization)

### Target State (After All Optimizations)
- **Test files**: ~45 (consolidate duplicates)
- **Test functions**: ~200 (consolidate similar tests)
- **Base test cases**: ~55,000
- **Wall clock time (1 rank)**: 48 minutes
- **Wall clock time (CI optimized)**: 38 minutes
- **CI jobs**: ~10-12
- **Annual cost**: ~$6,000

### Reductions
- **Test count**: 89.6% reduction
- **Wall clock time**: 99.3% reduction (94.1 hrs → 38 min)
- **CI time**: 93-99% reduction (depending on baseline)
- **Cost**: 95-99% reduction

---

## Appendix: Timing Methodology

### How Wall Clock Times Were Estimated

Since actual CI logs were not accessible via GitHub API, timing estimates were calculated based on test type complexity:

| Test Type | Time per Test | Rationale |
|-----------|--------------|-----------|
| **Tensor Creation** | 50ms | Simple GPU memory allocation and initialization. Fast local operations. |
| **Atomic Operations** | 100ms | GPU kernel execution with atomic operations. Slightly slower than basic ops. |
| **RMA Operations** | 200ms | Remote memory access requires inter-GPU communication and synchronization. |
| **Collective Operations** | 500ms | Multi-GPU synchronization, barriers, and data movement across ranks. |
| **Benchmarks** | 1s | Performance measurements typically run multiple iterations. |

### Validation of Estimates

These estimates are conservative and based on typical GPU operation timings:
- **Memory allocation**: ~1-10ms
- **Kernel launch overhead**: ~10-50ms
- **Simple kernel execution**: ~10-100ms
- **Inter-GPU communication**: ~100-500ms
- **Collective operations with barriers**: ~200-1000ms

### Actual CI Times May Vary

Real CI execution times depend on:
- GPU hardware (MI300X, MI350X, MI355X specifications)
- System load and contention
- Network latency for multi-GPU communication
- Parallelization strategy (tests run in sequence vs parallel)
- Container/environment startup overhead
- Test isolation and cleanup time

**Recommendation**: Measure actual CI times using `--durations=0` flag in pytest to get precise timing data for validation.

### How to Get Actual Timing Data

Run tests with detailed timing output:
```bash
pytest tests/unittests/test_zeros_like.py --durations=0 -v
```

This will show:
- Slowest test durations
- Average time per test
- Total execution time

Compare actual measurements against these estimates to refine the analysis.
