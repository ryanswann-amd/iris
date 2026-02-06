# GPU Concurrency in CI Workflows

## Overview

The Iris CI workflows have been restructured to enable parallel execution of jobs on idle GPUs. This document explains how GPU-based concurrency works in our GitHub Actions workflows.

## Problem Statement

Previously, all CI jobs ran serially even when they used different GPU sets. For example:
- Test A using GPUs 0,1
- Test B using GPUs 2,3
- Test C using GPUs 4,5,6,7

These would run one after another, wasting GPU resources. With 8 GPUs available, tests A, B, and C could all run simultaneously since they don't share any GPUs.

## Solution: GPU-Based Concurrency Groups

We use GitHub Actions' `concurrency` feature with GPU device identifiers as group names. This allows:
- **Parallel execution** when jobs use different GPU sets
- **Serialized execution** when jobs use the same GPU set

### How It Works

Each job defines a concurrency group based on its GPU allocation:

```yaml
concurrency:
  group: gpu-${{ matrix.gpu_devices }}
  cancel-in-progress: false
```

When multiple jobs have the same `group` value, they run sequentially. When jobs have different `group` values, they can run in parallel (subject to runner availability).

## GPU Allocation Patterns

### iris-tests.yml

The test workflow uses four distinct GPU allocation patterns:

| GPU Devices | Tests Using This Pattern | Can Run In Parallel With |
|-------------|-------------------------|-------------------------|
| `0,1` | 1-rank tests | `2,3`, `4,5,6,7` |
| `2,3` | 2-rank tests | `0,1`, `4,5,6,7` |
| `4,5,6,7` | 4-rank tests | `0,1`, `2,3` |
| `0,1,2,3,4,5,6,7` | 8-rank tests | None (uses all GPUs) |

**Parallelization Example:**
- All 1-rank tests (using GPUs 0,1) run sequentially with each other
- All 2-rank tests (using GPUs 2,3) run sequentially with each other
- BUT: 1-rank and 2-rank tests can run in parallel
- AND: 4-rank tests (using GPUs 4,5,6,7) can run alongside both 1-rank AND 2-rank tests

This means up to 3 jobs can run simultaneously (one using 0,1; one using 2,3; one using 4,5,6,7), dramatically improving CI throughput.

### iris-external-validation-test.yml

| Job | Concurrency Group | GPU Usage |
|-----|------------------|-----------|
| `external-validation-test` | `gpu-all` | Unspecified (uses available GPUs) |
| `external-gluon-validation-test` | `gpu-0,1` | GPUs 0,1 |

These two jobs can potentially run in parallel since they use different concurrency groups.

### iris-performance-regression-test.yml

All performance tests use the full 8-GPU system:

```yaml
concurrency:
  group: gpu-0,1,2,3,4,5,6,7
  cancel-in-progress: false
```

Performance benchmarks run sequentially with each other and with any 8-rank tests from other workflows, but can run in parallel with tests using smaller GPU subsets.

## Benefits

1. **Improved CI Throughput**: Tests that don't share GPUs run in parallel
2. **Resource Utilization**: Idle GPUs are utilized instead of sitting idle
3. **Faster Feedback**: Developers get test results faster
4. **Safe Execution**: Jobs that would conflict (same GPUs) are automatically serialized

## Single Runner Architecture

This system is designed for a **single runner** with 8 GPUs. The concurrency groups ensure that:
- Multiple matrix jobs can run simultaneously on the same runner
- GPU conflicts are automatically prevented
- The runner's 8 GPUs are utilized efficiently

## Example Execution Timeline

**Before (Serial Execution):**
```
Job 1 (GPUs 0,1)     [====]
Job 2 (GPUs 2,3)              [====]
Job 3 (GPUs 4,5,6,7)                   [====]
Job 4 (GPUs 0,1,2,3,4,5,6,7)                     [========]
Total time: 4 units
```

**After (Parallel Execution):**
```
Job 1 (GPUs 0,1)     [====]
Job 2 (GPUs 2,3)     [====]
Job 3 (GPUs 4,5,6,7) [====]
Job 4 (GPUs 0,1,2,3,4,5,6,7)   [========]
Total time: 2 units (50% reduction)
```

## Implementation Details

### cancel-in-progress: false

We set `cancel-in-progress: false` to ensure that jobs complete even when new commits are pushed. This is important because:
- CI jobs can take significant time on real hardware
- We want complete test coverage, not just the latest commit
- Jobs are already serialized by GPU usage, so cancellation wouldn't save resources

### Matrix Strategy

The concurrency groups work seamlessly with GitHub Actions matrix strategies:
- Each matrix job gets its own concurrency group based on `matrix.gpu_devices`
- Matrix jobs with different `gpu_devices` values can run in parallel
- Matrix jobs with the same `gpu_devices` value run sequentially

## Monitoring and Verification

To verify parallel execution:
1. Check the workflow run timeline in GitHub Actions
2. Jobs with different concurrency groups should show overlapping execution times
3. Jobs with the same concurrency group should run sequentially

## Future Enhancements

Potential improvements:
- Dynamic GPU allocation based on availability
- More fine-grained GPU sharing for compatible workloads
- Cross-workflow concurrency coordination
- GPU utilization metrics and reporting
