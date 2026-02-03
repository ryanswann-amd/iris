# Fused Operations

Fused GEMM+CCL operations provided by iris.ops.

## matmul_all_reduce

Fused matrix multiplication and all-reduce.

Computes: `output = all_reduce(A @ B + bias)`

```{eval-rst}
.. autofunction:: iris.ops.matmul_all_reduce
```

### Preamble

Pre-allocate workspace for matmul_all_reduce.

```{eval-rst}
.. autofunction:: iris.ops.matmul_all_reduce_preamble
```

## all_gather_matmul

Fused all-gather and matrix multiplication.

Computes: `output = all_gather(A_sharded) @ B + bias`

```{eval-rst}
.. autofunction:: iris.ops.all_gather_matmul
```

### Preamble

Pre-allocate workspace for all_gather_matmul.

```{eval-rst}
.. autofunction:: iris.ops.all_gather_matmul_preamble
```

## matmul_all_gather

Fused matrix multiplication and all-gather.

Computes: `output = all_gather(A @ B + bias)` along M dimension

```{eval-rst}
.. autofunction:: iris.ops.matmul_all_gather
```

## matmul_reduce_scatter

Fused matrix multiplication and reduce-scatter.

Computes: `output = reduce_scatter(A @ B + bias)` along N dimension

```{eval-rst}
.. autofunction:: iris.ops.matmul_reduce_scatter
```

### Preamble

Pre-allocate workspace for matmul_reduce_scatter.

```{eval-rst}
.. autofunction:: iris.ops.matmul_reduce_scatter_preamble
```

## OpsNamespace

Namespace class for accessing fused operations through `shmem.ops`.

```{eval-rst}
.. autoclass:: iris.ops.OpsNamespace
   :members:
   :undoc-members:
```
