# Device-Side Operations

Device-side collective operations provided by iris.x for use in Triton kernels.

## All-Reduce Operations

### all_reduce_atomic

Atomic-based all-reduce implementation (default algorithm).

```{eval-rst}
.. autofunction:: iris.x.all_reduce_atomic
```

### all_reduce_ring

Ring algorithm for all-reduce.

```{eval-rst}
.. autofunction:: iris.x.all_reduce_ring
```

### all_reduce_two_shot

Two-shot algorithm for all-reduce.

```{eval-rst}
.. autofunction:: iris.x.all_reduce_two_shot
```

### all_reduce_one_shot

One-shot algorithm for all-reduce.

```{eval-rst}
.. autofunction:: iris.x.all_reduce_one_shot
```

### all_reduce_spinlock

Spinlock-based all-reduce implementation.

```{eval-rst}
.. autofunction:: iris.x.all_reduce_spinlock
```

## Other Collective Operations

### all_gather

Gather data from all ranks and distribute to all ranks.

```{eval-rst}
.. autofunction:: iris.x.all_gather
```

### all_to_all

Scatter data from all ranks to all ranks.

```{eval-rst}
.. autofunction:: iris.x.all_to_all
```

### reduce_scatter

Reduce values across all ranks and scatter the result.

```{eval-rst}
.. autofunction:: iris.x.reduce_scatter
```

### gather

Point-to-point gather operation from a source rank.

```{eval-rst}
.. autofunction:: iris.x.gather
```
