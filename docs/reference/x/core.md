# Core Abstractions

Core classes and functions for iris.x tile-level primitives.

**Note:** `iris.DeviceContext` (from the main iris module) should be used for device-side context, not documented here as it's part of the main iris API.

## TileView

Represents a tile's position and size in a 2D grid.

```{eval-rst}
.. autoclass:: iris.x.TileView
   :members:
   :undoc-members:
```

## Tile (deprecated)

Legacy tile representation. Use TileView instead.

```{eval-rst}
.. autoclass:: iris.x.Tile
   :members:
   :undoc-members:
```

## TensorView

Represents a tensor's memory layout (pointer, shape, strides).

**Note:** Use `iris.x.make_tensor_view()` factory function to create TensorView instances in JIT context.

```{eval-rst}
.. autoclass:: iris.x.TensorView
   :members:
   :undoc-members:
```

## AllReduceConfig

Configuration for selecting all-reduce algorithms.

Takes an integer variant code (0=atomic, 1=ring, 2=one_shot, 3=two_shot, 4=spinlock) and a locks pointer.

```{eval-rst}
.. autoclass:: iris.x.AllReduceConfig
   :members:
   :undoc-members:
```

## Helper Functions

### make_tensor_view

Factory function to create TensorView in JIT context.

```{eval-rst}
.. autofunction:: iris.x.make_tensor_view
```

### tile_layout

Compute the memory layout for a tile.

```{eval-rst}
.. autofunction:: iris.x.tile_layout
```

### tile_ptr

Compute pointer to tile data in a tensor.

```{eval-rst}
.. autofunction:: iris.x.tile_ptr
```

### offset_ptr

Offset a pointer by tile coordinates.

```{eval-rst}
.. autofunction:: iris.x.offset_ptr
```
