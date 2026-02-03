# Core Abstractions

Core classes and functions for iris.x tile-level primitives.

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

```{eval-rst}
.. autoclass:: iris.x.TensorView
   :members:
   :undoc-members:
```

## DeviceContext

Holds rank, world size, and heap bases for communication.

```{eval-rst}
.. autoclass:: iris.x.DeviceContext
   :members:
   :undoc-members:
```

## AllReduceConfig

Configuration for selecting all-reduce algorithms.

```{eval-rst}
.. autoclass:: iris.x.AllReduceConfig
   :members:
   :undoc-members:
```

## Helper Functions

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
