# HIP Module API

Low-level HIP runtime integration for AMD GPU device management and memory operations.

This module provides public APIs for device management, device attribute queries, and IPC memory operations.

## Device Management

### count_devices
```{eval-rst}
.. autofunction:: iris.hip.count_devices
```

### set_device
```{eval-rst}
.. autofunction:: iris.hip.set_device
```

## Device Attributes

### get_cu_count
```{eval-rst}
.. autofunction:: iris.hip.get_cu_count
```

### get_num_xcc
```{eval-rst}
.. autofunction:: iris.hip.get_num_xcc
```

### get_wall_clock_rate
```{eval-rst}
.. autofunction:: iris.hip.get_wall_clock_rate
```

## IPC Memory Operations

### get_ipc_handle
```{eval-rst}
.. autofunction:: iris.hip.get_ipc_handle
```

### open_ipc_handle
```{eval-rst}
.. autofunction:: iris.hip.open_ipc_handle
```

