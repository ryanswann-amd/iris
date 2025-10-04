# HIP Module API

Low-level HIP runtime integration for AMD GPU device management and memory operations.

```{eval-rst}
.. automodule:: iris.hip
   :members:
   :undoc-members:
   :show-inheritance:
```

## Device Management

### count_devices
```{eval-rst}
.. autofunction:: iris.hip.count_devices
```

### set_device
```{eval-rst}
.. autofunction:: iris.hip.set_device
```

### get_device_id
```{eval-rst}
.. autofunction:: iris.hip.get_device_id
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

### get_rocm_version
```{eval-rst}
.. autofunction:: iris.hip.get_rocm_version
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

### hipIpcMemHandle_t
```{eval-rst}
.. autoclass:: iris.hip.hipIpcMemHandle_t
   :members:
```

## Error Handling

### hip_try
```{eval-rst}
.. autofunction:: iris.hip.hip_try
```

