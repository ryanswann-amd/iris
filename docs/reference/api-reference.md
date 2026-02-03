# API Reference

Explore Iris APIs. The reference is broken down into focused sections to mirror common workflows:

- The `Iris` class itself (constructor and helper utilities)
- Tensor-like creation methods on the `Iris` context
- Triton device-side functions for remote memory ops and atomics
- Collective communication operations (CCL)
- Fused GEMM+CCL operations
- Device-side tile-level primitives
- Experimental Gluon APIs (using `@aggregate` and `@gluon.jit`)

Use the links below to navigate:

- [Triton](triton/overview.md)
  - [Iris Class](triton/class.md)
  - [Tensor Creation](triton/tensor-creation.md)
  - [Device Functions](triton/device-functions.md)
- [iris.ccl - Collective Communication](ccl/overview.md)
  - [Operations](ccl/operations.md)
  - [Configuration](ccl/config.md)
- [iris.ops - Fused GEMM+CCL](ops/overview.md)
  - [Operations](ops/operations.md)
  - [Configuration](ops/config.md)
- [iris.x - Device-Side Primitives](x/overview.md)
  - [Core Abstractions](x/core.md)
  - [Operations](x/operations.md)
- [Gluon (Experimental)](gluon/overview.md)
  - [Iris Class](gluon/class.md)
  - [Tensor Creation](gluon/tensor-creation.md)
  - [Device Functions](gluon/device-functions.md)

