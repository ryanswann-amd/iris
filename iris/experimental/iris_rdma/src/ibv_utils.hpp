// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <infiniband/verbs.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace iris_rdma {

// Error checking macros
#define CHECK_ZERO(expr, msg)                                           \
  do {                                                                  \
    int ret = (expr);                                                   \
    if (ret != 0) {                                                     \
      fprintf(stderr, "[ERROR] %s failed with code %d: %s\n", msg, ret, \
              strerror(ret));                                           \
      abort();                                                          \
    }                                                                   \
  } while (0)

#define CHECK_NNULL(ptr, msg)                             \
  do {                                                    \
    if ((ptr) == nullptr) {                               \
      fprintf(stderr, "[ERROR] %s returned NULL\n", msg); \
      abort();                                            \
    }                                                     \
  } while (0)

#define DEBUG_PRINT(fmt, ...)                                       \
  do {                                                              \
    if (getenv("IRIS_RDMA_DEBUG")) {                                \
      fprintf(stderr, "[IRIS_RDMA_DEBUG] " fmt "\n", ##__VA_ARGS__); \
    }                                                               \
  } while (0)

// Vendor detection
enum class NICVendor { NONE, IONIC, BNXT, MLX5 };

// QP destination info for connection
struct QPDestInfo {
  int lid;
  int qpn;
  int psn;
  union ibv_gid gid;
};

// QP metadata exposed to Python
struct QPInfo {
  uint32_t qp_num;
  uint32_t lkey;
  uint32_t rkey;
  int dst_rank;
};

// Helper functions
inline void dump_ibv_device(struct ibv_device* device) {
  DEBUG_PRINT("IBV Device: %s", ibv_get_device_name(device));
}

inline void dump_ibv_context(struct ibv_context* ctx) {
  DEBUG_PRINT("IBV Context: device=%s", ctx->device->name);
}

inline void dump_ibv_pd(struct ibv_pd* pd) {
  DEBUG_PRINT("IBV PD: handle=%u", pd->handle);
}

inline void dump_ibv_port_attr(struct ibv_port_attr* attr) {
  DEBUG_PRINT("Port Attr: state=%d, lid=%d, link_layer=%d, active_mtu=%d",
              attr->state, attr->lid, attr->link_layer, attr->active_mtu);
}

inline int ibv_mtu_to_int(enum ibv_mtu mtu) {
  switch (mtu) {
    case IBV_MTU_256:
      return 256;
    case IBV_MTU_512:
      return 512;
    case IBV_MTU_1024:
      return 1024;
    case IBV_MTU_2048:
      return 2048;
    case IBV_MTU_4096:
      return 4096;
    default:
      fprintf(stderr, "[ERROR] Invalid ibv_mtu\n");
      return 0;
  }
}

}  // namespace iris_rdma

