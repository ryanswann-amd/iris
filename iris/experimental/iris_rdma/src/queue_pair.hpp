// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <infiniband/verbs.h>
#include "ibv_utils.hpp"

namespace iris_rdma {

/**
 * @brief Simplified Queue Pair wrapper for host-side operations
 *
 * Unlike the full rocSHMEM QueuePair, this version only maintains
 * metadata needed for RDMA operations from Python/host code.
 */
class QueuePair {
 public:
  /**
   * @brief Constructor
   * @param qp InfiniBand queue pair
   * @param cq InfiniBand completion queue
   * @param dst_rank Destination rank for this QP
   * @param vendor NIC vendor type
   */
  inline QueuePair(struct ibv_qp* qp,
                   struct ibv_cq* cq,
                   int dst_rank,
                   NICVendor vendor)
      : qp_(qp),
        cq_(cq),
        dst_rank_(dst_rank),
        vendor_(vendor),
        lkey_(0),
        rkey_(0) {
    CHECK_NNULL(qp_, "QueuePair: ibv_qp");
    CHECK_NNULL(cq_, "QueuePair: ibv_cq");
    qp_num_ = qp_->qp_num;
    DEBUG_PRINT("QueuePair created: qp_num=%u, dst_rank=%d", qp_num_, dst_rank_);
  }

  /**
   * @brief Destructor
   */
  inline ~QueuePair() {
    DEBUG_PRINT("QueuePair destroyed: qp_num=%u, dst_rank=%d", qp_num_, dst_rank_);
  }

  /**
   * @brief Get QP number
   */
  uint32_t getQPNum() const { return qp_num_; }

  /**
   * @brief Get local key for memory region
   */
  uint32_t getLKey() const { return lkey_; }

  /**
   * @brief Get remote key for destination rank
   */
  uint32_t getRKey() const { return rkey_; }

  /**
   * @brief Get destination rank
   */
  int getDstRank() const { return dst_rank_; }

  /**
   * @brief Set remote key (after exchange)
   */
  void setRKey(uint32_t rkey) { rkey_ = rkey; }

  /**
   * @brief Set local key (from memory registration)
   */
  void setLKey(uint32_t lkey) { lkey_ = lkey; }

  /**
   * @brief Get underlying ibv_qp pointer
   */
  struct ibv_qp* getIBVQP() { return qp_; }

  /**
   * @brief Get underlying ibv_cq pointer
   */
  struct ibv_cq* getIBVCQ() { return cq_; }

  /**
   * @brief Get QP info for Python
   */
  inline QPInfo getInfo() const {
    QPInfo info;
    info.qp_num = qp_num_;
    info.lkey = lkey_;
    info.rkey = rkey_;
    info.dst_rank = dst_rank_;
    return info;
  }

 private:
  struct ibv_qp* qp_;
  struct ibv_cq* cq_;
  int dst_rank_;
  NICVendor vendor_;

  uint32_t qp_num_;
  uint32_t lkey_;
  uint32_t rkey_;
};

}  // namespace iris_rdma

