// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once 

#include <dlfcn.h>
#include <infiniband/verbs.h>
#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <cerrno>
#include <stdexcept>

#include "ibv_utils.hpp"
#include "queue_pair.hpp"
#include "torch_bootstrap.hpp"

// Vendor-specific headers
#ifdef HAVE_MLX5
#include <infiniband/mlx5dv.h>
#endif

#ifdef HAVE_BNXT
#include <infiniband/bnxt_re-abi.h>
#endif

namespace iris_rdma {

/**
 * @brief Main network backend for InfiniBand setup
 *
 * Handles:
 * - Device detection and initialization
 * - Protection domain creation
 * - Queue pair creation and state transitions
 * - Memory registration
 * - QP connection info exchange
 */
class NetworkBackend {
 public:
  /**
   * @brief Constructor
   * @param bootstrap PyTorch bootstrap for cross-rank communication
   * @param device_name Optional device name (NULL for auto-detect)
   */
  NetworkBackend(std::shared_ptr<TorchBootstrap> bootstrap,
                 const char* device_name = nullptr)
      : bootstrap_(bootstrap),
        requested_dev_(device_name),
        context_(nullptr),
        pd_orig_(nullptr),
        pd_parent_(nullptr),
        vendor_(NICVendor::NONE),
        port_(1),
        gid_index_(0),
        heap_mr_(nullptr),
        heap_base_(0),
        heap_size_(0),
        mlx5dv_handle_(nullptr),
        bnxtdv_handle_(nullptr) {
    if (!bootstrap_) {
      throw std::runtime_error("Bootstrap cannot be null");
    }
    rank_ = bootstrap_->getRank();
    world_size_ = bootstrap_->getWorldSize();
    DEBUG_PRINT("NetworkBackend created: rank=%d, world_size=%d", rank_, world_size_);
  }

  /**
   * @brief Destructor - cleanup InfiniBand resources
   */
  ~NetworkBackend() {
    DEBUG_PRINT("NetworkBackend cleanup started");
    
    qps_.clear();
    
    for (auto* cq : cqs_) {
      if (cq) {
        ibv_destroy_cq(cq);
      }
    }
    cqs_.clear();
    
    if (heap_mr_) {
      ibv_dereg_mr(heap_mr_);
      heap_mr_ = nullptr;
    }
    
    if (pd_parent_) {
      ibv_dealloc_pd(pd_parent_);
      pd_parent_ = nullptr;
    }
    
    if (pd_orig_) {
      ibv_dealloc_pd(pd_orig_);
      pd_orig_ = nullptr;
    }
    
    if (context_) {
      ibv_close_device(context_);
      context_ = nullptr;
    }
    
    if (mlx5dv_handle_) {
      dlclose(mlx5dv_handle_);
      mlx5dv_handle_ = nullptr;
    }
    
    if (bnxtdv_handle_) {
      dlclose(bnxtdv_handle_);
      bnxtdv_handle_ = nullptr;
    }
    
    DEBUG_PRINT("NetworkBackend cleanup completed");
  }

  /**
   * @brief Initialize the network (setup QPs, transition to RTS)
   */
  void init() {
    DEBUG_PRINT("NetworkBackend::init() started");
    
    autodetectDVLibs();
    openIBDevice();
    createQueues();
    exchangeQPDestInfo();
    modifyQPsResetToInit();
    modifyQPsInitToRTR();
    modifyQPsRTRToRTS();
    bootstrap_->barrier();
    
    DEBUG_PRINT("NetworkBackend::init() completed");
  }

  /**
   * @brief Register memory for RDMA
   * @param ptr Pointer to memory region
   * @param size Size in bytes
   */
  void registerMemory(void* ptr, size_t size) {
    DEBUG_PRINT("Registering memory: ptr=%p, size=%zu", ptr, size);

    int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                 IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

    heap_mr_ = ibv_reg_mr(pd_orig_, ptr, size, access);
    if (heap_mr_ == nullptr) {
      int err = errno;
      fprintf(stderr, "[ERROR] ibv_reg_mr returned NULL for ptr=%p, size=%zu, errno=%d (%s)\n", 
              ptr, size, err, strerror(err));
      char error_msg[256];
      snprintf(error_msg, sizeof(error_msg), 
               "ibv_reg_mr failed with errno %d (%s) - GPUDirect RDMA may not be enabled", 
               err, strerror(err));
      throw std::runtime_error(error_msg);
    }

    // Store local heap base
    heap_base_ = reinterpret_cast<uint64_t>(ptr);
    heap_size_ = size;

    // Exchange remote keys
    rkeys_.resize(world_size_);
    std::vector<uint32_t> all_rkeys(world_size_);
    all_rkeys[rank_] = heap_mr_->rkey;
    bootstrap_->allGather(all_rkeys.data(), sizeof(uint32_t));
    for (int i = 0; i < world_size_; i++) {
      rkeys_[i] = all_rkeys[i];
    }

    // Exchange heap base addresses (collective operation)
    remote_heap_bases_.resize(world_size_);
    std::vector<uint64_t> all_heap_bases(world_size_);
    all_heap_bases[rank_] = heap_base_;
    bootstrap_->allGather(all_heap_bases.data(), sizeof(uint64_t));
    for (int i = 0; i < world_size_; i++) {
      remote_heap_bases_[i] = all_heap_bases[i];
    }

    // Update QPs with lkey and rkey
    uint32_t lkey = heap_mr_->lkey;
    for (int i = 0; i < world_size_; i++) {
      if (i < qps_.size() && qps_[i]) {
        qps_[i]->setLKey(lkey);
        qps_[i]->setRKey(rkeys_[i]);
      }
    }

    DEBUG_PRINT("Memory registered: lkey=%u, rkey=%u, heap_base=%p", 
                lkey, heap_mr_->rkey, ptr);
  }

  /**
   * @brief Get queue pair for destination rank
   * @param dst_rank Destination rank
   * @return Pointer to QueuePair object
   */
  QueuePair* getQP(int dst_rank) {
    if (dst_rank >= 0 && dst_rank < qps_.size()) {
      return qps_[dst_rank].get();
    }
    return nullptr;
  }

  /**
   * @brief Get QP info for Python
   * @param dst_rank Destination rank
   * @return QPInfo structure
   */
  QPInfo getQPInfo(int dst_rank) {
    QueuePair* qp = getQP(dst_rank);
    if (qp) {
      return qp->getInfo();
    }
    return QPInfo{0, 0, 0, dst_rank};
  }




  /**
   * @brief Get rank
   */
  int getRank() const { return rank_; }

  /**
   * @brief Get world size
   */
  int getWorldSize() const { return world_size_; }

  /**
   * @brief Get remote heap base address for a rank
   * @param rank Remote rank
   * @return Remote heap base address (0 if not registered)
   */
  uint64_t getRemoteHeapBase(int rank) const {
    if (rank >= 0 && rank < remote_heap_bases_.size()) {
      return remote_heap_bases_[rank];
    }
    return 0;
  }

  /**
   * @brief Get local heap base address
   * @return Local heap base address (0 if not registered)
   */
  uint64_t getHeapBase() const { return heap_base_; }

  /**
   * @brief Get heap size
   * @return Heap size in bytes (0 if not registered)
   */
  size_t getHeapSize() const { return heap_size_; }

  /**
   * @brief RDMA Write operation
   * @param dst_rank Destination rank
   * @param local_addr Local buffer address
   * @param remote_addr Remote buffer address
   * @param size Size in bytes
   * @param wr_id Work request ID (for completion tracking)
   * @return 0 on success, non-zero on error
   */
  int rdmaWrite(int dst_rank, void* local_addr, uint64_t remote_addr, 
                size_t size, uint64_t wr_id = 0) {
    QueuePair* qp = getQP(dst_rank);
    if (!qp) {
      return -1;
    }

    struct ibv_sge sge;
    sge.addr = (uintptr_t)local_addr;
    sge.length = size;
    sge.lkey = qp->getLKey();

    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = qp->getRKey();

    struct ibv_send_wr* bad_wr;
    int ret = ibv_post_send(qp->getIBVQP(), &wr, &bad_wr);
    
    DEBUG_PRINT("RDMA Write to rank %d: local=%p remote=%lx size=%zu ret=%d", 
                dst_rank, local_addr, remote_addr, size, ret);
    
    return ret;
  }

  /**
   * @brief RDMA Read operation
   * @param dst_rank Destination rank
   * @param local_addr Local buffer address
   * @param remote_addr Remote buffer address
   * @param size Size in bytes
   * @param wr_id Work request ID (for completion tracking)
   * @return 0 on success, non-zero on error
   */
  int rdmaRead(int dst_rank, void* local_addr, uint64_t remote_addr,
               size_t size, uint64_t wr_id = 0) {
    QueuePair* qp = getQP(dst_rank);
    if (!qp) {
      return -1;
    }

    struct ibv_sge sge;
    sge.addr = (uintptr_t)local_addr;
    sge.length = size;
    sge.lkey = qp->getLKey();

    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = qp->getRKey();

    struct ibv_send_wr* bad_wr;
    int ret = ibv_post_send(qp->getIBVQP(), &wr, &bad_wr);
    
    DEBUG_PRINT("RDMA Read from rank %d: local=%p remote=%lx size=%zu ret=%d", 
                dst_rank, local_addr, remote_addr, size, ret);
    
    return ret;
  }

  /**
   * @brief Poll completion queue for RDMA operations
   * @param dst_rank Destination rank (to poll specific CQ)
   * @param max_completions Maximum number of completions to poll
   * @return Number of completions polled (negative on error)
   */
  int pollCQ(int dst_rank, int max_completions = 1) {
    QueuePair* qp = getQP(dst_rank);
    if (!qp) {
      return -1;
    }

    struct ibv_wc wc[16];
    int num_to_poll = (max_completions < 16) ? max_completions : 16;
    int n = ibv_poll_cq(qp->getIBVCQ(), num_to_poll, wc);
    
    if (n < 0) {
      DEBUG_PRINT("CQ poll error for rank %d", dst_rank);
      return n;
    }
    
    // Check for errors in completions
    for (int i = 0; i < n; i++) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        fprintf(stderr, "[ERROR] Work completion failed: status=%d (%s) wr_id=%lu\n",
                wc[i].status, ibv_wc_status_str(wc[i].status), wc[i].wr_id);
        return -1;
      }
    }
    
    DEBUG_PRINT("Polled %d completions from rank %d", n, dst_rank);
    return n;
  }



 private:
  // Bootstrap
  std::shared_ptr<TorchBootstrap> bootstrap_;
  int rank_;
  int world_size_;

  // Device configuration
  const char* requested_dev_;
  struct ibv_context* context_;
  struct ibv_pd* pd_orig_;
  struct ibv_pd* pd_parent_;  // For MLX5/IONIC
  NICVendor vendor_;

  // Port configuration
  struct ibv_port_attr portinfo_;
  union ibv_gid gid_;
  int port_;
  int gid_index_;

  // Memory registration
  struct ibv_mr* heap_mr_;
  std::vector<uint32_t> rkeys_;  // Remote keys from all ranks
  uint64_t heap_base_;  // Local heap base address
  size_t heap_size_;  // Local heap size
  std::vector<uint64_t> remote_heap_bases_;  // Heap base addresses from all ranks

  // Queue pairs
  std::vector<std::unique_ptr<QueuePair>> qps_;
  std::vector<struct ibv_cq*> cqs_;
  std::vector<QPDestInfo> dest_info_;

  // Dynamic library handles for vendor-specific libraries
  void* mlx5dv_handle_;
  void* bnxtdv_handle_;

  // Setup functions (extracted from rocSHMEM)

  // Vendor-specific init
  void autodetectDVLibs() {
    DEBUG_PRINT("Auto-detecting vendor libraries...");

    // Try MLX5
    if (mlx5DVDLInit() == 0) {
      vendor_ = NICVendor::MLX5;
      DEBUG_PRINT("Detected MLX5 vendor");
      return;
    }

    // Try BNXT
    if (bnxtDVDLInit() == 0) {
      vendor_ = NICVendor::BNXT;
      DEBUG_PRINT("Detected BNXT vendor");
      return;
    }

    // Default to standard verbs
    vendor_ = NICVendor::NONE;
    DEBUG_PRINT("Using standard InfiniBand verbs");
  }

  int mlx5DVDLInit() {
    mlx5dv_handle_ = dlopen("libmlx5.so", RTLD_NOW);
    if (!mlx5dv_handle_) {
      mlx5dv_handle_ = dlopen("libmlx5.so.1", RTLD_NOW);
    }

    if (!mlx5dv_handle_) {
      DEBUG_PRINT("Could not open libmlx5.so");
      return -1;
    }

    return 0;
  }

  int bnxtDVDLInit() {
    bnxtdv_handle_ = dlopen("libbnxt_re.so", RTLD_NOW);
    if (!bnxtdv_handle_) {
      bnxtdv_handle_ = dlopen("/usr/local/lib/libbnxt_re.so", RTLD_NOW);
    }

    if (!bnxtdv_handle_) {
      DEBUG_PRINT("Could not open libbnxt_re.so");
      return -1;
    }

    return 0;
  }

  void openIBDevice() {
    DEBUG_PRINT("Opening InfiniBand device...");

    struct ibv_device** device_list = nullptr;
    struct ibv_device* device = nullptr;
    int num_devices = 0;

    device_list = ibv_get_device_list(&num_devices);
    CHECK_NNULL(device_list, "ibv_get_device_list");

    if (num_devices == 0) {
      throw std::runtime_error("No InfiniBand devices found");
    }

    // Select device
    device = device_list[0];  // Default to first device

    if (requested_dev_) {
      for (int i = 0; i < num_devices; i++) {
        const char* dev_name = ibv_get_device_name(device_list[i]);
        CHECK_NNULL(dev_name, "ibv_get_device_name");

        if (strstr(dev_name, requested_dev_)) {
          device = device_list[i];
          break;
        }
      }
    }

    // Open device
    context_ = ibv_open_device(device);
    CHECK_NNULL(context_, "ibv_open_device");
    dump_ibv_context(context_);
    dump_ibv_device(context_->device);

    // Allocate protection domain
    pd_orig_ = ibv_alloc_pd(context_);
    CHECK_NNULL(pd_orig_, "ibv_alloc_pd");
    dump_ibv_pd(pd_orig_);

    // Create parent domain for MLX5/IONIC
    if (vendor_ == NICVendor::MLX5) {
      createParentDomain();
    }

    // Query port
    int err = ibv_query_port(context_, port_, &portinfo_);
    CHECK_ZERO(err, "ibv_query_port");
    dump_ibv_port_attr(&portinfo_);

    // Select GID index
    selectGIDIndex();

    ibv_free_device_list(device_list);

    DEBUG_PRINT("InfiniBand device opened: %s",
                ibv_get_device_name(context_->device));
  }

  void createParentDomain() {
    DEBUG_PRINT("Creating parent domain...");

    struct ibv_parent_domain_init_attr pattr;
    memset(&pattr, 0, sizeof(pattr));

    pattr.pd = pd_orig_;
    pattr.td = nullptr;
    pattr.comp_mask = 0;

    pd_parent_ = ibv_alloc_parent_domain(context_, &pattr);
    CHECK_NNULL(pd_parent_, "ibv_alloc_parent_domain");
    dump_ibv_pd(pd_parent_);
  }

  void selectGIDIndex() {
    DEBUG_PRINT("Selecting GID index...");

    const uint8_t local_gid_prefix[2] = {0xFE, 0x80};
    int selected_gid_index = -1;
    union ibv_gid selected_gid;
    int err;

    int gid_tbl_len = portinfo_.gid_tbl_len;

    for (int i = 0; i < gid_tbl_len; i++) {
      union ibv_gid current_gid;
      err = ibv_query_gid(context_, port_, i, &current_gid);
      if (err != 0)
        continue;

      // Skip local GIDs
      if (memcmp(current_gid.raw, &local_gid_prefix, 2) == 0) {
        continue;
      }

      // Use first non-local GID
      if (selected_gid_index == -1) {
        selected_gid_index = i;
        selected_gid = current_gid;
        break;
      }
    }

    if (selected_gid_index == -1) {
      selected_gid_index = 0;
      err = ibv_query_gid(context_, port_, 0, &selected_gid);
      CHECK_ZERO(err, "ibv_query_gid");
    }

    gid_index_ = selected_gid_index;
    gid_ = selected_gid;

    DEBUG_PRINT("Selected GID index: %d", gid_index_);
  }

  void createQueues() {
    DEBUG_PRINT("Creating queues...");

    int ncqes = 64;      // Number of CQ entries
    int sq_length = 64;  // Send queue length

    // Resize vectors
    dest_info_.resize(world_size_);
    cqs_.resize(world_size_);
    qps_.resize(world_size_);

    // Create CQs and QPs
    createCQs(ncqes);
    createQPs(sq_length);

    DEBUG_PRINT("Created %d queue pairs", world_size_);
  }

  void createCQs(int ncqes) {
    DEBUG_PRINT("Creating completion queues: ncqes=%d", ncqes);

    struct ibv_cq_init_attr_ex cq_attr;
    memset(&cq_attr, 0, sizeof(cq_attr));

    cq_attr.cqe = ncqes;
    cq_attr.cq_context = nullptr;
    cq_attr.channel = nullptr;
    cq_attr.comp_vector = 0;
    cq_attr.flags = 0;

    if (pd_parent_) {
      cq_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_PD;
      cq_attr.parent_domain = pd_parent_;
    }

    for (int i = 0; i < world_size_; i++) {
      struct ibv_cq_ex* cq_ex = ibv_create_cq_ex(context_, &cq_attr);
      CHECK_NNULL(cq_ex, "ibv_create_cq_ex");

      cqs_[i] = ibv_cq_ex_to_cq(cq_ex);
      CHECK_NNULL(cqs_[i], "ibv_cq_ex_to_cq");
    }
  }

  void createQPs(int sq_length) {
    DEBUG_PRINT("Creating queue pairs: sq_length=%d", sq_length);

    struct ibv_qp_init_attr_ex attr;
    memset(&attr, 0, sizeof(attr));

    attr.cap.max_send_wr = sq_length;
    attr.cap.max_send_sge = 1;
    attr.cap.max_inline_data = 8;
    attr.sq_sig_all = 0;
    attr.qp_type = IBV_QPT_RC;
    attr.comp_mask = IBV_QP_INIT_ATTR_PD;
    attr.pd = pd_parent_ ? pd_parent_ : pd_orig_;

    for (int i = 0; i < world_size_; i++) {
      attr.send_cq = cqs_[i];
      attr.recv_cq = cqs_[i];

      struct ibv_qp* qp = ibv_create_qp_ex(context_, &attr);
      CHECK_NNULL(qp, "ibv_create_qp_ex");

      qps_[i] = std::make_unique<QueuePair>(qp, cqs_[i], i, vendor_);
    }
  }

  void exchangeQPDestInfo() {
    DEBUG_PRINT("Exchanging QP destination info...");

    // Fill local dest info
    for (int i = 0; i < world_size_; i++) {
      dest_info_[i].lid = portinfo_.lid;
      dest_info_[i].qpn = qps_[i]->getQPNum();
      dest_info_[i].psn = 0;
      dest_info_[i].gid = gid_;
    }

    // All-gather dest info
    bootstrap_->allGather(dest_info_.data(), sizeof(QPDestInfo));

    DEBUG_PRINT("QP destination info exchanged");
  }

  void modifyQPsResetToInit() {
    DEBUG_PRINT("Transitioning QPs: RESET -> INIT");

    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = port_;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
                           IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

    int attr_mask =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

    for (int i = 0; i < world_size_; i++) {
      int err = ibv_modify_qp(qps_[i]->getIBVQP(), &attr, attr_mask);
      CHECK_ZERO(err, "modify_qp (RESET->INIT)");
    }
  }

  void modifyQPsInitToRTR() {
    DEBUG_PRINT("Transitioning QPs: INIT -> RTR");

    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = portinfo_.active_mtu;
    attr.min_rnr_timer = 12;
    attr.max_dest_rd_atomic = 1;
    attr.ah_attr.port_num = port_;

    if (portinfo_.link_layer == IBV_LINK_LAYER_ETHERNET) {
      attr.ah_attr.grh.sgid_index = gid_index_;
      attr.ah_attr.is_global = 1;
      attr.ah_attr.grh.hop_limit = 1;
      attr.ah_attr.sl = 1;
      attr.ah_attr.grh.traffic_class = 0;
    }

    int attr_mask = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_RQ_PSN |
                    IBV_QP_DEST_QPN | IBV_QP_AV | IBV_QP_MAX_DEST_RD_ATOMIC |
                    IBV_QP_MIN_RNR_TIMER;

    for (int i = 0; i < world_size_; i++) {
      attr.rq_psn = dest_info_[i].psn;
      attr.dest_qp_num = dest_info_[i].qpn;

      if (portinfo_.link_layer == IBV_LINK_LAYER_ETHERNET) {
        memcpy(&attr.ah_attr.grh.dgid, &dest_info_[i].gid, 16);
      } else {
        attr.ah_attr.dlid = dest_info_[i].lid;
      }

      int err = ibv_modify_qp(qps_[i]->getIBVQP(), &attr, attr_mask);
      CHECK_ZERO(err, "modify_qp (INIT->RTR)");
    }
  }

  void modifyQPsRTRToRTS() {
    DEBUG_PRINT("Transitioning QPs: RTR -> RTS");

    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = 1;

    int attr_mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC |
                    IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY;

    for (int i = 0; i < world_size_; i++) {
      attr.sq_psn = dest_info_[i].psn;

      int err = ibv_modify_qp(qps_[i]->getIBVQP(), &attr, attr_mask);
      CHECK_ZERO(err, "modify_qp (RTR->RTS)");
    }
  }

};

}  // namespace iris_rdma
