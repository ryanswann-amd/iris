// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file iris_manager.hpp
 * @brief Complete Iris RDMA integration: Network + Queue + Proxy Thread
 *
 * Combines:
 * - NetworkBackend (InfiniBand RDMA)
 * - TritonDeviceQueue (GPU->CPU queue)
 * - Proxy Thread (processes RDMA operations from queue)
 */

#pragma once

#include <thread>
#include <atomic>
#include "network_backend.hpp"
#include "queue.hpp"

namespace iris {

/**
 * @brief Complete Iris RDMA Proxy
 *
 * Integration of network_backend + TritonDeviceQueue + Proxy Thread
 * Provides a unified interface for Triton kernels to perform RDMA operations
 */
class rdma_proxy {
 public:
  /**
   * @brief Constructor
   * @param bootstrap PyTorch bootstrap for distributed communication
   * @param heap_base Pointer to symmetric heap
   * @param heap_size Size of symmetric heap in bytes
   * @param queue_size Queue capacity (default: 512)
   */
  rdma_proxy(std::shared_ptr<rdma::torch_bootstrap> bootstrap,
             void* heap_base,
             size_t heap_size,
             int queue_size = 512)
      : heap_base_((uint64_t)heap_base),
        heap_size_(heap_size),
        running_(false) {
    
    // Step 1: Create network_backend and initialize
    backend_ = std::make_unique<network_backend>(bootstrap);
    backend_->init();
    
    // Step 2: Register symmetric heap (collective operation)
    backend_->register_memory(heap_base, heap_size);
    
    // Step 3: Create CPU-GPU queue
    queue_ = std::make_unique<rdma::queue>(queue_size);
  }

  ~rdma_proxy() {
    if (running_) {
      stop_proxy_thread();
    }
  }

  /**
   * @brief Start the proxy thread that processes RDMA operations
   */
  void start_proxy_thread() {
    if (running_) return;
    running_ = true;
    proxy_thread_ = std::thread(&rdma_proxy::proxy_loop, this);
  }

  /**
   * @brief Stop the proxy thread
   */
  void stop_proxy_thread() {
    running_ = false;
    if (proxy_thread_.joinable()) {
      proxy_thread_.join();
    }
  }

  /**
   * @brief Get the queue state pointer (for passing to Triton kernels)
   */
  rdma::queue_state_t* get_queue_ptr() {
    return queue_->get_queue_ptr();
  }

  /**
   * @brief Get heap base address
   */
  uint64_t get_heap_base() { return heap_base_; }

  /**
   * @brief Get the network_backend (for direct RDMA operations)
   */
  network_backend* get_backend() { return backend_.get(); }

  /**
   * @brief Get remote heap base for a given rank
   */
  uint64_t get_remote_heap_base(int rank) {
    return backend_->get_remote_heap_base(rank);
  }

  /**
   * @brief Get rank
   */
  int get_rank() const { return backend_->get_rank(); }

  /**
   * @brief Get world size
   */
  int get_world_size() const { return backend_->get_world_size(); }
  
  /**
   * @brief Check if queue is empty (all work processed)
   */
  bool is_queue_empty() const { return queue_->is_empty(); }

 private:
  /**
   * @brief Main proxy loop - processes RDMA operations from GPU queue
   */
  void proxy_loop() {
    rdma::work_item_t item;

    while (running_) {
      // Poll for work from GPU queue
      if (queue_->poll(item)) {
        process_work_item(item);
      }
    }
  }

  /**
   * @brief Debug helper to print work item data
   */
  void debug_print_work_item(const rdma::work_item_t& item) {
    static bool debug_enabled = (getenv("IRIS_DEBUG_DATA") != nullptr);
    if (!debug_enabled || item.header.size_bytes < 4) return;
    
    // Extract info from work item
    auto op_type = static_cast<rdma::operation_type>(item.header.op_type);
    const char* op_name = (op_type == rdma::operation_type::PUT) ? "PUT" : 
                          (op_type == rdma::operation_type::GET) ? "GET" : "OP";
    int dst_rank = item.header.rank;
    uint64_t src_ptr = item.header.src_ptr;
    uint64_t dst_ptr = item.header.dst_ptr;
    size_t size = item.header.size_bytes;
    void* data = (void*)src_ptr;
    
    static const char* dtype_env = getenv("IRIS_DTYPE");
    bool is_bf16 = (dtype_env && strcmp(dtype_env, "bfloat16") == 0);
    bool is_fp16 = (dtype_env && strcmp(dtype_env, "float16") == 0);
    bool is_fp32 = (!dtype_env || strcmp(dtype_env, "float32") == 0);
    
    if (is_bf16 || is_fp16) {
      // 2-byte types
      int elem_count = std::min((int)(size / 2), 10);
      uint16_t* data_ptr = (uint16_t*)data;
      LOG_DATA_DEBUG("[%s] rank=%d dst=%d size=%zu (bf16) src=%lx dst=%lx: first values", 
                     op_name, backend_->get_rank(), dst_rank, size, src_ptr, dst_ptr);
      for (int i = 0; i < elem_count; i++) {
        uint32_t fp32_bits = ((uint32_t)data_ptr[i]) << 16;
        float value = *reinterpret_cast<float*>(&fp32_bits);
        fprintf(stderr, "%.1f ", value);
      }
      fprintf(stderr, "\n");
    } else if (is_fp32) {
      // 4-byte types
      int elem_count = std::min((int)(size / 4), 10);
      float* float_ptr = (float*)data;
      LOG_DATA_DEBUG("[%s] rank=%d dst=%d size=%zu (fp32) src=%lx dst=%lx: first values", 
                     op_name, backend_->get_rank(), dst_rank, size, src_ptr, dst_ptr);
      for (int i = 0; i < elem_count; i++) {
        fprintf(stderr, "%.1f ", float_ptr[i]);
      }
      fprintf(stderr, "\n");
    }
  }

  /**
   * @brief Process a single work item from the queue
   */
  void process_work_item(const rdma::work_item_t& item) {
    auto op_type = static_cast<rdma::operation_type>(item.header.op_type);
    int dst_rank = item.header.rank;
    
    // Get addresses from queue metadata
    uint64_t src_ptr = item.header.src_ptr;  // Pointer/offset in registered heap
    uint64_t dst_ptr = item.header.dst_ptr;  // Remote destination
    size_t size = item.header.size_bytes;
    
    switch (op_type) {
      case rdma::operation_type::PUT: {
        // RDMA Write: Data is already in the registered heap at src_ptr
        // No memcpy needed - just RDMA directly from heap!
        void* local_addr = (void*)src_ptr;
        
        LOG_DEBUG("PUT: rank=%d src=%lx dst=%lx size=%zu", 
                  dst_rank, src_ptr, dst_ptr, size);
        
        debug_print_work_item(item);
        
        int ret = backend_->rdma_write(dst_rank, local_addr, dst_ptr, size);
        if (ret != 0) {
          LOG_ERROR("RDMA write failed: dst=%d size=%lu", dst_rank, size);
        } else {
          // Poll for completion
          int n = 0;
          for (int attempt = 0; attempt < 100; attempt++) {
            n = backend_->poll_cq(dst_rank, 1);
            if (n > 0) break;
            std::this_thread::sleep_for(std::chrono::microseconds(10));
          }
          if (n <= 0) {
            LOG_DEBUG("Warning: PUT completion not polled (may be OK if async)");
          }
        }
        
        // Signal completion
        queue_->pop();
        break;
      }
      
      case rdma::operation_type::GET: {
        // RDMA Read: Read from remote directly into registered heap at src_ptr
        // GPU will read from heap after completion
        void* local_addr = (void*)src_ptr;
        
        LOG_DEBUG("GET: rank=%d src=%lx dst=%lx size=%zu", 
                  dst_rank, dst_ptr, src_ptr, size);
        
        int ret = backend_->rdma_read(dst_rank, local_addr, dst_ptr, size);
        if (ret != 0) {
          LOG_ERROR("RDMA read failed: dst=%d size=%lu", dst_rank, size);
        } else {
          // Poll for completion
          int n = 0;
          for (int attempt = 0; attempt < 100; attempt++) {
            n = backend_->poll_cq(dst_rank, 1);
            if (n > 0) break;
            std::this_thread::sleep_for(std::chrono::microseconds(10));
          }
          if (n <= 0) {
            LOG_DEBUG("Warning: GET completion not polled (may be OK if async)");
          }
        }
        
        // Signal completion - GPU can now read from heap at src_ptr
        queue_->pop();
        break;
      }
      
      case rdma::operation_type::FLUSH: {
        // Flush all pending operations for this rank
        LOG_DEBUG("FLUSH: rank=%d", dst_rank);
        
        int total = 0;
        int n;
        do {
          n = backend_->poll_cq(dst_rank, 16);
          if (n > 0) total += n;
        } while (n > 0);
        
        queue_->pop();
        break;
      }
      
      default:
        LOG_ERROR("Unknown operation type: %d", item.header.op_type);
        queue_->pop();
    }
  }

  std::unique_ptr<network_backend> backend_;
  std::unique_ptr<rdma::queue> queue_;
  
  uint64_t heap_base_;
  size_t heap_size_;
  
  std::atomic<bool> running_;
  std::thread proxy_thread_;
};

}  // namespace iris

