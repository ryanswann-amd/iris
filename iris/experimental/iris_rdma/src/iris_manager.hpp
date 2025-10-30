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
 * @brief Complete Iris RDMA Manager
 *
 * Integration of NetworkBackend + TritonDeviceQueue + Proxy Thread
 * Provides a unified interface for Triton kernels to perform RDMA operations
 */
class IrisManager {
 public:
  /**
   * @brief Constructor
   * @param bootstrap PyTorch bootstrap for distributed communication
   * @param heap_base Pointer to symmetric heap
   * @param heap_size Size of symmetric heap in bytes
   * @param queue_size Queue capacity (default: 512)
   */
  IrisManager(std::shared_ptr<iris_rdma::TorchBootstrap> bootstrap,
              void* heap_base,
              size_t heap_size,
              int queue_size = 512)
      : heap_base_((uint64_t)heap_base),
        heap_size_(heap_size),
        running_(false) {
    
    // Step 1: Create NetworkBackend and initialize
    backend_ = std::make_unique<iris_rdma::NetworkBackend>(bootstrap);
    backend_->init();
    
    // Step 2: Register symmetric heap (collective operation)
    backend_->registerMemory(heap_base, heap_size);
    
    // Step 3: Create CPU-GPU queue
    queue_ = std::make_unique<gpu_cpu_queue::Queue>(queue_size);
  }

  ~IrisManager() {
    if (running_) {
      stopProxyThread();
    }
  }

  /**
   * @brief Start the proxy thread that processes RDMA operations
   */
  void startProxyThread() {
    if (running_) return;
    running_ = true;
    proxy_thread_ = std::thread(&IrisManager::proxyLoop, this);
  }

  /**
   * @brief Stop the proxy thread
   */
  void stopProxyThread() {
    running_ = false;
    if (proxy_thread_.joinable()) {
      proxy_thread_.join();
    }
  }

  /**
   * @brief Get the queue state pointer (for passing to Triton kernels)
   */
  gpu_cpu_queue::QueueState* getQueuePtr() {
    return queue_->getQueuePtr();
  }

  /**
   * @brief Get heap base address
   */
  uint64_t getHeapBase() { return heap_base_; }

  /**
   * @brief Get the NetworkBackend (for direct RDMA operations)
   */
  iris_rdma::NetworkBackend* getBackend() { return backend_.get(); }

  /**
   * @brief Get remote heap base for a given rank
   */
  uint64_t getRemoteHeapBase(int rank) {
    return backend_->getRemoteHeapBase(rank);
  }

  /**
   * @brief Get rank
   */
  int getRank() const { return backend_->getRank(); }

  /**
   * @brief Get world size
   */
  int getWorldSize() const { return backend_->getWorldSize(); }
  
  /**
   * @brief Check if queue is empty (all work processed)
   */
  bool isQueueEmpty() const { return queue_->isEmpty(); }

 private:
  /**
   * @brief Main proxy loop - processes RDMA operations from GPU queue
   */
  void proxyLoop() {
    gpu_cpu_queue::WorkItem item;
    int checkCounter = 1000;

    while (true) {
      // Check if should stop
      if (checkCounter-- == 0) {
        checkCounter = 1000;
        if (!running_) break;
      }

      // Poll for work from GPU queue
      if (queue_->poll(item)) {
        processWorkItem(item);
      }
    }
  }

  /**
   * @brief Process a single work item from the queue
   */
  void processWorkItem(const gpu_cpu_queue::WorkItem& item) {
    auto op_type = static_cast<gpu_cpu_queue::OperationType>(item.header.op_type);
    int dst_rank = item.header.rank;
    
    // Get addresses from queue metadata
    uint64_t src_ptr = item.header.src_ptr;  // Pointer/offset in registered heap
    uint64_t dst_ptr = item.header.dst_ptr;  // Remote destination
    size_t size = item.header.size_bytes;
    
    switch (op_type) {
      case gpu_cpu_queue::OperationType::PUT: {
        // RDMA Write: Data is already in the registered heap at src_ptr
        // No memcpy needed - just RDMA directly from heap!
        void* local_addr = (void*)src_ptr;
        
        DEBUG_PRINT("[IrisManager] PUT: rank=%d src=%lx dst=%lx size=%zu", 
                    dst_rank, src_ptr, dst_ptr, size);
        
        int ret = backend_->rdmaWrite(dst_rank, local_addr, dst_ptr, size);
        if (ret != 0) {
          fprintf(stderr, "[IrisManager] RDMA write failed: dst=%d size=%lu\n", dst_rank, size);
        } else {
          // Poll for completion
          int n = 0;
          for (int attempt = 0; attempt < 100; attempt++) {
            n = backend_->pollCQ(dst_rank, 1);
            if (n > 0) break;
            std::this_thread::sleep_for(std::chrono::microseconds(10));
          }
          if (n <= 0) {
            DEBUG_PRINT("[IrisManager] Warning: PUT completion not polled (may be OK if async)");
          }
        }
        
        // Signal completion
        queue_->pop();
        break;
      }
      
      case gpu_cpu_queue::OperationType::GET: {
        // RDMA Read: Read from remote directly into registered heap at src_ptr
        // GPU will read from heap after completion
        void* local_addr = (void*)src_ptr;
        
        DEBUG_PRINT("[IrisManager] GET: rank=%d src=%lx dst=%lx size=%zu", 
                    dst_rank, dst_ptr, src_ptr, size);
        
        int ret = backend_->rdmaRead(dst_rank, local_addr, dst_ptr, size);
        if (ret != 0) {
          fprintf(stderr, "[IrisManager] RDMA read failed: dst=%d size=%lu\n", dst_rank, size);
        } else {
          // Poll for completion
          int n = 0;
          for (int attempt = 0; attempt < 100; attempt++) {
            n = backend_->pollCQ(dst_rank, 1);
            if (n > 0) break;
            std::this_thread::sleep_for(std::chrono::microseconds(10));
          }
          if (n <= 0) {
            DEBUG_PRINT("[IrisManager] Warning: GET completion not polled (may be OK if async)");
          }
        }
        
        // Signal completion - GPU can now read from heap at src_ptr
        queue_->pop();
        break;
      }
      
      case gpu_cpu_queue::OperationType::FLUSH: {
        // Flush all pending operations for this rank
        DEBUG_PRINT("[IrisManager] FLUSH: rank=%d", dst_rank);
        
        int total = 0;
        int n;
        do {
          n = backend_->pollCQ(dst_rank, 16);
          if (n > 0) total += n;
        } while (n > 0);
        
        queue_->pop();
        break;
      }
      
      default:
        fprintf(stderr, "[IrisManager] Unknown operation type: %d\n", item.header.op_type);
        queue_->pop();
    }
  }

  std::unique_ptr<iris_rdma::NetworkBackend> backend_;
  std::unique_ptr<gpu_cpu_queue::Queue> queue_;
  
  uint64_t heap_base_;
  size_t heap_size_;
  
  std::atomic<bool> running_;
  std::thread proxy_thread_;
};

}  // namespace iris

