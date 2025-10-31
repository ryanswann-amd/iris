// GPU-to-CPU Queue - C++ Host Side
// Exposes queue pointer to Python/Triton

#ifndef QUEUE_HPP_
#define QUEUE_HPP_

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>

namespace gpu_cpu_queue {

// Operation types - simplified for Iris
enum class OperationType : uint8_t {
  NOP = 0,
  PUT = 1,    // RDMA write
  GET = 2,    // RDMA read
  FLUSH = 3,  // Flush connection
};

// Work item structure - metadata only, no data storage
// Data is stored in the registered symmetric heap
struct alignas(16) WorkItemHeader {
  uint64_t dst_ptr;     // Destination pointer (where to write on remote)
  uint64_t src_ptr;     // Source pointer (offset in local registered heap)
  uint32_t size_bytes;  // Size in bytes to transfer (WRITE LAST as ready flag)
  uint16_t rank;        // Remote rank
  uint8_t op_type;      // Operation type (see OperationType enum)
  uint8_t reserved;     // Reserved for future use
};

// Note: Completion is signaled by tail pointer advancement, not a flag
struct alignas(16) WorkItem {
  WorkItemHeader header;
};

// Queue state visible to both CPU and GPU
struct QueueState {
  WorkItem* items;      // Queue buffer (pinned host memory)
  uint64_t* head;       // Head pointer (device memory, GPU writes)
  uint64_t* tail;       // Tail pointer (host memory, CPU writes, GPU reads)
  uint64_t* tailCache;  // Cached tail (device memory)
  int32_t size;         // Queue capacity
};

// CPU-side queue management
class Queue {
 public:
  explicit Queue(int size = 512) : size_(size) {
    // Allocate pinned memory for QueueState struct (GPU needs to read this)
    hipHostMalloc(&state_, sizeof(QueueState));

    // Allocate pinned memory for queue items
    hipHostMalloc(&state_->items, size * sizeof(WorkItem));
    memset(state_->items, 0, size * sizeof(WorkItem));

    // Allocate device memory for head
    hipMalloc(&state_->head, sizeof(uint64_t));
    hipMemset(state_->head, 0, sizeof(uint64_t));

    // Allocate pinned memory for tail (CPU writes, GPU reads)
    hipHostMalloc(&state_->tail, sizeof(uint64_t));
    *state_->tail = 0;

    // Allocate device memory for tail cache
    hipMalloc(&state_->tailCache, sizeof(uint64_t));
    hipMemset(state_->tailCache, 0, sizeof(uint64_t));

    state_->size = size;
  }

  ~Queue() {
    hipHostFree(state_->items);
    hipFree(state_->head);
    hipHostFree(state_->tail);
    hipFree(state_->tailCache);
    hipHostFree(state_);
  }

  // Get raw pointer to queue state for Triton
  QueueState* getQueuePtr() { return state_; }

  // Poll for new work item (non-blocking)
  bool poll(WorkItem& item) {
    uint64_t currentTail = *state_->tail;
    WorkItem* ptr = &state_->items[currentTail % size_];

    // Atomic load of size_bytes (acquire semantics) - use as ready flag
    // size_bytes == 0 means slot is empty/processed
    uint32_t size_bytes =
        reinterpret_cast<std::atomic<uint32_t>*>(&ptr->header.size_bytes)->load(std::memory_order_acquire);

    // Check if slot is ready
    if (size_bytes == 0) {
      return false;  // Queue empty
    }

    // Copy entire work item (just header now, no data array)
    memcpy(&item, ptr, sizeof(WorkItem));

    return true;
  }

  // Mark work item as processed
  void pop() {
    uint64_t currentTail = *state_->tail;

    // Clear the size_bytes to mark as processed
    state_->items[currentTail % size_].header.size_bytes = 0;

    // Advance tail with release semantics (GPU will reload this into tailCache)
    uint64_t newTail = currentTail + 1;
    reinterpret_cast<std::atomic<uint64_t>*>(state_->tail)->store(newTail, std::memory_order_release);
  }

  // Get queue statistics
  uint64_t getTail() const { return *state_->tail; }

  uint64_t getHead() const {
    uint64_t h;
    hipMemcpy(&h, state_->head, sizeof(uint64_t), hipMemcpyDeviceToHost);
    return h;
  }

  int getSize() const { return size_; }
  
  // Check if queue is empty (all work processed)
  bool isEmpty() const {
    uint64_t h;
    hipMemcpy(&h, state_->head, sizeof(uint64_t), hipMemcpyDeviceToHost);
    return h == *state_->tail;
  }

 private:
  QueueState* state_;
  int size_;
};

}  // namespace gpu_cpu_queue

#endif  // QUEUE_HPP_
