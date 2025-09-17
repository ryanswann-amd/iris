#ifndef TORCH_ALLOCATOR_H
#define TORCH_ALLOCATOR_H

#include <ATen/ATen.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "hip/hip_runtime.h"

#define hip_try(error)                                                                            \
  if (error != hipSuccess) {                                                                      \
    std::cerr << "[finegrained_allocator] Hip error: " << hipGetErrorString(error) << " at line " \
              << __LINE__ << std::endl;                                                           \
    std::exit(EXIT_FAILURE);                                                                      \
  }

namespace torch_allocator {

// Types
using Tensor        = at::Tensor;
using ScalarType    = at::ScalarType;
using TensorOptions = at::TensorOptions;

// Free list node for deallocated memory
struct FreeNode {
  void* ptr;
  size_t size;
  std::unique_ptr<FreeNode> next;

  FreeNode(void* p, size_t s) : ptr(p), size(s), next(nullptr) {}
};

// Main allocator class
class TorchAllocator {
 private:
  size_t heap_size_;
  int64_t device_id_;
  void* heap_base_;                      // Base pointer of the big chunk
  char* bump_ptr_;                       // Current bump pointer
  char* heap_end_;                       // End of heap
  std::unique_ptr<FreeNode> free_list_;  // Free list for deallocated memory
  mutable std::mutex allocator_mutex_;
  size_t alignment_;

  // Internal methods
  void* allocate_from_free_list(size_t size) {
    // Use pointer-to-unique_ptr traversal to manipulate links
    std::unique_ptr<FreeNode>* prev_link = &free_list_;
    while (prev_link->get() != nullptr) {
      FreeNode* current = prev_link->get();
      if (current->size >= size) {
        void* ptr       = current->ptr;
        size_t leftover = current->size - size;
        // Remove current node from list
        *prev_link = std::move(current->next);
        // If there's leftover space, add it back as a new node
        if (leftover > 0) { add_to_free_list(static_cast<char*>(ptr) + size, leftover); }
        return ptr;
      }
      prev_link = &((*prev_link)->next);
    }
    return nullptr;  // No suitable free block found
  }

  void add_to_free_list(void* ptr, size_t size) {
    if (ptr == nullptr || size == 0) return;
    auto new_node  = std::make_unique<FreeNode>(ptr, size);
    new_node->next = std::move(free_list_);
    free_list_     = std::move(new_node);
  }

  size_t align_size(size_t size) { return (size + alignment_ - 1) & ~(alignment_ - 1); }

 public:
  explicit TorchAllocator(size_t heap_size  = 1ULL << 30,
                          int64_t device_id = 0,
                          size_t alignment  = 1024)
      : heap_size_(heap_size), device_id_(device_id), free_list_(nullptr), alignment_(alignment) {
    // Use fine-grained allocator
    hip_try(hipExtMallocWithFlags(&heap_base_, heap_size_, hipDeviceMallocFinegrained));

    // Initialize bump allocator
    bump_ptr_ = static_cast<char*>(heap_base_);
    heap_end_ = bump_ptr_ + heap_size_;
  }

  ~TorchAllocator() {
    // For now, don't free memory to isolate if hipFree is causing the segfault
    // The memory will be freed when the HIP context is destroyed
  }

  // Simple memory allocation interface
  void* allocate(size_t size) {
    std::lock_guard<std::mutex> lock(allocator_mutex_);

    size_t aligned_size = align_size(size);
    void* ptr           = nullptr;

    // First try to allocate from free list
    ptr = allocate_from_free_list(aligned_size);

    // If not found in free list, bump allocate
    if (!ptr) {
      char* aligned_ptr =
          reinterpret_cast<char*>(align_size(reinterpret_cast<uintptr_t>(bump_ptr_)));

      if (aligned_ptr + aligned_size <= heap_end_) {
        ptr       = aligned_ptr;
        bump_ptr_ = aligned_ptr + aligned_size;
      } else {
        throw std::runtime_error("Out of memory: heap exhausted");
      }
    }

    return ptr;
  }

  // Deallocation
  void deallocate(void* ptr, size_t size) {
    if (ptr == nullptr || size == 0) return;
    std::lock_guard<std::mutex> lock(allocator_mutex_);
    add_to_free_list(ptr, size);
  }
};

// Factory function
inline std::unique_ptr<TorchAllocator> create_allocator(size_t heap_size  = 1ULL << 30,
                                                        int64_t device_id = 0) {
  return std::make_unique<TorchAllocator>(heap_size, device_id);
}

}  // namespace torch_allocator

#endif  // TORCH_ALLOCATOR_H
