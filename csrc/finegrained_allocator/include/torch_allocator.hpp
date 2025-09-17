#ifndef TORCH_ALLOCATOR_H
#define TORCH_ALLOCATOR_H

#include <ATen/ATen.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <cstring>
#include "hip/hip_runtime.h"

namespace torch_allocator {

// Simplify ATen types
using Tensor = at::Tensor;
using ScalarType = at::ScalarType;
using TensorOptions = at::TensorOptions;

// Free list node for deallocated memory
struct FreeNode {
    void* ptr;
    size_t size;
    std::unique_ptr<FreeNode> next;

    FreeNode(void* p, size_t s) : ptr(p), size(s), next(nullptr) {}
};

// Main allocator class - Simple bump allocator with free list
class TorchAllocator {
private:
    size_t heap_size_;
    int64_t device_id_;
    void* heap_base_;           // Base pointer of the big chunk
    char* bump_ptr_;           // Current bump pointer
    char* heap_end_;           // End of heap
    std::unique_ptr<FreeNode> free_list_; // Free list for deallocated memory
    mutable std::mutex allocator_mutex_;
    size_t alignment_;
    
    // Internal methods
    void* allocate_from_free_list(size_t size) {
        // Use pointer-to-unique_ptr traversal to manipulate links
        std::unique_ptr<FreeNode>* prev_link = &free_list_;
        while (prev_link->get() != nullptr) {
            FreeNode* current = prev_link->get();
            if (current->size >= size) {
                void* ptr = current->ptr;
                size_t leftover = current->size - size;
                // Remove current node from list
                *prev_link = std::move(current->next);
                // If there's leftover space, add it back as a new node
                if (leftover > 0) {
                    add_to_free_list(static_cast<char*>(ptr) + size, leftover);
                }
                return ptr;
            }
            prev_link = &((*prev_link)->next);
        }
        return nullptr; // No suitable free block found
    }
    
    void add_to_free_list(void* ptr, size_t size) {
        auto new_node = std::make_unique<FreeNode>(ptr, size);
        new_node->next = std::move(free_list_);
        free_list_ = std::move(new_node);
    }
    
    size_t align_size(size_t size) {
        return (size + alignment_ - 1) & ~(alignment_ - 1);
    }
    
public:
    explicit TorchAllocator(size_t heap_size = 1ULL << 30, int64_t device_id = 0, size_t alignment = 1024)
        : heap_size_(heap_size), device_id_(device_id), free_list_(nullptr), alignment_(alignment) {
        
        // Set HIP device
        int current_device;
        hipGetDevice(&current_device);
        hipSetDevice(device_id_);
        
        // Allocate big chunk at the beginning using HIP fine-grained allocator
        hipError_t hip_err = hipExtMallocWithFlags(&heap_base_, heap_size_, hipDeviceMallocFinegrained);
        if (hip_err != hipSuccess || !heap_base_) {
            hipSetDevice(current_device); // Restore original device
            throw std::runtime_error("Failed to allocate heap with HIP fine-grained allocator");
        }
        hipSetDevice(current_device); // Restore original device
        
        // Initialize bump allocator
        bump_ptr_ = static_cast<char*>(heap_base_);
        heap_end_ = bump_ptr_ + heap_size_;
    }
    
    ~TorchAllocator() {
        clear_all();
        if (heap_base_) {
            int current_device;
            hipGetDevice(&current_device);
            hipSetDevice(device_id_);
            hipFree(heap_base_);
            hipSetDevice(current_device);
        }
    }
    
    // Simple memory allocation interface
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(allocator_mutex_);
        
        size_t aligned_size = align_size(size);
        void* ptr = nullptr;
        
        // First try to allocate from free list
        ptr = allocate_from_free_list(aligned_size);
        
        // If not found in free list, bump allocate
        if (!ptr) {
            char* aligned_ptr = reinterpret_cast<char*>(align_size(reinterpret_cast<uintptr_t>(bump_ptr_)));
            
            if (aligned_ptr + aligned_size <= heap_end_) {
                ptr = aligned_ptr;
                bump_ptr_ = aligned_ptr + aligned_size;
            } else {
                throw std::runtime_error("Out of memory: heap exhausted");
            }
        }
        
        return ptr;
    }
    
    // Deallocation (size-aware)
    void deallocate_with_size(void* ptr, size_t size) {
        std::lock_guard<std::mutex> lock(allocator_mutex_);
        add_to_free_list(ptr, size);
    }

    // Disable size-less deallocation in simplified mode
    void deallocate(void* ptr) = delete;
    
    
    // Utility methods (simplified)
    size_t get_total_capacity() const { return heap_size_; }
    
    void print_stats() const = delete;
    
    // Memory management
    void clear_all() {
        std::lock_guard<std::mutex> lock(allocator_mutex_);
        // unique_ptr chain drops automatically
        free_list_.reset();
    }
    
    void reset() {
        std::lock_guard<std::mutex> lock(allocator_mutex_);
        clear_all();
        bump_ptr_ = static_cast<char*>(heap_base_);
    }
};

// Factory function
inline std::unique_ptr<TorchAllocator> create_allocator(size_t heap_size = 1ULL << 30, int64_t device_id = 0) {
    return std::make_unique<TorchAllocator>(heap_size, device_id);
}

} // namespace torch_allocator

#endif // TORCH_ALLOCATOR_H
