// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

// vmem_allocator.hpp - Virtual memory allocator using HIP's vmem APIs

#pragma once

#define FMT_HEADER_ONLY
#include <fmt/core.h>
#include <fmt/color.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <iostream>
#include <map>
#include <memory_resource>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace iris {
namespace memory {

// Log levels
enum class LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARNING = 2,
  ERROR = 3
};

// Logging control
#ifndef IRIS_LOG_LEVEL
#define IRIS_LOG_LEVEL LogLevel::INFO  // Default log level
#endif

// Log helpers using fmt library with colors
template<typename... Args>
void log_debug(fmt::format_string<Args...> fmt_str, Args&&... args) {
  if constexpr (static_cast<int>(IRIS_LOG_LEVEL) <= static_cast<int>(LogLevel::DEBUG)) {
    fmt::print(fg(fmt::color::gray), "[IRIS DEBUG] {}\n",
               fmt::format(fmt_str, std::forward<Args>(args)...));
  }
}

template<typename... Args>
void log_info(fmt::format_string<Args...> fmt_str, Args&&... args) {
  if constexpr (static_cast<int>(IRIS_LOG_LEVEL) <= static_cast<int>(LogLevel::INFO)) {
    fmt::print(fg(fmt::color::cyan), "[IRIS INFO] {}\n",
               fmt::format(fmt_str, std::forward<Args>(args)...));
  }
}

template<typename... Args>
void log_warning(fmt::format_string<Args...> fmt_str, Args&&... args) {
  if constexpr (static_cast<int>(IRIS_LOG_LEVEL) <= static_cast<int>(LogLevel::WARNING)) {
    fmt::print(stderr, fg(fmt::color::yellow) | fmt::emphasis::bold,
               "[IRIS WARNING] {}\n",
               fmt::format(fmt_str, std::forward<Args>(args)...));
  }
}

template<typename... Args>
void log_error(fmt::format_string<Args...> fmt_str, Args&&... args) {
  if constexpr (static_cast<int>(IRIS_LOG_LEVEL) <= static_cast<int>(LogLevel::ERROR)) {
    fmt::print(stderr, fg(fmt::color::red) | fmt::emphasis::bold,
               "[IRIS ERROR] {}\n",
               fmt::format(fmt_str, std::forward<Args>(args)...));
  }
}

#define hip_try(expr)                                                          \
  do {                                                                         \
    hipError_t status = (expr);                                                \
    if (status != hipSuccess) {                                                \
      log_error("HIP error at {}:{}: {}", __FILE__, __LINE__,                 \
                hipGetErrorString(status));                                    \
      throw std::runtime_error(std::string("IRIS error at ") + __FILE__ +     \
                               ":" + std::to_string(__LINE__) + ": " +         \
                               hipGetErrorString(status));                     \
    }                                                                          \
  } while (0)


// Allocation metadata
struct AllocationInfo {
  std::size_t size;
  hipMemGenericAllocationHandle_t handle;
  bool is_imported;
};

// Free block in the VA space
struct FreeBlock {
  void* va;
  std::size_t size;
};

class SymmetricHeapResource : public std::pmr::memory_resource {
private:
  void* base_va_;
  std::size_t heap_size_;
  void* current_va_;
  std::map<void*, AllocationInfo> allocations_;
  std::vector<FreeBlock> free_list_;  // Track freed VA regions for reuse
  hipMemAllocationProp alloc_prop_;
  int device_id_;
  std::size_t granularity_;
  std::size_t cumulative_allocated_;  // Track cumulative size for hipMemSetAccess workaround

  // Helper: Align size to granularity
  std::size_t align_to_granularity(std::size_t size) const {
    return (size + granularity_ - 1) & ~(granularity_ - 1);
  }

  // Helper: Find and remove a suitable free block
  void* find_free_block(std::size_t size) {
    for (auto it = free_list_.begin(); it != free_list_.end(); ++it) {
      if (it->size >= size) {
        void* va = it->va;
        std::size_t block_size = it->size;
        free_list_.erase(it);

        // If the block is larger than needed, split it and return the remainder
        if (block_size > size) {
          void* remainder_va = static_cast<char*>(va) + size;
          std::size_t remainder_size = block_size - size;
          free_list_.push_back({remainder_va, remainder_size});
          log_debug("Split free block: using {} bytes at {}, returning {} bytes to free list",
                    size, va, remainder_size);
        }

        return va;
      }
    }
    return nullptr;  // No suitable block found
  }

  // Helper: Add block to free list
  void add_to_free_list(void* va, std::size_t size) {
    log_debug("Adding {} bytes at {} to free list", size, va);
    free_list_.push_back({va, size});
    // TODO: Could merge adjacent free blocks here for better space utilization
  }

  // Helper: Check space
  void check_space(std::size_t size) const {
    if (static_cast<char*>(current_va_) + size >
        static_cast<char*>(base_va_) + heap_size_) {
      throw std::bad_alloc();
    }
  }

  // Helper: Get allocation info
  const AllocationInfo& get_allocation_info(void* ptr) const {
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
      throw std::runtime_error("Unknown pointer");
    }
    return it->second;
  }

  // Helper: Track allocation (no bump for reused blocks)
  void track_allocation(void* va, std::size_t size,
                        hipMemGenericAllocationHandle_t handle,
                        bool is_imported) {
    allocations_[va] = {size, handle, is_imported};
  }

  // std::pmr::memory_resource interface
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    // Ensure we're operating on the intended device for this allocator.
    // In complex Python/Triton programs the current device can drift.
    hip_try(hipSetDevice(device_id_));

    std::size_t size = align_to_granularity(bytes);

    // DISABLED: Free list disabled for debugging
    // First, try to reuse a freed block
    // void* va = find_free_block(size);

    // if (va) {
    //   // Reusing a free block
    //   log_debug("Allocating {} bytes (aligned to {}) at {} [REUSED from free list]",
    //             bytes, size, va);
    // } else {
      // No suitable free block, bump allocate
      void* va = current_va_;
      check_space(size);
      log_debug("Allocating {} bytes (aligned to {}) at {} [NEW]", bytes, size, va);
      current_va_ = static_cast<char*>(va) + size;  // Bump the pointer
    // }

    // Create physical memory and map it
    hipMemGenericAllocationHandle_t handle;
    hip_try(hipMemCreate(&handle, size, &alloc_prop_, 0));
    hip_try(hipMemMap(va, size, 0, handle, 0));

    // ROCm workaround: hipMemSetAccess validates against cumulative size of ALL sub-buffers
    // Must call with base_va and cumulative size, not current va and current size
    cumulative_allocated_ += size;
    log_debug("Cumulative allocated size now: {} bytes", cumulative_allocated_);

    hipMemAccessDesc access_desc;
    access_desc.location = alloc_prop_.location;
    access_desc.flags = hipMemAccessFlagsProtReadWrite;
    hip_try(hipMemSetAccess(base_va_, cumulative_allocated_, &access_desc, 1));
    log_debug("Set access on base_va_ {} with cumulative size {}", base_va_, cumulative_allocated_);

    track_allocation(va, size, handle, false);

    log_info("Allocated {} bytes at {}", size, va);
    return va;
  }

  void do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) override {
    if (!ptr) return;
    hip_try(hipSetDevice(device_id_));

    const auto& info = get_allocation_info(ptr);
    if (info.is_imported) {
      log_error("Cannot deallocate imported pointer at {}", ptr);
      throw std::runtime_error("Cannot deallocate imported pointer, use unimport_buffer()");
    }

    log_debug("Deallocating {} bytes at {} [DEFERRED]", info.size, ptr);

    // ROCm workaround limitation: Cannot unmap individual allocations when using cumulative hipMemSetAccess
    // The cumulative approach requires all memory from base_va to cumulative_allocated to stay mapped
    // Unmapping creates "holes" which cause hipMemSetAccess to fail on unmapped regions
    //
    // Solution: Defer ALL cleanup (unmap + release) until destructor
    // This effectively makes deallocation a no-op during the allocator's lifetime
    // Physical memory and VA mappings are only freed when the allocator is destroyed
    //
    // Limitation: Memory usage is "high water mark" - never decreases until destruction

    // Do NOT unmap or release here - all cleanup happens in destructor
    // Keep allocation tracked so destructor can clean it up properly

    log_info("Deallocated {} bytes at {} [cleanup deferred until destructor due to ROCm workaround]", info.size, ptr);
  }

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

public:
  // Constructor
  SymmetricHeapResource(void* requested_base, std::size_t heap_size, int device_id = 0)
      : base_va_(nullptr),
        heap_size_(heap_size),
        current_va_(nullptr),
        device_id_(device_id),
        granularity_(4096),
        cumulative_allocated_(0) {

    log_info("Initializing symmetric heap resource");

    // Use the caller-provided device_id (do NOT override with hipGetDevice()).
    // In multi-process setups, relying on hipGetDevice() here can drift from the intended rank->device mapping.
    hip_try(hipSetDevice(device_id_));
    log_debug("Using device {}", device_id_);

    alloc_prop_ = {};
    alloc_prop_.type = hipMemAllocationTypePinned;
    alloc_prop_.location.type = hipMemLocationTypeDevice;
    alloc_prop_.location.id = device_id_;

    hip_try(hipMemGetAllocationGranularity(&granularity_, &alloc_prop_,
                                           hipMemAllocationGranularityMinimum));
    log_debug("Allocation granularity: {} bytes", granularity_);

    hip_try(hipMemAddressReserve(&base_va_, heap_size_, granularity_,
                                 requested_base, 0));
    log_info("Reserved VA range: {} - {} ({} bytes)",
        base_va_,
        static_cast<void*>(static_cast<char*>(base_va_) + heap_size_),
        heap_size_);

    current_va_ = base_va_;
  }

  ~SymmetricHeapResource() override {
    log_debug("Destructor: cleaning up {} allocations", allocations_.size());

    // Ensure all GPU operations are complete before cleanup
    (void)hipDeviceSynchronize();

    // Clean up all allocations - ignore errors in destructor
    for (auto& [va, info] : allocations_) {
      log_debug("  Unmapping {} bytes at {}", info.size, va);
      (void)hipMemUnmap(va, info.size);
      (void)hipMemRelease(info.handle);
    }
    allocations_.clear();

    log_debug("Freeing VA range {} - {} ({} bytes)",
              base_va_,
              static_cast<void*>(static_cast<char*>(base_va_) + heap_size_),
              heap_size_);
    (void)hipMemAddressFree(base_va_, heap_size_);

    log_info("Symmetric heap resource destroyed");
  }

  // Import hipMalloc buffer
  void* import_buffer(void* external_ptr, std::size_t bytes) {
    hip_try(hipSetDevice(device_id_));
    if (!external_ptr) {
      log_error("Cannot import null pointer");
      throw std::runtime_error("Cannot import null pointer");
    }

    std::size_t aligned_size = align_to_granularity(bytes);

    // DISABLED: Free list disabled for debugging
    // Try to reuse a free block first
    // void* va = find_free_block(aligned_size);

    // if (va) {
    //   log_info("Importing {} bytes from {} to {} [REUSED from free list]",
    //            bytes, external_ptr, va);
    // } else {
      // No suitable free block, bump allocate
      void* va = current_va_;
      check_space(aligned_size);
      log_info("Importing {} bytes from {} to {} [NEW]", bytes, external_ptr, va);
      current_va_ = static_cast<char*>(va) + aligned_size;  // Bump the pointer
    // }

    log_debug("Aligned the size to {} bytes", aligned_size);

    int dmabuf_fd{-1};
    hip_try(hipMemGetHandleForAddressRange((void*)&dmabuf_fd, external_ptr, aligned_size,
                                           hipMemRangeHandleTypeDmaBufFd, 0));
    log_debug("Got dmabuf_fd={} for external_ptr={}", dmabuf_fd, external_ptr);

    hipMemGenericAllocationHandle_t handle{};
    hip_try(hipMemImportFromShareableHandle(&handle, (void*)(intptr_t)dmabuf_fd,
                                            hipMemHandleTypePosixFileDescriptor));
    log_debug("Imported handle from fd, handle={}", (void*)handle);

    hip_try(hipMemMap(va, aligned_size, 0, handle, 0));
    log_debug("Mapped handle to VA {}, size {}", va, aligned_size);

    // ROCm workaround: hipMemSetAccess validates against cumulative size of ALL sub-buffers
    // Must call with base_va and cumulative size, not current va and current size
    cumulative_allocated_ += aligned_size;
    log_debug("Cumulative allocated size now: {} bytes", cumulative_allocated_);

    hipMemAccessDesc access_desc;
    access_desc.location = alloc_prop_.location;  // Use same as allocate()
    access_desc.flags = hipMemAccessFlagsProtReadWrite;
    log_debug("Setting access: device type={}, id={}, flags={}",
              (int)access_desc.location.type, access_desc.location.id, (int)access_desc.flags);
    hip_try(hipMemSetAccess(base_va_, cumulative_allocated_, &access_desc, 1));
    log_debug("Access set successfully on base_va_ {} with cumulative size {}", base_va_, cumulative_allocated_);

    track_allocation(va, aligned_size, handle, true);

    log_info("Imported {} bytes from {} to {}", aligned_size, external_ptr, va);
    return va;
  }

  // Import a DMA-BUF FD and map it at an explicit VA (e.g., base + deterministic offset).
  // This is the primitive needed for multi-process sharing where the receiver does not
  // have the sender's `external_ptr`, only a DMA-BUF handle (fd).
  void* import_dmabuf_at(void* target_va, int dmabuf_fd, std::size_t bytes) {
    hip_try(hipSetDevice(device_id_));
    if (!target_va) {
      log_error("Cannot import to null target_va");
      throw std::runtime_error("Cannot import to null target_va");
    }
    if (dmabuf_fd < 0) {
      log_error("Cannot import from invalid dmabuf_fd={}", dmabuf_fd);
      throw std::runtime_error("Invalid dmabuf_fd");
    }
    if (bytes == 0) {
      log_error("Cannot import 0 bytes at {}", target_va);
      throw std::runtime_error("Cannot import 0 bytes");
    }

    std::size_t aligned_size = align_to_granularity(bytes);

    // Bounds check: must be within reserved VA range.
    auto base = static_cast<char*>(base_va_);
    auto end = base + heap_size_;
    auto t = static_cast<char*>(target_va);
    if (t < base || (t + aligned_size) > end) {
      log_error("Target VA {} (size {}) is outside reserved range {} - {}",
                target_va, aligned_size, base_va_,
                static_cast<void*>(end));
      throw std::runtime_error("Target VA outside reserved range");
    }

    // Alignment check: target VA must be granularity-aligned for hipMemMap.
    if ((reinterpret_cast<std::uintptr_t>(target_va) % granularity_) != 0) {
      log_error("Target VA {} is not aligned to granularity {}", target_va, granularity_);
      throw std::runtime_error("Target VA is not granularity-aligned");
    }

    // If already tracked, allow idempotent import when size matches and it's imported.
    auto existing = allocations_.find(target_va);
    if (existing != allocations_.end()) {
      if (existing->second.is_imported && existing->second.size == aligned_size) {
        log_debug("Target VA {} already imported (size {}), treating as no-op", target_va, aligned_size);
        (void)::close(dmabuf_fd);
        return target_va;
      }
      log_error("Target VA {} already mapped/tracked (size {}, imported={})",
                target_va, existing->second.size, existing->second.is_imported);
      throw std::runtime_error("Target VA already mapped");
    }

    // Disallow overlaps with any tracked allocation range.
    auto new_begin = reinterpret_cast<std::uintptr_t>(target_va);
    auto new_end = new_begin + aligned_size;
    for (const auto& [va, info] : allocations_) {
      auto begin = reinterpret_cast<std::uintptr_t>(va);
      auto endp = begin + info.size;
      if (!(new_end <= begin || new_begin >= endp)) {
        log_error("Import range [{}..{}) overlaps existing allocation [{}..{})",
                  (void*)new_begin, (void*)new_end, (void*)begin, (void*)endp);
        throw std::runtime_error("Import range overlaps existing allocation");
      }
    }

    hipMemGenericAllocationHandle_t handle{};
    hip_try(hipMemImportFromShareableHandle(&handle, (void*)(intptr_t)dmabuf_fd,
                                            hipMemHandleTypePosixFileDescriptor));
    log_debug("Imported handle from dmabuf fd={}, handle={}", dmabuf_fd, (void*)handle);
    // FD is no longer needed after import (avoid leaks).
    (void)::close(dmabuf_fd);

    hip_try(hipMemMap(target_va, aligned_size, 0, handle, 0));
    log_debug("Mapped imported handle to VA {}, size {}", target_va, aligned_size);

    // ROCm workaround: Calculate the maximum extent that's been mapped
    // For import_dmabuf_at, we need to track the farthest mapped address
    auto target_end = static_cast<char*>(target_va) + aligned_size;
    auto cumulative_end = static_cast<char*>(base_va_) + cumulative_allocated_;
    if (target_end > cumulative_end) {
      cumulative_allocated_ = static_cast<char*>(target_end) - static_cast<char*>(base_va_);
      log_debug("Updated cumulative allocated to {} bytes (covers import at {})",
                cumulative_allocated_, target_va);
    }

    hipMemAccessDesc access_desc;
    access_desc.location = alloc_prop_.location;
    access_desc.flags = hipMemAccessFlagsProtReadWrite;
    hip_try(hipMemSetAccess(base_va_, cumulative_allocated_, &access_desc, 1));
    log_debug("Set access on base_va_ {} with cumulative size {}", base_va_, cumulative_allocated_);

    track_allocation(target_va, aligned_size, handle, true);
    log_info("Imported dmabuf fd={} to {}, size {}", dmabuf_fd, target_va, aligned_size);
    return target_va;
  }

  // Export a DMA-BUF FD for a (physically-backed) address range.
  int export_dmabuf(void* ptr, std::size_t bytes) {
    if (!ptr) {
      log_error("Cannot export null pointer");
      throw std::runtime_error("Cannot export null pointer");
    }
    std::size_t aligned_size = align_to_granularity(bytes);
    int dmabuf_fd{-1};
    hip_try(hipMemGetHandleForAddressRange((void*)&dmabuf_fd, ptr, aligned_size,
                                           hipMemRangeHandleTypeDmaBufFd, 0));
    log_debug("Exported dmabuf_fd={} for ptr={}, size={}", dmabuf_fd, ptr, aligned_size);
    return dmabuf_fd;
  }

  // Unimport
  void unimport_buffer(void* ptr) {
    if (!ptr) return;

    const auto& info = get_allocation_info(ptr);
    if (!info.is_imported) {
      log_error("Pointer at {} is not imported", ptr);
      throw std::runtime_error("Not an imported pointer");
    }

    log_debug("Unimporting {} bytes at {}", info.size, ptr);

    hip_try(hipMemUnmap(ptr, info.size));
    hip_try(hipMemRelease(info.handle));

    // Add the freed VA region to the free list for reuse
    add_to_free_list(ptr, info.size);

    allocations_.erase(ptr);

    log_info("Unimported {} bytes at {} [added to free list]", info.size, ptr);
  }

  // Query functions
  void* base() const { return base_va_; }
  std::size_t heap_size() const { return heap_size_; }
  std::size_t granularity() const { return granularity_; }
  std::size_t bytes_allocated() const {
    return static_cast<char*>(current_va_) - static_cast<char*>(base_va_);
  }
  std::size_t free_list_size() const { return free_list_.size(); }
  std::size_t free_list_bytes() const {
    std::size_t total = 0;
    for (const auto& block : free_list_) {
      total += block.size;
    }
    return total;
  }
  std::size_t active_allocations() const { return allocations_.size(); }

  SymmetricHeapResource(const SymmetricHeapResource&) = delete;
  SymmetricHeapResource& operator=(const SymmetricHeapResource&) = delete;
};

#undef hip_try

}  // namespace memory
}  // namespace iris
