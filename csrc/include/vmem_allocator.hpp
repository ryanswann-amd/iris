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
  void* external_ptr;  // For imported allocations, store the original external pointer
};

class SymmetricHeapResource : public std::pmr::memory_resource {
private:
  void* base_va_;
  std::size_t heap_size_;
  std::map<void*, AllocationInfo> allocations_;
  hipMemAllocationProp alloc_prop_;
  int device_id_;
  std::size_t granularity_;

  // Single VA reservation with consecutive allocations
  void* current_va_;              // Current bump allocation pointer
  std::size_t cumulative_allocated_;  // Cumulative size for hipMemSetAccess workaround

  // Helper: Align size to granularity
  std::size_t align_to_granularity(std::size_t size) const {
    return (size + granularity_ - 1) & ~(granularity_ - 1);
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
                        bool is_imported, void* external_ptr = nullptr) {
    allocations_[va] = {size, handle, is_imported, external_ptr};
  }

  // std::pmr::memory_resource interface
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    hip_try(hipSetDevice(device_id_));

    std::size_t size = align_to_granularity(bytes);

    // For empty allocations (0 bytes), allocate minimum granularity
    // hipMemCreate fails with 0 bytes, so we always allocate at least granularity_
    if (size == 0) {
      size = granularity_;
    }

    // Bump allocate consecutively
    void* va = current_va_;
    if (static_cast<char*>(va) + size > static_cast<char*>(base_va_) + heap_size_) {
      log_error("Heap out of space");
      throw std::bad_alloc();
    }

    log_debug("Allocating {} bytes (aligned to {}) at VA {}", bytes, size, va);
    current_va_ = static_cast<char*>(va) + size;  // Bump for next allocation

    // Create physical memory and map it
    hipMemGenericAllocationHandle_t handle;
    hip_try(hipMemCreate(&handle, size, &alloc_prop_, 0));
    hip_try(hipMemMap(va, size, 0, handle, 0));

    // Update cumulative size
    cumulative_allocated_ += size;
    log_debug("Cumulative allocated: {} bytes", cumulative_allocated_);

    // Always call hipMemSetAccess from base_va with cumulative size
    hipMemAccessDesc access_desc;
    access_desc.location = alloc_prop_.location;
    access_desc.flags = hipMemAccessFlagsProtReadWrite;
    hip_try(hipMemSetAccess(base_va_, cumulative_allocated_, &access_desc, 1));
    log_debug("Set access on base_va {} with cumulative size {}", base_va_, cumulative_allocated_);

    track_allocation(va, size, handle, false);

    log_info("Allocated {} bytes at {}", size, va);
    return va;
  }

  void do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) override {
    if (!ptr) return;

    const auto& info = get_allocation_info(ptr);
    if (info.is_imported) {
      // Imported buffers: just do nothing
      log_info("Deallocated imported buffer at {} [no-op]", ptr);
      return;
    }

    // Physical allocations: deferred until destructor
    log_info("Deallocated {} bytes at {} [cleanup deferred]", info.size, ptr);
  }

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

public:
  // Constructor
  SymmetricHeapResource(void* requested_base, std::size_t heap_size, int device_id = 0)
      : base_va_(nullptr),
        heap_size_(heap_size),
        device_id_(device_id),
        granularity_(4096),
        cumulative_allocated_(0) {

    log_info("Initializing symmetric heap with single VA reservation");

    hip_try(hipSetDevice(device_id_));
    log_debug("Using device {}", device_id_);

    alloc_prop_ = {};
    alloc_prop_.type = hipMemAllocationTypePinned;
    alloc_prop_.location.type = hipMemLocationTypeDevice;
    alloc_prop_.location.id = device_id_;

    hip_try(hipMemGetAllocationGranularity(&granularity_, &alloc_prop_,
                                           hipMemAllocationGranularityMinimum));
    log_debug("Allocation granularity: {} bytes", granularity_);

    // Reserve a SINGLE VA range for all allocations (physical + imports)
    hip_try(hipMemAddressReserve(&base_va_, heap_size_, granularity_,
                                 requested_base, 0));
    log_info("Reserved VA range: {} - {} ({} MB)",
        base_va_,
        static_cast<void*>(static_cast<char*>(base_va_) + heap_size_),
        heap_size_ / (1024*1024));

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

    log_debug("Freeing VA range {} ({} bytes)", base_va_, heap_size_);
    (void)hipMemAddressFree(base_va_, heap_size_);

    log_info("Symmetric heap resource destroyed");
  }

  // Import external buffer (e.g., from PyTorch/hipMalloc)
  void* import_buffer(void* external_ptr, std::size_t bytes) {
    hip_try(hipSetDevice(device_id_));
    if (!external_ptr) {
      log_error("Cannot import null pointer");
      throw std::runtime_error("Cannot import null pointer");
    }

    // Query the base allocation to handle offset pointers (PyTorch caching allocator)
    void* alloc_base = nullptr;
    std::size_t alloc_size = 0;
    hip_try(hipMemGetAddressRange(&alloc_base, &alloc_size, external_ptr));
    std::size_t offset_in_alloc = static_cast<char*>(external_ptr) - static_cast<char*>(alloc_base);
    log_debug("External ptr {} is at offset {} in allocation base {} (size {})",
              external_ptr, offset_in_alloc, alloc_base, alloc_size);

    std::size_t aligned_size = align_to_granularity(alloc_size);

    // Bump allocate consecutively
    void* va_base = current_va_;
    if (static_cast<char*>(va_base) + aligned_size > static_cast<char*>(base_va_) + heap_size_) {
      log_error("Heap out of space");
      throw std::bad_alloc();
    }

    log_info("Importing {} bytes (full alloc) from base {} to VA {}",
             alloc_size, alloc_base, va_base);
    current_va_ = static_cast<char*>(va_base) + aligned_size;  // Bump for next allocation

    // Export the BASE allocation to DMA-BUF
    int dmabuf_fd = -1;
    hip_try(hipMemGetHandleForAddressRange((void*)&dmabuf_fd, alloc_base, aligned_size,
                                           hipMemRangeHandleTypeDmaBufFd, 0));
    log_debug("Got dmabuf_fd={} for alloc_base={}", dmabuf_fd, alloc_base);

    // Import DMA-BUF handle
    hipMemGenericAllocationHandle_t handle{};
    hip_try(hipMemImportFromShareableHandle(&handle, (void*)(intptr_t)dmabuf_fd,
                                            hipMemHandleTypePosixFileDescriptor));
    log_debug("Imported handle from fd, handle={}", (void*)handle);

    ::close(dmabuf_fd);

    // Map into our VA range (consecutive!)
    hip_try(hipMemMap(va_base, aligned_size, 0, handle, 0));
    log_debug("Mapped handle to VA {}, size {}", va_base, aligned_size);

    // Update cumulative size
    cumulative_allocated_ += aligned_size;
    log_debug("Cumulative allocated: {} bytes", cumulative_allocated_);

    // CRITICAL: Always call hipMemSetAccess from base_va with cumulative size
    hipMemAccessDesc access_desc;
    access_desc.location = alloc_prop_.location;
    access_desc.flags = hipMemAccessFlagsProtReadWrite;
    hip_try(hipMemSetAccess(base_va_, cumulative_allocated_, &access_desc, 1));
    log_debug("Set access on base_va {} with cumulative size {}", base_va_, cumulative_allocated_);

    // Store the external allocation base pointer so we can re-export it later
    track_allocation(va_base, aligned_size, handle, true, alloc_base);

    // Return remapped VA with same offset as original pointer (for symmetric heap)
    void* va_with_offset = static_cast<char*>(va_base) + offset_in_alloc;
    log_info("Imported {} bytes from {} to {} (base {} + offset {})",
             alloc_size, external_ptr, va_with_offset, va_base, offset_in_alloc);
    return va_with_offset;
  }

  // Import a DMA-BUF FD and map it at an explicit VA.
  //
  // For symmetric heap imports (rma=false), target_va must be within our reserved range.
  // For RMA imports (rma=true), target_va can be at arbitrary peer VAs outside our range.
  void* import_dmabuf_at(void* target_va, int dmabuf_fd, std::size_t bytes, bool rma = false) {
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

    // Bounds check: only for non-RMA imports within our reserved range
    if (!rma) {
      auto base = static_cast<char*>(base_va_);
      auto end = base + heap_size_;
      auto t = static_cast<char*>(target_va);
      if (t < base || (t + aligned_size) > end) {
        log_error("Target VA {} (size {}) is outside reserved range {} - {}",
                  target_va, aligned_size, base_va_,
                  static_cast<void*>(end));
        throw std::runtime_error("Target VA outside reserved range");
      }
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

    // Disallow overlaps with any tracked allocation range (only for non-RMA imports)
    if (!rma) {
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
    }

    hipMemGenericAllocationHandle_t handle{};
    hip_try(hipMemImportFromShareableHandle(&handle, (void*)(intptr_t)dmabuf_fd,
                                            hipMemHandleTypePosixFileDescriptor));
    log_debug("Imported handle from dmabuf fd={}, handle={}", dmabuf_fd, (void*)handle);
    // FD is no longer needed after import (avoid leaks).
    (void)::close(dmabuf_fd);

    hip_try(hipMemMap(target_va, aligned_size, 0, handle, 0));
    log_debug("Mapped imported handle to VA {}, size {}", target_va, aligned_size);

    // Set access permissions
    hipMemAccessDesc access_desc;
    access_desc.location = alloc_prop_.location;
    access_desc.flags = hipMemAccessFlagsProtReadWrite;

    if (!rma) {
      // For non-RMA: Calculate cumulative size and set access from base_va
      auto target_end = static_cast<char*>(target_va) + aligned_size;
      auto cumulative_end = static_cast<char*>(base_va_) + cumulative_allocated_;
      if (target_end > cumulative_end) {
        cumulative_allocated_ = static_cast<char*>(target_end) - static_cast<char*>(base_va_);
        log_debug("Updated cumulative allocated to {} bytes (covers import at {})",
                  cumulative_allocated_, target_va);
      }
      hip_try(hipMemSetAccess(base_va_, cumulative_allocated_, &access_desc, 1));
      log_debug("Set access on base_va {} with cumulative size {}", base_va_, cumulative_allocated_);
    } else {
      // For RMA: hipMemMap is sufficient; access permissions come from the exporting process
      // We don't call hipMemSetAccess on unmapped VA ranges
      log_debug("RMA import complete at {} (skipping hipMemSetAccess)", target_va);
    }

    track_allocation(target_va, aligned_size, handle, true);
    log_info("Imported dmabuf to {} (rma={}), size {}", target_va, rma, aligned_size);
    return target_va;
  }

  // Export a DMA-BUF FD for a (physically-backed) address range.
  int export_dmabuf(void* ptr, std::size_t bytes) {
    if (!ptr) {
      log_error("Cannot export null pointer");
      throw std::runtime_error("Cannot export null pointer");
    }
    std::size_t aligned_size = align_to_granularity(bytes);

    // Check if this is an imported allocation
    const auto& info = get_allocation_info(ptr);
    void* export_ptr = ptr;

    if (info.is_imported && info.external_ptr) {
      // For imported allocations, export the original external pointer
      // This is the actual PyTorch/hipMalloc allocation that we can export
      export_ptr = info.external_ptr;
      log_debug("Exporting imported allocation: using external_ptr={} instead of VA={}",
                export_ptr, ptr);
    }

    int dmabuf_fd{-1};
    hip_try(hipMemGetHandleForAddressRange((void*)&dmabuf_fd, export_ptr, aligned_size,
                                           hipMemRangeHandleTypeDmaBufFd, 0));
    log_debug("Exported dmabuf_fd={} for ptr={}, size={}", dmabuf_fd, export_ptr, aligned_size);
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

    // Do nothing - cleanup deferred to destructor
    log_info("Unimported {} bytes at {} [cleanup deferred]", info.size, ptr);
  }

  // Query functions
  void* base() const { return base_va_; }
  std::size_t heap_size() const { return heap_size_; }
  std::size_t granularity() const { return granularity_; }
  std::size_t bytes_allocated() const {
    return cumulative_allocated_;
  }
  std::size_t active_allocations() const { return allocations_.size(); }

  SymmetricHeapResource(const SymmetricHeapResource&) = delete;
  SymmetricHeapResource& operator=(const SymmetricHeapResource&) = delete;
};

#undef hip_try

}  // namespace memory
}  // namespace iris
