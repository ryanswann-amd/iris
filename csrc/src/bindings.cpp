// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <hip/hip_runtime.h>
#include <vmem_allocator.hpp>
#include <stdexcept>

namespace py = pybind11;

PYBIND11_MODULE(_iris_vmem, m) {
    m.doc() = "Iris Virtual Memory Allocator";

    // Expose SymmetricHeapResource - simple pointer-based interface
    py::class_<iris::memory::SymmetricHeapResource>(m, "SymmetricHeapResource")
        .def(py::init([](size_t heap_size, int device_id, py::object requested_base) {
                 void* base_ptr = nullptr;
                 if (!requested_base.is_none()) {
                     base_ptr = reinterpret_cast<void*>(requested_base.cast<uintptr_t>());
                 }
                 return new iris::memory::SymmetricHeapResource(base_ptr, heap_size, device_id);
             }),
             "Initialize symmetric heap resource",
             py::arg("heap_size"),
             py::arg("device_id") = 0,
             py::arg("requested_base") = py::none())
        .def("allocate",
             [](iris::memory::SymmetricHeapResource& self, size_t bytes) {
                 return reinterpret_cast<uintptr_t>(self.allocate(bytes, 1));
             },
             "Allocate memory and return pointer as integer",
             py::arg("bytes"))
        .def("deallocate",
             [](iris::memory::SymmetricHeapResource& self, uintptr_t ptr, size_t bytes) {
                 self.deallocate(reinterpret_cast<void*>(ptr), bytes, 1);
             },
             "Deallocate memory",
             py::arg("ptr"),
             py::arg("bytes"))
        .def("import_buffer",
             [](iris::memory::SymmetricHeapResource& self, uintptr_t external_ptr, size_t bytes) {
                 return reinterpret_cast<uintptr_t>(
                     self.import_buffer(reinterpret_cast<void*>(external_ptr), bytes)
                 );
             },
             "Import external buffer into symmetric heap",
             py::arg("external_ptr"),
             py::arg("bytes"))
        .def("unimport_buffer",
             [](iris::memory::SymmetricHeapResource& self, uintptr_t ptr) {
                 self.unimport_buffer(reinterpret_cast<void*>(ptr));
             },
             "Unimport buffer",
             py::arg("ptr"))
        .def("import_dmabuf_at",
             [](iris::memory::SymmetricHeapResource& self, uintptr_t target_ptr, int fd, size_t bytes) {
                 return reinterpret_cast<uintptr_t>(
                     self.import_dmabuf_at(reinterpret_cast<void*>(target_ptr), fd, bytes)
                 );
             },
             "Import a DMA-BUF FD and map it at an explicit VA (target_ptr).",
             py::arg("target_ptr"),
             py::arg("fd"),
             py::arg("bytes"))
        .def("export_dmabuf",
             [](iris::memory::SymmetricHeapResource& self, uintptr_t ptr, size_t bytes) {
                 return self.export_dmabuf(reinterpret_cast<void*>(ptr), bytes);
             },
             "Export DMA-BUF FD for an allocation range.",
             py::arg("ptr"),
             py::arg("bytes"))
        .def("base",
             [](const iris::memory::SymmetricHeapResource& self) {
                 return reinterpret_cast<uintptr_t>(self.base());
             },
             "Get base address of heap")
        .def("heap_size", &iris::memory::SymmetricHeapResource::heap_size,
             "Get heap size in bytes")
        .def("granularity", &iris::memory::SymmetricHeapResource::granularity,
             "Get allocation granularity")
        .def("bytes_allocated", &iris::memory::SymmetricHeapResource::bytes_allocated,
             "Get total bytes allocated (bump pointer position)")
        .def("active_allocations", &iris::memory::SymmetricHeapResource::active_allocations,
             "Get number of active allocations");
}
