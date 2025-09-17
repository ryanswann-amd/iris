#include <ATen/ATen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>
#include <algorithm>
#include <cctype>
#include <string>
#include "torch_allocator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(torch_allocator, m) {
  m.doc() = "PyTorch Tensor Allocator with Advanced Memory Management";

  py::class_<torch_allocator::TorchAllocator, std::unique_ptr<torch_allocator::TorchAllocator>>(
      m, "TorchAllocator")
      .def(py::init<size_t, int64_t, size_t>(),
           py::arg("heap_size") = 1ULL << 30,
           py::arg("device_id") = 0,
           py::arg("alignment") = 1024,
           "Initialize allocator with heap size, device ID, and alignment")
      .def(
          "allocate_tensor",
          [](torch_allocator::TorchAllocator& self,
             const std::vector<int64_t>& shape,
             c10::ScalarType dtype,
             int64_t device_id) -> at::Tensor {
            // Calculate total size needed
            size_t total_size = 1;
            for (int64_t dim : shape) { total_size *= dim; }

            // Get element size using sizeof
            size_t element_size = at::elementSize(dtype);

            size_t total_bytes = total_size * element_size;

            // Allocate memory from our fine-grained allocator
            void* ptr = self.allocate(total_bytes);

            // Create tensor with deleter that calls deallocate
            at::TensorOptions options =
                at::TensorOptions().dtype(dtype).device(at::DeviceType::CUDA, device_id);

            // Capture allocator pointer and size
            auto deleter = [alloc_ptr = &self, bytes = total_bytes](void* p) {
              if (alloc_ptr && p) { alloc_ptr->deallocate(p, bytes); }
            };

            return at::from_blob(ptr, shape, deleter, options);
          },
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device_id") = 0,
          "Allocate memory and create PyTorch tensor");

  // Factory function
  m.def("create_allocator",
        &torch_allocator::create_allocator,
        py::arg("heap_size") = 1ULL << 30,
        py::arg("device_id") = 0,
        "Create a new TorchAllocator instance");

  // Version info
  m.attr("__version__") = "0.1.0";
}
