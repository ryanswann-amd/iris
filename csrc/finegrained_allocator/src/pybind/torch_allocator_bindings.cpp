#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/csrc/utils/pybind.h>
#include <string>
#include <algorithm>
#include <cctype>
#include <ATen/ATen.h>
#include "torch_allocator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(torch_allocator, m) {
    m.doc() = "PyTorch Tensor Allocator with Advanced Memory Management";

    py::class_<torch_allocator::TorchAllocator, std::unique_ptr<torch_allocator::TorchAllocator>>(m, "TorchAllocator")
        .def(py::init<size_t, int64_t, size_t>(), 
             py::arg("heap_size") = 1ULL << 30, 
             py::arg("device_id") = 0, 
             py::arg("alignment") = 1024,
             "Initialize allocator with heap size, device ID, and alignment")
        .def("clear_all", &torch_allocator::TorchAllocator::clear_all,
             "Clear all allocations")
        
        .def("allocate_tensor", [](torch_allocator::TorchAllocator& self, const std::vector<int64_t>& shape,
                                   const std::string& dtype_str, int64_t device_id) -> at::Tensor {
            // Map string to at::ScalarType
            auto to_dtype = [](const std::string& s) -> at::ScalarType {
                std::string v = s;
                // lowercase
                std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c){ return std::tolower(c); });
                if (v == "float32" || v == "float") return at::ScalarType::Float;
                if (v == "float64" || v == "double") return at::ScalarType::Double;
                if (v == "float16" || v == "half") return at::ScalarType::Half;
                if (v == "bfloat16" || v == "bf16") return at::ScalarType::BFloat16;
                if (v == "int64" || v == "long") return at::ScalarType::Long;
                if (v == "int32" || v == "int") return at::ScalarType::Int;
                if (v == "int16" || v == "short") return at::ScalarType::Short;
                if (v == "int8") return at::ScalarType::Char;
                if (v == "uint8" || v == "byte") return at::ScalarType::Byte;
                if (v == "bool") return at::ScalarType::Bool;
                throw std::runtime_error("Unsupported dtype string: " + s);
            };
            at::ScalarType dtype = to_dtype(dtype_str);
            
            // Calculate total size needed
            size_t total_size = 1;
            for (int64_t dim : shape) {
                total_size *= dim;
            }
            
            // Get element size using sizeof
            size_t element_size = at::elementSize(dtype);
            
            size_t total_bytes = total_size * element_size;
            
            // Allocate memory
            void* ptr = self.allocate(total_bytes);

            // Keep allocator alive and set deleter to return memory (with size)
            py::object keepalive = py::cast(&self);
            size_t bytes = total_bytes;
            auto deleter = [alloc_ptr = &self, _keep = std::move(keepalive), bytes](void* p) mutable {
                if (alloc_ptr && p) {
                    alloc_ptr->deallocate_with_size(p, bytes);
                }
            };

            // Create tensor from allocated memory with custom deleter
            at::TensorOptions options = at::TensorOptions().dtype(dtype).device(at::DeviceType::CUDA, device_id);
            return at::from_blob(ptr, shape, deleter, options);
        }, py::arg("shape"), py::arg("dtype"), py::arg("device_id") = 0,
        "Allocate memory and create PyTorch tensor");

    // Factory function
    m.def("create_allocator", &torch_allocator::create_allocator,
          py::arg("heap_size") = 1ULL << 30, py::arg("device_id") = 0,
          "Create a new TorchAllocator instance");

    // Version info
    m.attr("__version__") = "0.1.0";
}
