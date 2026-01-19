// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <Python.h>
#include <hip/hip_runtime.h>
#include <memory_resource>
#include <stdexcept>
#include <string>

namespace iris {
namespace gpu_array {

// Structure to hold GPU Array Interface data (__cuda_array_interface__ compatible)
struct GpuArrayInterface {
    void* data;           // Device pointer
    int64_t* shape;       // Shape array
    int64_t* strides;     // Strides array (in bytes)
    int ndim;            // Number of dimensions
    std::string typestr; // Data type string (e.g., "<f4" for float32)
    int version;         // Interface version (3)
    
    // Cleanup
    ~GpuArrayInterface() {
        if (shape) delete[] shape;
        if (strides) delete[] strides;
    }
};

// Context for managing lifetime of GPU array interface objects
struct GpuArrayContext {
    std::pmr::memory_resource* allocator;  // Pointer to allocator
    void* data_ptr;                         // Device pointer
    size_t size;                           // Allocation size in bytes
    GpuArrayInterface* interface;          // Interface struct
    
    ~GpuArrayContext() {
        if (interface) delete interface;
    }
};

// Convert dtype string to GPU Array Interface typestr
// Format: <endianness><type><size>
// endianness: < (little), > (big), | (not applicable)
// type: i (int), u (uint), f (float), c (complex)
// size: bytes per element
inline std::string dtype_to_typestr(const std::string& dtype_str) {
    if (dtype_str == "int32") return "<i4";
    if (dtype_str == "int64") return "<i8";
    if (dtype_str == "float32") return "<f4";
    if (dtype_str == "float64") return "<f8";
    if (dtype_str == "uint32") return "<u4";
    if (dtype_str == "uint64") return "<u8";
    throw std::runtime_error("Unsupported dtype: " + dtype_str);
}

// Get element size from dtype string
inline size_t get_element_size(const std::string& dtype_str) {
    if (dtype_str == "int32" || dtype_str == "uint32" || dtype_str == "float32") return 4;
    if (dtype_str == "int64" || dtype_str == "uint64" || dtype_str == "float64") return 8;
    throw std::runtime_error("Unknown dtype: " + dtype_str);
}

// Compute row-major strides (in bytes)
inline void compute_strides(int64_t* strides, const int64_t* shape, int ndim, size_t element_size) {
    int64_t stride = element_size;
    for (int i = ndim - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

// Create Python dict for __cuda_array_interface__
inline PyObject* create_gpu_array_interface_dict(
    void* data_ptr,
    const int64_t* shape,
    int ndim,
    const std::string& typestr,
    bool readonly = false
) {
    PyObject* dict = PyDict_New();
    if (!dict) {
        throw std::runtime_error("Failed to create dict");
    }

    // shape: tuple of ints
    PyObject* shape_tuple = PyTuple_New(ndim);
    for (int i = 0; i < ndim; ++i) {
        PyTuple_SetItem(shape_tuple, i, PyLong_FromLongLong(shape[i]));
    }
    PyDict_SetItemString(dict, "shape", shape_tuple);
    Py_DECREF(shape_tuple);

    // typestr: string
    PyObject* typestr_obj = PyUnicode_FromString(typestr.c_str());
    PyDict_SetItemString(dict, "typestr", typestr_obj);
    Py_DECREF(typestr_obj);

    // data: tuple (pointer, readonly)
    PyObject* data_tuple = PyTuple_New(2);
    PyTuple_SetItem(data_tuple, 0, PyLong_FromVoidPtr(data_ptr));
    PyTuple_SetItem(data_tuple, 1, PyBool_FromLong(readonly ? 1 : 0));
    PyDict_SetItemString(dict, "data", data_tuple);
    Py_DECREF(data_tuple);

    // version: int (should be 3)
    PyObject* version_obj = PyLong_FromLong(3);
    PyDict_SetItemString(dict, "version", version_obj);
    Py_DECREF(version_obj);

    // strides: None for C-contiguous (optional, can compute if needed)
    Py_INCREF(Py_None);
    PyDict_SetItemString(dict, "strides", Py_None);

    // descr: None for simple types (optional)
    Py_INCREF(Py_None);
    PyDict_SetItemString(dict, "descr", Py_None);

    return dict;
}

}  // namespace gpu_array
}  // namespace iris
