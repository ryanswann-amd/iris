# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.21)
project(iris_vmem LANGUAGES CXX HIP)

# CPM for dependency management
include(cmake/CPM.cmake)

# Find required packages
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(hip REQUIRED)

# Use CPM to fetch pybind11
CPMAddPackage(
  NAME pybind11
  GITHUB_REPOSITORY pybind/pybind11
  GIT_TAG v2.11.1
)

# Use CPM to fetch fmt
CPMAddPackage(
  NAME fmt
  GITHUB_REPOSITORY fmtlib/fmt
  GIT_TAG 10.2.1
)

# Create Python module
pybind11_add_module(_iris_vmem csrc/src/bindings.cpp)

# Set include directories
target_include_directories(_iris_vmem PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/include
    ${Python3_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(_iris_vmem PRIVATE
    hip::host
    fmt::fmt
)

# Set C++ standard
set_target_properties(_iris_vmem PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Install target
install(TARGETS _iris_vmem
    LIBRARY DESTINATION iris
)
