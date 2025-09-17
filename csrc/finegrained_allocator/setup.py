#!/usr/bin/env python3

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

enable_debug = os.environ.get("DEBUG", "0") == "1"

cxx_args = [
    "-O3",
    "-fPIC",
    "-std=c++17",
    "-D__HIP_PLATFORM_AMD__",
    "-Wall",
    "-Wextra",
]
hipcc_args = [
    "-O3",
]
if enable_debug:
    cxx_args.append("-g")
    hipcc_args.append("-g")

setup(
    name="torch_allocator",
    version="0.1.0",
    description="HIP fine-grained bump allocator for PyTorch",
    ext_modules=[
        CUDAExtension(
            name="torch_allocator",
            sources=[
                "src/pybind/torch_allocator_bindings.cpp",
            ],
            include_dirs=[os.path.abspath("include")],
            extra_compile_args={
                "cxx": cxx_args,
                "hipcc": hipcc_args,
            },
            extra_link_args=[],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
