#!/bin/bash

# Build script for torch_allocator single header library

set -e

echo "Building torch_allocator..."

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "Error: setup.py not found. Please run this script from the torch_allocator directory."
    exit 1
fi

# Check if HIP is available
if ! command -v hipcc &> /dev/null; then
    echo "Error: hipcc not found. Make sure HIP is installed and in PATH."
    exit 1
fi

echo "Using hipcc compiler: $(which hipcc)"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# Build the extension
echo "Building Python extension..."
python setup.py build_ext --inplace

echo "Build complete!"
echo ""
echo "To test the allocator, run:"
echo "python examples/basic_usage.py"
