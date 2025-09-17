#include "torch_allocator.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>

using namespace torch_allocator;

void test_basic_allocation() {
    std::cout << "Testing basic allocation..." << std::endl;
    
    auto allocator = create_allocator(1024 * 1024, 0); // 1MB heap
    
    // Test allocation
    auto tensor = allocator->allocate_tensor({100, 100}, torch::kFloat32);
    assert(tensor.defined());
    assert(tensor.size(0) == 100);
    assert(tensor.size(1) == 100);
    assert(tensor.dtype() == torch::kFloat32);
    
    // Test deallocation
    allocator->deallocate_tensor(tensor);
    
    std::cout << "Basic allocation test passed!" << std::endl;
}

void test_multiple_allocations() {
    std::cout << "Testing multiple allocations..." << std::endl;
    
    auto allocator = create_allocator(10 * 1024 * 1024, 0); // 10MB heap
    
    std::vector<torch::Tensor> tensors;
    
    // Allocate multiple tensors
    for (int i = 0; i < 10; ++i) {
        auto tensor = allocator->allocate_tensor({1000, 1000}, torch::kFloat32);
        tensors.push_back(tensor);
    }
    
    assert(allocator->get_num_allocations() == 10);
    
    // Deallocate half
    for (int i = 0; i < 5; ++i) {
        allocator->deallocate_tensor(tensors[i]);
    }
    
    assert(allocator->get_num_allocations() == 5);
    
    // Deallocate rest
    for (int i = 5; i < 10; ++i) {
        allocator->deallocate_tensor(tensors[i]);
    }
    
    assert(allocator->get_num_allocations() == 0);
    
    std::cout << "Multiple allocations test passed!" << std::endl;
}

void test_different_dtypes() {
    std::cout << "Testing different dtypes..." << std::endl;
    
    auto allocator = create_allocator(1024 * 1024, 0);
    
    // Test different dtypes
    auto tensor_f32 = allocator->allocate_tensor({100, 100}, torch::kFloat32);
    auto tensor_f64 = allocator->allocate_tensor({100, 100}, torch::kFloat64);
    auto tensor_i32 = allocator->allocate_tensor({100, 100}, torch::kInt32);
    auto tensor_i64 = allocator->allocate_tensor({100, 100}, torch::kInt64);
    
    assert(tensor_f32.dtype() == torch::kFloat32);
    assert(tensor_f64.dtype() == torch::kFloat64);
    assert(tensor_i32.dtype() == torch::kInt32);
    assert(tensor_i64.dtype() == torch::kInt64);
    
    // Cleanup
    allocator->deallocate_tensor(tensor_f32);
    allocator->deallocate_tensor(tensor_f64);
    allocator->deallocate_tensor(tensor_i32);
    allocator->deallocate_tensor(tensor_i64);
    
    std::cout << "Different dtypes test passed!" << std::endl;
}

void test_memory_reuse() {
    std::cout << "Testing memory reuse..." << std::endl;
    
    auto allocator = create_allocator(1024 * 1024, 0);
    
    // Allocate and deallocate repeatedly
    for (int i = 0; i < 100; ++i) {
        auto tensor = allocator->allocate_tensor({100, 100}, torch::kFloat32);
        allocator->deallocate_tensor(tensor);
    }
    
    assert(allocator->get_num_allocations() == 0);
    
    std::cout << "Memory reuse test passed!" << std::endl;
}

void test_performance() {
    std::cout << "Testing performance..." << std::endl;
    
    auto allocator = create_allocator(100 * 1024 * 1024, 0); // 100MB heap
    
    const int num_iterations = 1000;
    const int tensor_size = 1000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        auto tensor = allocator->allocate_tensor({tensor_size, tensor_size}, torch::kFloat32);
        allocator->deallocate_tensor(tensor);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Performance test: " << num_iterations << " allocations/deallocations in " 
              << duration.count() << " ms" << std::endl;
    
    allocator->print_stats();
}

void test_edge_cases() {
    std::cout << "Testing edge cases..." << std::endl;
    
    auto allocator = create_allocator(1024 * 1024, 0);
    
    // Test empty tensor
    auto empty_tensor = allocator->allocate_tensor({}, torch::kFloat32);
    assert(empty_tensor.numel() == 1);
    allocator->deallocate_tensor(empty_tensor);
    
    // Test large tensor
    try {
        auto large_tensor = allocator->allocate_tensor({10000, 10000}, torch::kFloat32);
        allocator->deallocate_tensor(large_tensor);
    } catch (const std::exception& e) {
        std::cout << "Large tensor allocation failed (expected): " << e.what() << std::endl;
    }
    
    std::cout << "Edge cases test passed!" << std::endl;
}

int main() {
    try {
        test_basic_allocation();
        test_multiple_allocations();
        test_different_dtypes();
        test_memory_reuse();
        test_performance();
        test_edge_cases();
        
        std::cout << "\nAll tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
