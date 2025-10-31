// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

/******************************************************************************
 * Python Bindings for Iris RDMA Backend using PyBind11
 *****************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "network_backend.hpp"
#include "queue_pair.hpp"
#include "torch_bootstrap.hpp"
#include "iris_manager.hpp"

namespace py = pybind11;
using namespace iris_rdma;

PYBIND11_MODULE(_iris_rdma_backend, m) {
  m.doc() =
      "Iris RDMA Backend: InfiniBand RDMA with PyTorch Integration";

  // Expose NICVendor enum
  py::enum_<NICVendor>(m, "NICVendor")
      .value("NONE", NICVendor::NONE)
      .value("IONIC", NICVendor::IONIC)
      .value("BNXT", NICVendor::BNXT)
      .value("MLX5", NICVendor::MLX5)
      .export_values();

  // Expose QPInfo struct
  py::class_<QPInfo>(m, "QPInfo")
      .def(py::init<>())
      .def_readwrite("qp_num", &QPInfo::qp_num)
      .def_readwrite("lkey", &QPInfo::lkey)
      .def_readwrite("rkey", &QPInfo::rkey)
      .def_readwrite("dst_rank", &QPInfo::dst_rank)
      .def("__repr__", [](const QPInfo& info) {
        return "<QPInfo qp_num=" + std::to_string(info.qp_num) +
               " lkey=" + std::to_string(info.lkey) +
               " rkey=" + std::to_string(info.rkey) +
               " dst_rank=" + std::to_string(info.dst_rank) + ">";
      });

  // Expose TorchBootstrap
  py::class_<TorchBootstrap, std::shared_ptr<TorchBootstrap>>(m,
                                                              "TorchBootstrap")
      .def(py::init([](py::object pg_obj) {
             // Extract c10d::ProcessGroup from Python object
             auto pg_ptr =
                 pg_obj.cast<c10::intrusive_ptr<c10d::ProcessGroup>>();
             return std::make_shared<TorchBootstrap>(pg_ptr);
           }),
           py::arg("process_group"))
      .def("get_rank", &TorchBootstrap::getRank)
      .def("get_world_size", &TorchBootstrap::getWorldSize)
      .def("barrier", &TorchBootstrap::barrier);

  // Expose QueuePair (read-only access)
  py::class_<QueuePair>(m, "QueuePair")
      .def("get_qp_num", &QueuePair::getQPNum)
      .def("get_lkey", &QueuePair::getLKey)
      .def("get_rkey", &QueuePair::getRKey)
      .def("get_dst_rank", &QueuePair::getDstRank)
      .def("get_info", &QueuePair::getInfo)
      .def("__repr__", [](const QueuePair& qp) {
        return "<QueuePair qp_num=" + std::to_string(qp.getQPNum()) +
               " dst_rank=" + std::to_string(qp.getDstRank()) + ">";
      });

  // Expose NetworkBackend
  py::class_<NetworkBackend>(m, "NetworkBackend")
      .def(py::init<std::shared_ptr<TorchBootstrap>, const char*>(),
           py::arg("bootstrap"), py::arg("device_name") = nullptr,
           "Create NetworkBackend with PyTorch bootstrap")
      .def("init", &NetworkBackend::init,
           "Initialize the network (setup QPs, transition to RTS)")
      .def(
          "register_memory",
          [](NetworkBackend& self, py::object obj, size_t size = 0) {
            void* ptr = nullptr;
            size_t actual_size = size;
            
            // Check if it's an integer (raw pointer)
            if (PyLong_Check(obj.ptr())) {
              ptr = reinterpret_cast<void*>(PyLong_AsVoidPtr(obj.ptr()));
              if (size == 0) {
                throw std::runtime_error("Size must be specified for raw pointer");
              }
              actual_size = size;
            }
            // Check if it's a PyTorch tensor
            else if (THPVariable_Check(obj.ptr())) {
              auto t = THPVariable_Unpack(obj.ptr());
              ptr = t.data_ptr();
              actual_size = t.numel() * t.element_size();
            }
            else {
              throw std::runtime_error("Expected a PyTorch tensor or integer address");
            }
            
            self.registerMemory(ptr, actual_size);
          },
          py::arg("obj"), py::arg("size") = 0,
          "Register memory for RDMA (supports CPU pinned or GPU memory via GPUDirect)")
      .def("get_qp", &NetworkBackend::getQP, py::arg("dst_rank"),
           py::return_value_policy::reference_internal,
           "Get queue pair for destination rank")
      .def("get_qp_info", &NetworkBackend::getQPInfo, py::arg("dst_rank"),
           "Get QP info for destination rank")
      .def("get_rank", &NetworkBackend::getRank, "Get rank")
      .def("get_world_size", &NetworkBackend::getWorldSize, "Get world size")
      .def("get_remote_heap_base", &NetworkBackend::getRemoteHeapBase,
           py::arg("rank"),
           "Get remote heap base address for a rank")
      .def("get_heap_base", &NetworkBackend::getHeapBase,
           "Get local heap base address")
      .def("get_heap_size", &NetworkBackend::getHeapSize,
           "Get heap size in bytes")
      .def("rdma_write",
           [](NetworkBackend& self, int dst_rank, uint64_t local_addr,
              uint64_t remote_addr, size_t size, uint64_t wr_id) {
             return self.rdmaWrite(dst_rank, reinterpret_cast<void*>(local_addr),
                                   remote_addr, size, wr_id);
           },
           py::arg("dst_rank"), py::arg("local_addr"), py::arg("remote_addr"),
           py::arg("size"), py::arg("wr_id") = 0,
           "RDMA write to remote rank (local_addr is integer address)")
      .def("rdma_read",
           [](NetworkBackend& self, int dst_rank, uint64_t local_addr,
              uint64_t remote_addr, size_t size, uint64_t wr_id) {
             return self.rdmaRead(dst_rank, reinterpret_cast<void*>(local_addr),
                                  remote_addr, size, wr_id);
           },
           py::arg("dst_rank"), py::arg("local_addr"), py::arg("remote_addr"),
           py::arg("size"), py::arg("wr_id") = 0,
           "RDMA read from remote rank (local_addr is integer address)")
      .def("poll_cq", &NetworkBackend::pollCQ,
           py::arg("dst_rank"), py::arg("max_completions") = 1,
           "Poll completion queue for RDMA operations")
      .def("__repr__", [](const NetworkBackend& backend) {
        return "<NetworkBackend rank=" + std::to_string(backend.getRank()) +
               " world_size=" + std::to_string(backend.getWorldSize()) + ">";
      });

  py::class_<iris::IrisManager>(m, "IrisManager")
      .def(py::init([](std::shared_ptr<TorchBootstrap> bootstrap, py::object heap_tensor, int queue_size) {
        // Extract heap pointer from tensor
        if (!THPVariable_Check(heap_tensor.ptr())) {
          throw std::runtime_error("heap_tensor must be a PyTorch tensor");
        }
        auto heap = THPVariable_Unpack(heap_tensor.ptr());
        void* heap_ptr = heap.data_ptr();
        size_t heap_size = heap.numel() * heap.element_size();
        
        return new iris::IrisManager(bootstrap, heap_ptr, heap_size, queue_size);
      }),
      py::arg("bootstrap"), py::arg("heap_tensor"), py::arg("queue_size") = 512,
      "Create IrisManager with NetworkBackend + Queue + Proxy Thread")
      .def("start_proxy_thread", &iris::IrisManager::startProxyThread,
           "Start proxy thread that processes RDMA operations from queue")
      .def("stop_proxy_thread", &iris::IrisManager::stopProxyThread,
           "Stop proxy thread")
      .def("get_queue_ptr", 
           [](iris::IrisManager& self) {
             return reinterpret_cast<uintptr_t>(self.getQueuePtr());
           },
           "Get queue pointer for Triton kernels")
      .def("get_heap_base", &iris::IrisManager::getHeapBase,
           "Get local heap base address")
      .def("get_remote_heap_base", &iris::IrisManager::getRemoteHeapBase,
           py::arg("rank"),
           "Get remote heap base address for a rank")
      .def("get_rank", &iris::IrisManager::getRank, "Get rank")
      .def("get_world_size", &iris::IrisManager::getWorldSize, "Get world size")
      .def("is_queue_empty", &iris::IrisManager::isQueueEmpty, 
           "Check if queue is empty (all work items processed)")
      .def("rdma_write",
           [](iris::IrisManager& self, int dst_rank, uint64_t local_addr,
              uint64_t remote_addr, size_t size, uint64_t wr_id) {
             auto backend = self.getBackend();
             return backend->rdmaWrite(dst_rank, reinterpret_cast<void*>(local_addr),
                                       remote_addr, size, wr_id);
           },
           py::arg("dst_rank"), py::arg("local_addr"), py::arg("remote_addr"),
           py::arg("size"), py::arg("wr_id") = 0,
           "RDMA write to remote rank (local_addr is integer address)")
      .def("rdma_read",
           [](iris::IrisManager& self, int dst_rank, uint64_t local_addr,
              uint64_t remote_addr, size_t size, uint64_t wr_id) {
             auto backend = self.getBackend();
             return backend->rdmaRead(dst_rank, reinterpret_cast<void*>(local_addr),
                                      remote_addr, size, wr_id);
           },
           py::arg("dst_rank"), py::arg("local_addr"), py::arg("remote_addr"),
           py::arg("size"), py::arg("wr_id") = 0,
           "RDMA read from remote rank (local_addr is integer address)")
      .def("poll_cq",
           [](iris::IrisManager& self, int dst_rank, int max_completions) {
             auto backend = self.getBackend();
             return backend->pollCQ(dst_rank, max_completions);
           },
           py::arg("dst_rank"), py::arg("max_completions") = 1,
           "Poll completion queue for RDMA operations")
      .def("__repr__", [](const iris::IrisManager& mgr) {
        return "<IrisManager rank=" + std::to_string(mgr.getRank()) +
               " world_size=" + std::to_string(mgr.getWorldSize()) + ">";
      });
}

