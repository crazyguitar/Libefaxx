/**
 * @file symmetric.h
 * @brief Shared SymmetricMemory components for IB and Fabric
 *
 * Contains common functionality: IPC handles, queue management, counters
 */
#pragma once

#include <cuda_runtime.h>
#include <rdma/request.h>

#include <device/common.cuh>
#include <queue/queue.cuh>
#include <unordered_map>
#include <vector>

namespace rdma {

/**
 * @brief Base class for symmetric memory IPC and queue management
 *
 * Handles CUDA IPC, posted/completed counters, and GPU request queue.
 * Provider-specific classes (IB/Fabric) inherit from this via CRTP.
 *
 * @tparam QueueType The queue type for GPU-CPU communication
 * @tparam Derived The derived class (CRTP pattern)
 */
template <typename QueueType, typename Derived>
class SymmetricMemoryBase {
 public:
  SymmetricMemoryBase(int world_size, bool is_device_buffer) : world_size_(world_size), is_device_buffer_(is_device_buffer) {
    if (is_device_buffer_) {
      CUDA_CHECK(cudaMallocManaged(&posted_, sizeof(uint64_t)));
      CUDA_CHECK(cudaMallocManaged(&completed_, sizeof(uint64_t)));
      *posted_ = 0;
      *completed_ = 0;
      CUDA_CHECK(cudaMallocManaged(&ipc_ptrs_, world_size_ * sizeof(void*)));
      for (int i = 0; i < world_size_; ++i) ipc_ptrs_[i] = nullptr;
    }
  }

  ~SymmetricMemoryBase() {
    if (is_device_buffer_) {
      for (auto& [rank, ptr] : ipc_remote_ptrs_) {
        if (ptr != static_cast<Derived*>(this)->Data()) {
          CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
        }
      }
      if (ipc_ptrs_) cudaFree(ipc_ptrs_);
      if (posted_) cudaFree(posted_);
      if (completed_) cudaFree(completed_);
    }
  }

  [[nodiscard]] QueueType* GetQueue() noexcept { return &queue_; }
  [[nodiscard]] uint64_t* GetPosted() noexcept { return posted_; }
  [[nodiscard]] uint64_t* GetCompleted() noexcept { return completed_; }

  [[nodiscard]] DeviceContext<QueueType> GetContext() noexcept { return {&queue_, posted_, completed_, is_device_buffer_ ? ipc_ptrs_ : nullptr}; }

  void Complete() noexcept { reinterpret_cast<cuda::std::atomic<uint64_t>*>(completed_)->fetch_add(1, cuda::std::memory_order_relaxed); }

  void OpenIPCHandles(const std::vector<cudaIpcMemHandle_t>& handles, const std::vector<int>& local_world_ranks, int local_rank) {
    if (!is_device_buffer_) return;
    for (size_t i = 0; i < handles.size(); ++i) {
      int peer = local_world_ranks[i];
      if (static_cast<int>(i) == local_rank) {
        ipc_remote_ptrs_[peer] = static_cast<Derived*>(this)->Data();
        ipc_ptrs_[peer] = static_cast<Derived*>(this)->Data();
      } else {
        void* ptr = nullptr;
        CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, handles[i], cudaIpcMemLazyEnablePeerAccess));
        ipc_remote_ptrs_[peer] = ptr;
        ipc_ptrs_[peer] = ptr;
      }
    }
  }

  static constexpr uint64_t EncodeImmdata(uint64_t imm_data, size_t ch) noexcept { return (imm_data << 8) | (ch & 0xFF); }

 protected:
  int world_size_;
  bool is_device_buffer_;
  QueueType queue_;
  uint64_t* posted_ = nullptr;
  uint64_t* completed_ = nullptr;
  std::unordered_map<int, void*> ipc_remote_ptrs_;
  void** ipc_ptrs_ = nullptr;
};

}  // namespace rdma
