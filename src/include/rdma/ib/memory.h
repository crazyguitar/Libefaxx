/**
 * @file memory.h
 * @brief Symmetric memory implementation for RDMA operations using ibverbs
 *
 * Buffer references Peer's channels[world_size][num_channels]
 * RMA IOVs: rma_iovs_[world_size][num_channels]
 */
#pragma once

#include <bootstrap/mpi/mpi.h>
#include <rdma/ib/buffer.h>
#include <rdma/ib/request.h>

#include <algorithm>
#include <iostream>
#include <queue/queue.cuh>
#include <type_traits>

namespace ib {

/**
 * @brief Symmetric memory class with 2D RMA IOV structure (ibverbs version)
 *
 * @tparam BufferType The underlying buffer type (DeviceDMABuffer or HostBuffer)
 * @tparam QueueType The queue type for GPU-CPU communication
 */
template <typename BufferType, typename QueueType = Queue<DeviceRequest>>
class SymmetricMemory : public BufferType {
 public:
  SymmetricMemory(std::vector<std::vector<Channel>>& channels, size_t size, int world_size, int device = -1, size_t align = BufferType::kAlign)
      : BufferType(channels, device, size, align), world_size_(world_size) {
    rma_iovs_.resize(world_size_);
    CUDA_CHECK(cudaMallocManaged(&posted_, sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocManaged(&completed_, sizeof(uint64_t)));
    *posted_ = 0;
    *completed_ = 0;

    if constexpr (std::is_same_v<BufferType, DeviceDMABuffer>) {
      CUDA_CHECK(cudaMallocManaged(&ipc_ptrs_, world_size_ * sizeof(void*)));
      for (int i = 0; i < world_size_; ++i) ipc_ptrs_[i] = nullptr;
    }
  }

  ~SymmetricMemory() {
    if constexpr (std::is_same_v<BufferType, DeviceDMABuffer>) {
      for (auto& [rank, ptr] : ipc_remote_ptrs_) {
        if (ptr != this->Data()) CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
      }
      if (ipc_ptrs_) cudaFree(ipc_ptrs_);
    }
    if (posted_) cudaFree(posted_);
    if (completed_) cudaFree(completed_);
  }

  [[nodiscard]] ib_rma_iov GetLocalRmaIov(size_t ch) noexcept { return BufferType::MakeRmaIov(this->RdmaData(), this->Size(), this->mrs_[ch]); }

  [[nodiscard]] std::vector<ib_rma_iov> GetLocalRmaIovs() {
    std::vector<ib_rma_iov> iovs;
    iovs.reserve(this->mrs_.size());
    for (size_t ch = 0; ch < this->mrs_.size(); ++ch) iovs.push_back(GetLocalRmaIov(ch));
    return iovs;
  }

  void SetRemoteRmaIovs(int rank, std::vector<ib_rma_iov> iovs) noexcept {
    ASSERT(rank >= 0 && rank < world_size_);
    rma_iovs_[rank] = std::move(iovs);
  }

  [[nodiscard]] const ib_rma_iov& GetRemoteRmaIov(int rank, size_t ch) const noexcept {
    ASSERT(rank >= 0 && rank < world_size_ && ch < rma_iovs_[rank].size());
    return rma_iovs_[rank][ch];
  }

  [[nodiscard]] const std::vector<ib_rma_iov>& GetRemoteRmaIovs(int rank) const noexcept {
    ASSERT(rank >= 0 && rank < world_size_);
    return rma_iovs_[rank];
  }

  [[nodiscard]] Coro<ssize_t> Write(int rank, uint64_t imm_data, size_t ch) {
    const auto& iov = GetRemoteRmaIov(rank, ch);
    return BufferType::Write(rank, iov.addr, iov.key, imm_data, ch);
  }

  [[nodiscard]] Coro<ssize_t> Writeall(int rank, uint64_t imm_data, size_t ch) {
    const auto& iov = GetRemoteRmaIov(rank, ch);
    return BufferType::Writeall(rank, iov.addr, iov.key, imm_data, ch);
  }

  static constexpr uint64_t EncodeImmdata(uint64_t imm_data, size_t ch) noexcept { return (imm_data << 8) | (ch & 0xFF); }

  [[nodiscard]] Coro<ssize_t> Writeall(int rank, uint64_t imm_data) {
    const size_t num_channels = this->mrs_.size();
    const size_t total_size = this->Size();
    const size_t chunk_size = total_size / num_channels;
    const auto& remote_rma = GetRemoteRmaIovs(rank);
    char* data = static_cast<char*>(this->RdmaData());

    std::vector<Future<Coro<ssize_t>>> futures;
    futures.reserve(num_channels);

    for (size_t ch = 0; ch < num_channels; ++ch) {
      size_t offset = ch * chunk_size;
      size_t len = (ch == num_channels - 1) ? (total_size - offset) : chunk_size;
      auto* mr = this->mrs_[ch];
      auto addr = remote_rma[ch].addr + offset;
      auto key = remote_rma[ch].key;
      futures.emplace_back(this->channels_[rank][ch].Writeall(data + offset, len, mr, addr, key, EncodeImmdata(imm_data, ch)));
    }

    ssize_t total_written = 0;
    for (auto& fut : futures) {
      ssize_t written = co_await fut;
      if (written < 0) co_return written;
      total_written += written;
    }
    co_return total_written;
  }

  [[nodiscard]] Coro<> WaitallImmdata(uint64_t imm_data) {
    for (size_t ch = 0; ch < this->mrs_.size(); ++ch) {
      co_await BufferType::WaitImmdata(EncodeImmdata(imm_data, ch));
    }
  }

  [[nodiscard]] Coro<ssize_t> Sendall(int rank, size_t ch) {
    const auto& iov = GetRemoteRmaIov(rank, ch);
    co_return co_await BufferType::Sendall(rank, iov.addr, iov.key, 1, ch);
  }

  [[nodiscard]] Coro<ssize_t> Recvall(int rank, size_t ch) { co_return co_await BufferType::Recvall(rank, 1, ch); }

  [[nodiscard]] QueueType* GetQueue() noexcept { return &queue_; }
  [[nodiscard]] uint64_t* GetPosted() noexcept { return posted_; }
  [[nodiscard]] uint64_t* GetCompleted() noexcept { return completed_; }

  [[nodiscard]] DeviceContext<QueueType> GetContext() noexcept {
    if constexpr (std::is_same_v<BufferType, DeviceDMABuffer>) {
      return {&queue_, posted_, completed_, ipc_ptrs_};
    } else {
      return {&queue_, posted_, completed_, nullptr};
    }
  }

  void Complete() noexcept { reinterpret_cast<cuda::std::atomic<uint64_t>*>(completed_)->fetch_add(1, cuda::std::memory_order_relaxed); }

  template <typename T = BufferType>
  typename std::enable_if_t<std::is_same_v<T, DeviceDMABuffer>, void>
  OpenIPCHandles(const std::vector<cudaIpcMemHandle_t>& handles, const std::vector<int>& local_world_ranks, int local_rank) {
    for (size_t i = 0; i < handles.size(); ++i) {
      int peer = local_world_ranks[i];
      if (static_cast<int>(i) == local_rank) {
        ipc_remote_ptrs_[peer] = this->Data();
        ipc_ptrs_[peer] = this->Data();
      } else {
        void* ptr = nullptr;
        CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, handles[i], cudaIpcMemLazyEnablePeerAccess));
        ipc_remote_ptrs_[peer] = ptr;
        ipc_ptrs_[peer] = ptr;
      }
    }
  }

 private:
  int world_size_;
  std::vector<std::vector<ib_rma_iov>> rma_iovs_;  // [world_size][num_channels]
  QueueType queue_;
  uint64_t* posted_ = nullptr;
  uint64_t* completed_ = nullptr;
  std::unordered_map<int, void*> ipc_remote_ptrs_;
  void** ipc_ptrs_ = nullptr;
};

template <typename QueueType = Queue<DeviceRequest>>
using SymmetricDMAMemoryT = SymmetricMemory<DeviceDMABuffer, QueueType>;
using SymmetricDMAMemory = SymmetricDMAMemoryT<>;

template <typename QueueType = Queue<DeviceRequest>>
using SymmetricHostMemoryT = SymmetricMemory<HostBuffer, QueueType>;
using SymmetricHostMemory = SymmetricHostMemoryT<>;

}  // namespace ib
