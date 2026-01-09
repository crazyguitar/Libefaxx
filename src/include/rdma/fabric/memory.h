/**
 * @file memory.h
 * @brief Symmetric memory implementation for RDMA operations
 *
 * Implementation based on NVSHMEM/OpenSHMEM design patterns for
 * high-performance GPU-to-GPU communication over fabric networks.
 */
#pragma once

#include <bootstrap/mpi/mpi.h>
#include <rdma/fabric/buffer.h>
#include <rdma/fabric/request.h>

#include <algorithm>
#include <queue/queue.cuh>
#include <type_traits>

namespace fi {

/**
 * @brief Symmetric memory class with 2D RMA IOV structure
 *
 * Manages RMA IOVs in a 2D structure: rma_iovs_[rank][channel]
 * enabling symmetric memory access patterns like NVSHMEM/OpenSHMEM.
 *
 * @tparam BufferType The underlying buffer type (DeviceDMABuffer, DevicePinBuffer, or HostBuffer)
 * @tparam QueueType The queue type for GPU-CPU communication (Queue, PinnedQueue, or GdrQueue)
 */
template <typename BufferType, typename QueueType = Queue<DeviceRequest>>
class SymmetricMemory : public BufferType {
 public:
  /**
   * @brief Construct SymmetricMemory with buffer allocation
   * @param channels Channels for transferring data
   * @param size Buffer size in bytes
   * @param world_size Number of ranks in the world
   * @param device CUDA device ID (ignored for HostBuffer, default -1)
   * @param align Memory alignment in bytes
   */
  SymmetricMemory(std::vector<Channel>& channels, size_t size, int world_size, int device = -1, size_t align = BufferType::kAlign)
      : BufferType(channels, device, size, align), world_size_(world_size) {
    rma_iovs_.resize(world_size_);
    if constexpr (!std::is_same_v<BufferType, HostBuffer>) {
      CUDA_CHECK(cudaMallocManaged(&posted_, sizeof(uint64_t)));
      CUDA_CHECK(cudaMallocManaged(&completed_, sizeof(uint64_t)));
      *posted_ = 0;
      *completed_ = 0;

      // Initialize IPC arrays for DeviceDMABuffer
      if constexpr (std::is_same_v<BufferType, DeviceDMABuffer>) {
        CUDA_CHECK(cudaMallocManaged(&ipc_ptrs_, world_size_ * sizeof(void*)));
        for (int i = 0; i < world_size_; ++i) {
          ipc_ptrs_[i] = nullptr;
        }
      }
    }
  }

  ~SymmetricMemory() {
    if constexpr (!std::is_same_v<BufferType, HostBuffer>) {
      // Cleanup IPC handles for DeviceDMABuffer
      if constexpr (std::is_same_v<BufferType, DeviceDMABuffer>) {
        for (auto& [rank, ptr] : ipc_remote_ptrs_) {
          if (ptr != this->Data()) {
            CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
          }
        }
        if (ipc_ptrs_) cudaFree(ipc_ptrs_);
      }

      if (posted_) cudaFree(posted_);
      if (completed_) cudaFree(completed_);
    }
  }

  /**
   * @brief Get local RMA IOV for a specific channel
   * @param ch Channel index
   * @return RMA IOV for the local buffer on the specified channel
   */
  [[nodiscard]] fi_rma_iov GetLocalRmaIov(size_t ch) noexcept { return BufferType::MakeRmaIov(this->RdmaData(), this->Size(), this->mrs_[ch]); }

  /**
   * @brief Get all local RMA IOVs (one per channel)
   * @return Vector of RMA IOVs for all channels
   */
  [[nodiscard]] std::vector<fi_rma_iov> GetLocalRmaIovs() {
    std::vector<fi_rma_iov> iovs;
    iovs.reserve(this->mrs_.size());
    for (size_t ch = 0; ch < this->mrs_.size(); ++ch) iovs.push_back(GetLocalRmaIov(ch));
    return iovs;
  }

  /**
   * @brief Set remote RMA IOVs for a specific rank
   * @param rank Remote rank
   * @param iovs Vector of RMA IOVs (one per channel)
   */
  void SetRemoteRmaIovs(int rank, std::vector<fi_rma_iov> iovs) noexcept {
    ASSERT(rank >= 0 && rank < world_size_);
    rma_iovs_[rank] = std::move(iovs);
  }

  /**
   * @brief Get remote RMA IOV for a specific rank and channel
   * @param rank Remote rank
   * @param ch Channel index
   * @return Reference to the RMA IOV
   */
  [[nodiscard]] const fi_rma_iov& GetRemoteRmaIov(int rank, size_t ch) const noexcept {
    ASSERT(rank >= 0 && rank < world_size_ && ch < rma_iovs_[rank].size());
    return rma_iovs_[rank][ch];
  }

  /**
   * @brief Get all remote RMA IOVs for a specific rank
   * @param rank Remote rank
   * @return Reference to vector of RMA IOVs for all channels
   */
  [[nodiscard]] const std::vector<fi_rma_iov>& GetRemoteRmaIovs(int rank) const noexcept {
    ASSERT(rank >= 0 && rank < world_size_);
    return rma_iovs_[rank];
  }

  /**
   * @brief Write entire buffer to remote rank (single channel)
   * @param rank Target rank
   * @param imm_data Immediate data to send
   * @param ch Channel index
   * @return Coroutine returning bytes written
   */
  [[nodiscard]] Coro<ssize_t> Write(int rank, uint64_t imm_data, size_t ch) {
    const auto& iov = GetRemoteRmaIov(rank, ch);
    return BufferType::Write(iov.addr, iov.key, imm_data, ch);
  }

  /**
   * @brief Write all data to remote rank (single channel, handles large transfers)
   * @param rank Target rank
   * @param imm_data Immediate data to send
   * @param ch Channel index
   * @return Coroutine returning bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(int rank, uint64_t imm_data, size_t ch) {
    const auto& iov = GetRemoteRmaIov(rank, ch);
    return BufferType::Writeall(iov.addr, iov.key, imm_data, ch);
  }

  /**
   * @brief Encode immediate data with channel index
   * @param imm_data Base immediate data value
   * @param ch Channel index
   * @return Encoded immediate data with channel in lower 8 bits
   */
  static constexpr uint64_t EncodeImmdata(uint64_t imm_data, size_t ch) noexcept { return (imm_data << 8) | (ch & 0xFF); }

  /**
   * @brief Write entire buffer to remote rank using all channels
   *
   * Splits buffer across all available channels for maximum throughput.
   * Each channel sends channel-encoded imm_data to signal completion.
   *
   * @param rank Target rank
   * @param imm_data Immediate data (encoded with channel index per channel)
   * @return Coroutine returning total bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(int rank, uint64_t imm_data) {
    const size_t num_channels = this->channels_.size();
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
      futures.emplace_back(this->channels_[ch].Writeall(data + offset, len, mr, addr, key, EncodeImmdata(imm_data, ch)));
    }

    ssize_t total_written = 0;
    for (auto& fut : futures) {
      ssize_t written = co_await fut;
      if (written < 0) co_return written;
      total_written += written;
    }
    co_return total_written;
  }

  /**
   * @brief Wait for immediate data from all channels
   * @param imm_data Expected immediate data value (must be > 0)
   * @return Coroutine that completes when all channels receive immediate data
   */
  [[nodiscard]] Coro<> WaitallImmdata(uint64_t imm_data) {
    for (size_t ch = 0; ch < this->channels_.size(); ++ch) {
      co_await BufferType::WaitImmdata(EncodeImmdata(imm_data, ch));
    }
  }

  /** @brief Sendall with rank param for API compatibility with IB (rank ignored for fabric) */
  [[nodiscard]] Coro<ssize_t> Sendall(int /*rank*/, size_t ch) { co_return co_await BufferType::Sendall(ch); }

  /** @brief Get queue pointer for CUDA kernel access */
  [[nodiscard]] QueueType* GetQueue() noexcept { return &queue_; }

  /** @brief Get posted counter pointer for CUDA kernel access */
  [[nodiscard]] uint64_t* GetPosted() noexcept { return posted_; }

  /** @brief Get completed counter pointer for CUDA kernel access */
  [[nodiscard]] uint64_t* GetCompleted() noexcept { return completed_; }

  /** @brief Get device context for CUDA kernel access */
  [[nodiscard]] DeviceContext<QueueType> GetContext() noexcept {
    if constexpr (std::is_same_v<BufferType, DeviceDMABuffer>) {
      return {&queue_, posted_, completed_, ipc_ptrs_};
    } else {
      return {&queue_, posted_, completed_, nullptr};
    }
  }

  /** @brief Increment completed counter (called by CPU after RDMA completion) */
  void Complete() noexcept { reinterpret_cast<cuda::std::atomic<uint64_t>*>(completed_)->fetch_add(1, cuda::std::memory_order_relaxed); }

  /** @brief Open IPC handles from peer ranks on the same node */
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
  int world_size_;                                 ///< Number of ranks in the world
  std::vector<std::vector<fi_rma_iov>> rma_iovs_;  ///< 2D RMA IOVs: [rank][channel]
  QueueType queue_;                                ///< GPU request queue
  uint64_t* posted_ = nullptr;                     ///< Posted operations counter (managed memory)
  uint64_t* completed_ = nullptr;                  ///< Completed operations counter (managed memory)

  // IPC members (only used for DeviceDMABuffer)
  std::unordered_map<int, void*> ipc_remote_ptrs_;  ///< Map of rank to remote GPU pointers
  void** ipc_ptrs_ = nullptr;                       ///< Device array of IPC pointers indexed by rank
};

/** @brief Symmetric memory with DMABUF for GPU direct access */
template <typename QueueType = Queue<DeviceRequest>>
using SymmetricDMAMemoryT = SymmetricMemory<DeviceDMABuffer, QueueType>;
using SymmetricDMAMemory = SymmetricDMAMemoryT<>;

/** @brief Symmetric memory with pinned host memory */
template <typename QueueType = Queue<DeviceRequest>>
using SymmetricPinMemoryT = SymmetricMemory<DevicePinBuffer, QueueType>;
using SymmetricPinMemory = SymmetricPinMemoryT<>;

/** @brief Symmetric memory with host memory */
template <typename QueueType = Queue<DeviceRequest>>
using SymmetricHostMemoryT = SymmetricMemory<HostBuffer, QueueType>;
using SymmetricHostMemory = SymmetricHostMemoryT<>;

}  // namespace fi
