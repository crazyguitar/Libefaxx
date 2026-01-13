/**
 * @file memory.h
 * @brief Libfabric symmetric memory backend
 */
#pragma once

#include <rdma/fabric/buffer.h>
#include <rdma/memory.h>

namespace fi {

/**
 * @brief Backend traits for libfabric symmetric memory
 */
struct MemoryBackend {
  using RmaIov = fi_rma_iov;
  using EFAVec = std::vector<EFA>&;
  using ChannelVec = std::vector<std::vector<Channel>>&;

  template <typename BufferType>
  static constexpr bool IsDeviceBuffer() {
    return std::is_same_v<BufferType, DeviceDMABuffer> || std::is_same_v<BufferType, DevicePinBuffer>;
  }

  template <typename BufferType, typename Mem>
  static Coro<ssize_t> Sendall(Mem& mem, int rank, size_t ch) {
    co_return co_await static_cast<BufferType&>(mem).Sendall(rank, ch);
  }

  template <typename BufferType, typename Mem>
  static Coro<ssize_t> Recvall(Mem& mem, int rank, size_t ch) {
    co_return co_await static_cast<BufferType&>(mem).Recvall(rank, ch);
  }
};

template <typename BufferType, typename QueueType = Queue<DeviceRequest>>
using SymmetricMemory = rdma::SymmetricMemory<MemoryBackend, BufferType, QueueType>;

template <typename QueueType = Queue<DeviceRequest>>
using SymmetricDMAMemoryT = SymmetricMemory<DeviceDMABuffer, QueueType>;
using SymmetricDMAMemory = SymmetricDMAMemoryT<>;

template <typename QueueType = Queue<DeviceRequest>>
using SymmetricPinMemoryT = SymmetricMemory<DevicePinBuffer, QueueType>;
using SymmetricPinMemory = SymmetricPinMemoryT<>;

template <typename QueueType = Queue<DeviceRequest>>
using SymmetricHostMemoryT = SymmetricMemory<HostBuffer, QueueType>;
using SymmetricHostMemory = SymmetricHostMemoryT<>;

}  // namespace fi
