/**
 * @file memory.h
 * @brief Symmetric memory implementation for RDMA operations using libfabric
 */
#pragma once

#include <bootstrap/mpi/mpi.h>
#include <rdma/fabric/buffer.h>
#include <rdma/request.h>
#include <rdma/symmetric.h>

#include <algorithm>
#include <type_traits>

namespace fi {

/**
 * @brief Symmetric memory class with 2D RMA IOV structure (libfabric version)
 *
 * @tparam BufferType The underlying buffer type (DeviceDMABuffer, DevicePinBuffer, or HostBuffer)
 * @tparam QueueType The queue type for GPU-CPU communication
 */
template <typename BufferType, typename QueueType = Queue<DeviceRequest>>
class SymmetricMemory : public BufferType, public rdma::SymmetricMemoryBase<QueueType, SymmetricMemory<BufferType, QueueType>> {
  using Base = rdma::SymmetricMemoryBase<QueueType, SymmetricMemory<BufferType, QueueType>>;

  static constexpr bool IsDeviceBuffer() { return std::is_same_v<BufferType, DeviceDMABuffer> || std::is_same_v<BufferType, DevicePinBuffer>; }

 public:
  SymmetricMemory(
      std::vector<EFA>& efas,
      std::vector<std::vector<Channel>>& channels,
      int device,
      size_t size,
      int world_size,
      size_t align = BufferType::kAlign
  )
      : BufferType(efas, channels, device, size, align), Base(world_size, IsDeviceBuffer()) {
    rma_iovs_.resize(world_size);
  }

  [[nodiscard]] fi_rma_iov GetLocalRmaIov(size_t ch) noexcept { return BufferType::MakeRmaIov(this->RdmaData(), this->Size(), this->mrs_[ch]); }

  [[nodiscard]] std::vector<fi_rma_iov> GetLocalRmaIovs() {
    std::vector<fi_rma_iov> iovs;
    iovs.reserve(this->mrs_.size());
    for (size_t ch = 0; ch < this->mrs_.size(); ++ch) iovs.push_back(GetLocalRmaIov(ch));
    return iovs;
  }

  void SetRemoteRmaIovs(int rank, std::vector<fi_rma_iov> iovs) noexcept {
    ASSERT(rank >= 0 && rank < this->world_size_);
    rma_iovs_[rank] = std::move(iovs);
  }

  [[nodiscard]] const fi_rma_iov& GetRemoteRmaIov(int rank, size_t ch) const noexcept {
    ASSERT(rank >= 0 && rank < this->world_size_ && ch < rma_iovs_[rank].size());
    return rma_iovs_[rank][ch];
  }

  [[nodiscard]] const std::vector<fi_rma_iov>& GetRemoteRmaIovs(int rank) const noexcept {
    ASSERT(rank >= 0 && rank < this->world_size_);
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

  [[nodiscard]] Coro<ssize_t> Writeall(int rank, uint64_t imm_data) {
    const size_t num_channels = this->channels_[rank].size();
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
      futures.emplace_back(this->channels_[rank][ch].Writeall(data + offset, len, mr, addr, key, Base::EncodeImmdata(imm_data, ch)));
    }

    // Must wait for ALL futures even on error to prevent use-after-free.
    ssize_t total_written = 0;
    ssize_t first_error = 0;
    for (auto& fut : futures) {
      ssize_t written = co_await fut;
      if (written < 0 && first_error == 0)
        first_error = written;
      else if (written >= 0)
        total_written += written;
    }
    co_return first_error < 0 ? first_error : total_written;
  }

  [[nodiscard]] Coro<> WaitallImmdata(uint64_t imm_data) {
    // Get num_channels from first non-empty row
    size_t num_channels = 0;
    for (auto& row : this->channels_) {
      if (!row.empty()) {
        num_channels = row.size();
        break;
      }
    }
    for (size_t ch = 0; ch < num_channels; ++ch) {
      co_await BufferType::WaitImmdata(Base::EncodeImmdata(imm_data, ch));
    }
  }

  [[nodiscard]] Coro<ssize_t> Sendall(int rank, size_t ch) { co_return co_await BufferType::Sendall(rank, ch); }
  [[nodiscard]] Coro<ssize_t> Recvall(int rank, size_t ch) { co_return co_await BufferType::Recvall(rank, ch); }

 private:
  std::vector<std::vector<fi_rma_iov>> rma_iovs_;  // [world_size][num_channels]
};

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
