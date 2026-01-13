/**
 * @file memory.h
 * @brief Unified symmetric memory template for RDMA backends
 */
#pragma once

#include <rdma/request.h>
#include <rdma/symmetric.h>

#include <algorithm>
#include <type_traits>
#include <vector>

namespace rdma {

/**
 * @brief Generic symmetric memory class for RDMA operations
 *
 * @tparam Backend Backend traits (fi::MemoryBackend or ib::MemoryBackend)
 * @tparam BufferType The underlying buffer type
 * @tparam QueueType The queue type for GPU-CPU communication
 */
template <typename Backend, typename BufferType, typename QueueType = Queue<DeviceRequest>>
class SymmetricMemory : public BufferType, public SymmetricMemoryBase<QueueType, SymmetricMemory<Backend, BufferType, QueueType>> {
  using Base = SymmetricMemoryBase<QueueType, SymmetricMemory<Backend, BufferType, QueueType>>;
  using RmaIov = typename Backend::RmaIov;

 public:
  /**
   * @brief Construct SymmetricMemory
   * @param efas EFA endpoints for memory registration
   * @param channels 2D channel array [world_size][num_channels]
   * @param device CUDA device ID
   * @param size Buffer size in bytes
   * @param world_size Number of ranks
   * @param align Memory alignment
   */
  SymmetricMemory(
      typename Backend::EFAVec& efas,
      typename Backend::ChannelVec& channels,
      int device,
      size_t size,
      int world_size,
      size_t align = BufferType::kAlign
  )
      : BufferType(efas, channels, device, size, align), Base(world_size, Backend::template IsDeviceBuffer<BufferType>()) {
    rma_iovs_.resize(world_size);
  }

  /**
   * @brief Get local RMA IOV for a specific channel
   * @param ch Channel index
   * @return RMA IOV with address, size, and remote key
   */
  [[nodiscard]] RmaIov GetLocalRmaIov(size_t ch) noexcept { return BufferType::MakeRmaIov(this->RdmaData(), this->Size(), this->mrs_[ch]); }

  /**
   * @brief Get local RMA IOVs for all channels
   * @return Vector of RMA IOVs
   */
  [[nodiscard]] std::vector<RmaIov> GetLocalRmaIovs() {
    std::vector<RmaIov> iovs;
    iovs.reserve(this->mrs_.size());
    for (size_t ch = 0; ch < this->mrs_.size(); ++ch) iovs.push_back(GetLocalRmaIov(ch));
    return iovs;
  }

  /**
   * @brief Set remote RMA IOVs for a specific rank
   * @param rank Remote rank
   * @param iovs Vector of RMA IOVs from remote rank
   */
  void SetRemoteRmaIovs(int rank, std::vector<RmaIov> iovs) noexcept {
    ASSERT(rank >= 0 && rank < this->world_size_);
    rma_iovs_[rank] = std::move(iovs);
  }

  /**
   * @brief Get remote RMA IOV for a specific rank and channel
   * @param rank Remote rank
   * @param ch Channel index
   * @return Reference to RMA IOV
   */
  [[nodiscard]] const RmaIov& GetRemoteRmaIov(int rank, size_t ch) const noexcept {
    ASSERT(rank >= 0 && rank < this->world_size_ && ch < rma_iovs_[rank].size());
    return rma_iovs_[rank][ch];
  }

  /**
   * @brief Get all remote RMA IOVs for a specific rank
   * @param rank Remote rank
   * @return Reference to vector of RMA IOVs
   */
  [[nodiscard]] const std::vector<RmaIov>& GetRemoteRmaIovs(int rank) const noexcept {
    ASSERT(rank >= 0 && rank < this->world_size_);
    return rma_iovs_[rank];
  }

  /**
   * @brief Write to remote rank on specific channel
   * @param rank Target rank
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Bytes written
   */
  [[nodiscard]] Coro<ssize_t> Write(int rank, uint64_t imm_data, size_t ch) {
    const auto& iov = GetRemoteRmaIov(rank, ch);
    return BufferType::Write(rank, iov.addr, iov.key, imm_data, ch);
  }

  /**
   * @brief Write all to remote rank on specific channel
   * @param rank Target rank
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Total bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(int rank, uint64_t imm_data, size_t ch) {
    const auto& iov = GetRemoteRmaIov(rank, ch);
    return BufferType::Writeall(rank, iov.addr, iov.key, imm_data, ch);
  }

  /**
   * @brief Write to remote rank across all channels (MultiDMA)
   * @param rank Target rank
   * @param imm_data Immediate data (encoded with channel for each chunk)
   * @return Total bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(int rank, uint64_t imm_data) {
    ASSERT(rank >= 0 && rank < this->world_size_);
    ASSERT(!this->channels_[rank].empty());
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
      ASSERT(ch < this->mrs_.size());
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

  /**
   * @brief Wait for immediate data from all channels
   * @param imm_data Base immediate data (encoded with channel for each wait)
   */
  [[nodiscard]] Coro<> WaitallImmdata(uint64_t imm_data) {
    const size_t num_channels = this->mrs_.size();
    for (size_t ch = 0; ch < num_channels; ++ch) {
      co_await BufferType::WaitImmdata(Base::EncodeImmdata(imm_data, ch));
    }
  }

  /** @brief Send all data to remote rank on specific channel */
  [[nodiscard]] Coro<ssize_t> Sendall(int rank, size_t ch) { return Backend::template Sendall<BufferType>(*this, rank, ch); }

  /** @brief Receive all data from remote rank on specific channel */
  [[nodiscard]] Coro<ssize_t> Recvall(int rank, size_t ch) { return Backend::template Recvall<BufferType>(*this, rank, ch); }

 private:
  std::vector<std::vector<RmaIov>> rma_iovs_;  ///< Remote RMA IOVs [world_size][num_channels]
};

}  // namespace rdma
