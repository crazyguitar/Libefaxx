/**
 * @file memory.h
 * @brief Device memory buffer with RDMA write operations
 *
 * Implementation based on SymmetricMemory design patterns for
 * high-performance GPU-to-GPU communication over fabric networks.
 */
#pragma once

#include <bootstrap/mpi/mpi.h>
#include <rdma/fabric/buffer.h>

/**
 * @brief Template for device buffer class with RDMA write capabilities
 *
 * Provides RDMA write operations using remote memory regions.
 * Manages remote RMA IOVs for write operations to peer buffers.
 *
 * @tparam BufferType The underlying buffer type (DeviceDMABuffer or DevicePinBuffer)
 */
template <typename BufferType>
class DeviceMemory : public BufferType {
 public:
  /**
   * @brief Construct DeviceMemory with GPU memory allocation
   * @param channels Channels for transferring data
   * @param device CUDA device ID
   * @param size Buffer size in bytes
   * @param align Memory alignment in bytes (default: 128)
   */
  DeviceMemory(std::vector<Channel>& channels, int device, size_t size, size_t align = BufferType::kAlign)
      : BufferType(channels, device, size, align) {}

  /**
   * @brief Write data to remote buffer
   * @param buffer Source buffer to write from
   * @param len Number of bytes to write
   * @param imm_data Immediate data to send
   * @param ch Channel index
   * @return Coroutine returning bytes written
   */
  [[nodiscard]] Coro<ssize_t> Write(void* __restrict__ buffer, size_t len, uint64_t imm_data, size_t ch) {
    const auto [addr, key] = GetRemoteAddr(ch, len);
    return BufferType::Write(buffer, len, addr, key, imm_data, ch);
  }

  /**
   * @brief Write all data to remote buffer
   * @param buffer Source buffer to write from
   * @param len Number of bytes to write
   * @param imm_data Immediate data to send
   * @param ch Channel index
   * @return Coroutine returning bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(void* __restrict__ buffer, size_t len, uint64_t imm_data, size_t ch) {
    const auto [addr, key] = GetRemoteAddr(ch, len);
    return BufferType::Writeall(buffer, len, addr, key, imm_data, ch);
  }

  /**
   * @brief Write data from internal buffer to remote buffer
   * @param len Number of bytes to write
   * @param imm_data Immediate data to send
   * @param ch Channel index
   * @return Coroutine returning bytes written
   */
  [[nodiscard]] Coro<ssize_t> Write(size_t len, uint64_t imm_data, size_t ch) {
    const auto [addr, key] = GetRemoteAddr(ch, len);
    return BufferType::Write(len, addr, key, imm_data, ch);
  }

  /**
   * @brief Encode imm_data with channel index for unique identification
   * @param imm_data Base immediate data value
   * @param ch Channel index
   * @return Encoded immediate data with channel in lower 8 bits
   */
  static constexpr uint64_t EncodeImmdata(uint64_t imm_data, size_t ch) noexcept { return (imm_data << 8) | (ch & 0xFF); }

  /**
   * @brief Write all data from entire internal buffer to remote buffer using all channels
   * Splits buffer across all available channels for maximum throughput
   * Each channel sends channel-encoded imm_data to signal completion
   * @param imm_data Immediate data to send (encoded with channel index per channel)
   * @return Coroutine returning total bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(uint64_t imm_data) {
    const size_t num_channels = this->channels_.size();
    const size_t total_size = this->Size();
    const size_t chunk_size = total_size / num_channels;
    const auto& remote_rma = this->GetRemoteRMA();
    char* data = static_cast<char*>(this->RdmaData());

    // Launch all channels in parallel
    std::vector<Future<Coro<ssize_t>>> futures;
    futures.reserve(num_channels);

    for (size_t ch = 0; ch < num_channels; ++ch) {
      size_t offset = ch * chunk_size;
      size_t len = (ch == num_channels - 1) ? (total_size - offset) : chunk_size;
      auto* mr = this->mrs_[ch];
      auto addr = remote_rma[ch].addr + offset;
      auto key = remote_rma[ch].key;
      uint64_t encoded_imm = EncodeImmdata(imm_data, ch);

      futures.emplace_back(this->channels_[ch].Writeall(data + offset, len, mr, addr, key, encoded_imm));
    }

    // Wait for all channels to complete
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
      uint64_t encoded_imm = EncodeImmdata(imm_data, ch);
      co_await BufferType::WaitImmdata(encoded_imm);
    }
  }

  /**
   * @brief Write entire internal buffer to remote buffer
   * @param imm_data Immediate data to send
   * @param ch Channel index
   * @return Coroutine returning bytes written
   */
  [[nodiscard]] Coro<ssize_t> Write(uint64_t imm_data, size_t ch) {
    const auto [addr, key] = GetRemoteAddr(ch, this->Size());
    return BufferType::Write(addr, key, imm_data, ch);
  }

  /**
   * @brief Write all data from internal buffer to remote buffer
   * @param len Number of bytes to write
   * @param imm_data Immediate data to send
   * @param ch Channel index
   * @return Coroutine returning bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(size_t len, uint64_t imm_data, size_t ch) {
    const auto [addr, key] = GetRemoteAddr(ch, len);
    return BufferType::Writeall(len, addr, key, imm_data, ch);
  }

  /**
   * @brief Write all data from entire internal buffer to remote buffer
   * @param imm_data Immediate data to send
   * @param ch Channel index
   * @return Coroutine returning bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(uint64_t imm_data, size_t ch) {
    const auto [addr, key] = GetRemoteAddr(ch, this->Size());
    return BufferType::Writeall(addr, key, imm_data, ch);
  }

 private:
  /**
   * @brief Get remote address and key for write operations
   * @param ch Channel index
   * @param len Length to validate
   * @param offset Offset within the remote buffer (default: 0)
   * @return Pair of remote address and key
   */
  [[nodiscard]] std::pair<uint64_t, uint64_t> GetRemoteAddr(size_t ch, size_t len, size_t offset = 0) const {
    const auto& remote_rma_iov = this->GetRemoteRMA();
    ASSERT(ch < remote_rma_iov.size());
    const auto& rma_iov = remote_rma_iov[ch];
    ASSERT(offset + len <= rma_iov.len);
    return {rma_iov.addr + offset, rma_iov.key};
  }
};

/**
 * @brief Device buffer with DMABUF and RDMA write capabilities
 */
using DeviceDMAMemory = DeviceMemory<DeviceDMABuffer>;

/**
 * @brief Device buffer with pinned host memory and RDMA write capabilities
 */
using DevicePinMemory = DeviceMemory<DevicePinBuffer>;
