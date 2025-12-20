/**
 * @file memory.h
 * @brief Device memory buffer with RDMA write operations
 *
 * Implementation based on SymmetricMemory design patterns for
 * high-performance GPU-to-GPU communication over fabric networks.
 */
#pragma once

#include <bootstrap/mpi/mpi.h>
#include <io/selector.h>
#include <rdma/fabric/buffer.h>
#include <rdma/fabric/request.h>

#include <algorithm>
#include <array>
#include <queue/queue.cuh>

/**
 * @brief Template for device buffer class with RDMA write capabilities
 *
 * Provides RDMA write operations using remote memory regions.
 * Manages remote RMA IOVs for write operations to peer buffers.
 *
 * @tparam BufferType The underlying buffer type (DeviceDMABuffer or DevicePinBuffer)
 */
template <typename BufferType>
class DeviceMemory : public BufferType, public detail::Selector {
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

 public:
  /**
   * @brief Awaiter for retrieving and merging GPU device requests
   *
   * Groups requests by type, merges contiguous address ranges within each type,
   * and returns the combined result when resumed.
   */
  struct DeviceRequestAwaiter {
    DeviceMemory* memory;

    constexpr bool await_ready() const noexcept { return false; }

    /**
     * @brief Merge contiguous requests by type and return combined result
     * @return Vector of merged device requests
     */
    inline std::vector<DeviceRequest> await_resume() noexcept {
      auto& reqs = memory->requests_;
      if (reqs.empty()) return {};

      // Group by type
      std::array<std::vector<DeviceRequest>, static_cast<size_t>(DeviceRequestType::kCount)> grouped;
      for (auto& r : reqs) grouped[r.type].push_back(r);
      reqs.clear();

      // Merge within each type, then combine
      std::vector<DeviceRequest> result;
      for (auto& g : grouped) {
        if (g.empty()) continue;
        std::sort(g.begin(), g.end(), [](const auto& a, const auto& b) { return a.src < b.src; });
        result.push_back(g[0]);
        for (size_t i = 1; i < g.size(); ++i) {
          auto& last = result.back();
          const auto& cur = g[i];
          if (cur.src == last.src + last.size && cur.dst == last.dst + last.size) {
            last.size += cur.size;
          } else {
            result.push_back(cur);
          }
        }
      }
      return result;
    }

    /**
     * @brief Suspend coroutine if no requests available
     * @param coroutine Coroutine handle to suspend
     * @return true if suspended, false if data ready
     */
    template <typename Promise>
    inline bool await_suspend(std::coroutine_handle<Promise> coroutine) noexcept {
      coroutine.promise().SetState(Handle::kSuspend);
      if (!memory->requests_.empty()) return false;
      memory->handles_.emplace_back(&coroutine.promise());
      return true;
    }
  };

  /**
   * @brief Get awaiter for device requests from GPU queue
   * @return DeviceRequestAwaiter for co_await
   */
  [[nodiscard]] auto GetDeviceRequests() { return DeviceRequestAwaiter{this}; }

  /**
   * @brief Poll GPU queue and resume waiting coroutines
   * @param duration Timeout duration (unused)
   * @return Vector of events for ready coroutines
   */
  [[nodiscard]] std::vector<Event> Select(ms duration) override final {
    DeviceRequest req;
    while (queue_.Pop(req)) requests_.emplace_back(req);
    if (requests_.empty() or handles_.empty()) return {};
    std::vector<Event> res;
    res.emplace_back(Event{-1, 0, handles_.front()});
    handles_.pop_front();
    return res;
  }

  [[nodiscard]] bool Stopped() const noexcept override final { return false; }

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

 private:
  Queue<DeviceRequest> queue_;
  std::vector<DeviceRequest> requests_;
  std::deque<Handle*> handles_;
};

/**
 * @brief Device buffer with DMABUF and RDMA write capabilities
 */
using DeviceDMAMemory = DeviceMemory<DeviceDMABuffer>;

/**
 * @brief Device buffer with pinned host memory and RDMA write capabilities
 */
using DevicePinMemory = DeviceMemory<DevicePinBuffer>;
