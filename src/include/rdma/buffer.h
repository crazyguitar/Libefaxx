/**
 * @file buffer.h
 * @brief Shared RDMA buffer utilities and base template
 *
 * Common awaiter for immediate data operations used by both IB and Fabric buffers
 */
#pragma once

#include <io/common.h>
#include <io/coro.h>
#include <io/handle.h>
#include <io/io.h>
#include <spdlog/fmt/fmt.h>

#include <vector>

namespace rdma {

/**
 * @brief Awaiter for immediate data operations
 */
template <typename Selector, typename ImmCtx>
struct ImmdataAwaiter {
  uint64_t imm_data{0};
  ImmCtx context{0};

  constexpr bool await_ready() const noexcept { return false; }

  void await_resume() noexcept { IO::Get().Quit<Selector>(context); }

  template <typename Promise>
  bool await_suspend(std::coroutine_handle<Promise> coroutine) {
    if (imm_data == 0) [[unlikely]]
      return false;
    context.handle = &coroutine.promise();
    context.imm_data = imm_data;
    if (IO::Get().Join<Selector>(context)) return false;
    coroutine.promise().SetState(Handle::kSuspend);
    return true;
  }
};

/**
 * @brief Base buffer class template for RDMA operations
 *
 * Abstract base class managing memory registration and RDMA operations.
 * References 2D channel structure: channels[world_size][num_channels]
 *
 * @tparam Backend Backend traits defining types and error handling
 */
template <typename Backend>
class Buffer : private NoCopy {
 public:
  using EFA = typename Backend::EFA;
  using Channel = typename Backend::Channel;
  using MR = typename Backend::MR;
  using RmaIov = typename Backend::RmaIov;
  using KeyType = typename Backend::KeyType;
  using ImmdataAwaiter = typename Backend::ImmdataAwaiter;

  Buffer() = delete;
  Buffer(Buffer&& other) = delete;
  Buffer& operator=(Buffer&& other) = delete;
  virtual ~Buffer() = default;

  /**
   * @brief Get pointer to buffer data
   * @return Pointer to aligned memory
   */
  [[nodiscard]] void* Data() noexcept { return data_; }

  /**
   * @brief Get pointer to RDMA-registered data (for use with MRs)
   * @return Pointer to memory registered for RDMA
   * @note Override in subclasses where RDMA registration differs from Data()
   */
  [[nodiscard]] virtual void* RdmaData() noexcept { return data_; }

  /**
   * @brief Get buffer size
   * @return Size in bytes
   */
  [[nodiscard]] size_t Size() const noexcept { return size_; }

  /**
   * @brief Send data through specified channel
   * @param rank Target rank
   * @param buffer Buffer to send
   * @param len Bytes to send
   * @param ch Channel index
   * @return Bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Send(int rank, void* __restrict__ buffer, size_t len, size_t ch)
    requires Backend::kSupportsSendRecv
  {
    ASSERT(buffer && len > 0 && ch < channels_[rank].size() && ch < mrs_.size());
    auto rc = co_await channels_[rank][ch].Send(buffer, len, mrs_[ch]);
    if (rc < 0) [[unlikely]]
      throw std::runtime_error(fmt::format("{} fail. error({}): {}", Backend::kSendOp, rc, Backend::FormatError(rc)));
    co_return rc;
  }

  /**
   * @brief Receive data through specified channel
   * @param rank Source rank
   * @param buffer Buffer to receive into
   * @param len Maximum bytes to receive
   * @param ch Channel index
   * @return Bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recv(int rank, void* __restrict__ buffer, size_t len, size_t ch)
    requires Backend::kSupportsSendRecv
  {
    ASSERT(buffer && len > 0 && ch < channels_[rank].size() && ch < mrs_.size());
    auto rc = co_await channels_[rank][ch].Recv(buffer, len, mrs_[ch]);
    if (rc < 0) [[unlikely]]
      throw std::runtime_error(fmt::format("{} fail. error({}): {}", Backend::kRecvOp, rc, Backend::FormatError(rc)));
    co_return rc;
  }

  /**
   * @brief Write data to remote memory
   * @param rank Target rank
   * @param buffer Buffer to write
   * @param len Bytes to write
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Bytes written
   */
  [[nodiscard]] Coro<ssize_t> Write(int rank, void* __restrict__ buffer, size_t len, uint64_t addr, KeyType key, uint64_t imm_data, size_t ch) {
    ASSERT(buffer && len > 0 && ch < channels_[rank].size() && ch < mrs_.size());
    auto rc = co_await channels_[rank][ch].Write(buffer, len, mrs_[ch], addr, key, imm_data);
    if (rc < 0) [[unlikely]]
      throw std::runtime_error(fmt::format("{} fail. error({}): {}", Backend::kWriteOp, rc, Backend::FormatError(rc)));
    co_return rc;
  }

  /**
   * @brief Send all data to remote peer
   * @param rank Target rank
   * @param buffer Buffer to send
   * @param len Total bytes to send
   * @param ch Channel index
   * @return Total bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Sendall(int rank, void* __restrict__ buffer, size_t len, size_t ch)
    requires Backend::kSupportsSendRecv
  {
    ASSERT(buffer && len > 0 && ch < channels_[rank].size() && ch < mrs_.size());
    auto rc = co_await channels_[rank][ch].Sendall(buffer, len, mrs_[ch]);
    if (rc < 0) [[unlikely]]
      throw std::runtime_error(fmt::format("{} fail. error({}): {}", Backend::kSendOp, rc, Backend::FormatError(rc)));
    co_return rc;
  }

  /**
   * @brief Receive all data from remote peer
   * @param rank Source rank
   * @param buffer Buffer to receive into
   * @param len Total bytes to receive
   * @param ch Channel index
   * @return Total bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recvall(int rank, void* __restrict__ buffer, size_t len, size_t ch)
    requires Backend::kSupportsSendRecv
  {
    ASSERT(buffer && len > 0 && ch < channels_[rank].size() && ch < mrs_.size());
    auto rc = co_await channels_[rank][ch].Recvall(buffer, len, mrs_[ch]);
    if (rc < 0) [[unlikely]]
      throw std::runtime_error(fmt::format("{} fail. error({}): {}", Backend::kRecvOp, rc, Backend::FormatError(rc)));
    co_return rc;
  }

  /**
   * @brief Write all data to remote memory
   * @param rank Target rank
   * @param buffer Buffer to write
   * @param len Bytes to write
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Total bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(int rank, void* __restrict__ buffer, size_t len, uint64_t addr, KeyType key, uint64_t imm_data, size_t ch) {
    ASSERT(buffer && len > 0 && ch < channels_[rank].size() && ch < mrs_.size());
    auto rc = co_await channels_[rank][ch].Writeall(buffer, len, mrs_[ch], addr, key, imm_data);
    if (rc < 0) [[unlikely]]
      throw std::runtime_error(fmt::format("{} fail. error({}): {}", Backend::kWriteOp, rc, Backend::FormatError(rc)));
    co_return rc;
  }

  /** @brief Send from internal buffer */
  [[nodiscard]] Coro<ssize_t> Send(int rank, size_t len, size_t ch)
    requires Backend::kSupportsSendRecv
  {
    co_return co_await Send(rank, data_, len, ch);
  }

  /** @brief Receive into internal buffer */
  [[nodiscard]] Coro<ssize_t> Recv(int rank, size_t len, size_t ch)
    requires Backend::kSupportsSendRecv
  {
    co_return co_await Recv(rank, data_, len, ch);
  }

  /** @brief Send entire internal buffer */
  [[nodiscard]] Coro<ssize_t> Send(int rank, size_t ch)
    requires Backend::kSupportsSendRecv
  {
    co_return co_await Send(rank, data_, size_, ch);
  }

  /** @brief Receive into entire internal buffer */
  [[nodiscard]] Coro<ssize_t> Recv(int rank, size_t ch)
    requires Backend::kSupportsSendRecv
  {
    co_return co_await Recv(rank, data_, size_, ch);
  }

  /** @brief Send all from internal buffer */
  [[nodiscard]] Coro<ssize_t> Sendall(int rank, size_t len, size_t ch)
    requires Backend::kSupportsSendRecv
  {
    co_return co_await Sendall(rank, data_, len, ch);
  }

  /** @brief Receive all into internal buffer */
  [[nodiscard]] Coro<ssize_t> Recvall(int rank, size_t len, size_t ch)
    requires Backend::kSupportsSendRecv
  {
    co_return co_await Recvall(rank, data_, len, ch);
  }

  /** @brief Send all of entire internal buffer */
  [[nodiscard]] Coro<ssize_t> Sendall(int rank, size_t ch)
    requires Backend::kSupportsSendRecv
  {
    co_return co_await Sendall(rank, data_, size_, ch);
  }

  /** @brief Receive all into entire internal buffer */
  [[nodiscard]] Coro<ssize_t> Recvall(int rank, size_t ch)
    requires Backend::kSupportsSendRecv
  {
    co_return co_await Recvall(rank, data_, size_, ch);
  }

  /**
   * @brief Write from internal buffer to remote memory
   * @param rank Target rank
   * @param len Bytes to write
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Bytes written
   */
  [[nodiscard]] Coro<ssize_t> Write(int rank, size_t len, uint64_t addr, KeyType key, uint64_t imm_data, size_t ch) {
    co_return co_await Write(rank, data_, len, addr, key, imm_data, ch);
  }

  /**
   * @brief Write entire internal buffer to remote memory
   * @param rank Target rank
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Bytes written
   */
  [[nodiscard]] Coro<ssize_t> Write(int rank, uint64_t addr, KeyType key, uint64_t imm_data, size_t ch) {
    co_return co_await Write(rank, data_, size_, addr, key, imm_data, ch);
  }

  /**
   * @brief Write all from internal buffer to remote memory
   * @param rank Target rank
   * @param len Bytes to write
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Total bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(int rank, size_t len, uint64_t addr, KeyType key, uint64_t imm_data, size_t ch) {
    co_return co_await Writeall(rank, data_, len, addr, key, imm_data, ch);
  }

  /**
   * @brief Write all of entire internal buffer to remote memory
   * @param rank Target rank
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Total bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(int rank, uint64_t addr, KeyType key, uint64_t imm_data, size_t ch) {
    co_return co_await Writeall(rank, data_, size_, addr, key, imm_data, ch);
  }

  /**
   * @brief Send all data using RDMA WRITE with immediate
   * @param rank Target rank
   * @param buffer Local buffer to send
   * @param len Number of bytes to send
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Total bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Sendall(int rank, void* __restrict__ buffer, size_t len, uint64_t addr, KeyType key, uint64_t imm_data, size_t ch) {
    ASSERT(buffer && len > 0 && ch < channels_[rank].size() && ch < mrs_.size());
    co_return co_await channels_[rank][ch].Sendall(buffer, len, mrs_[ch], addr, key, imm_data);
  }

  /** @brief Send all of entire internal buffer */
  [[nodiscard]] Coro<ssize_t> Sendall(int rank, uint64_t addr, KeyType key, uint64_t imm_data, size_t ch) {
    co_return co_await Sendall(rank, data_, size_, addr, key, imm_data, ch);
  }

  /** @brief Send all using RMA IOV */
  [[nodiscard]] Coro<ssize_t> Sendall(int rank, const RmaIov& iov, uint64_t imm_data, size_t ch) {
    co_return co_await Sendall(rank, data_, size_, iov.addr, iov.key, imm_data, ch);
  }

  /**
   * @brief Receive by waiting for RDMA WRITE with immediate completion
   * @param rank Source rank
   * @param len Expected bytes
   * @param imm_data Expected immediate data
   * @param ch Channel index
   * @return Bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recvall(int rank, size_t len, uint64_t imm_data, size_t ch) {
    ASSERT(len > 0 && ch < channels_[rank].size() && ch < mrs_.size());
    co_return co_await channels_[rank][ch].Recvall(data_, len, mrs_[ch], imm_data);
  }

  /**
   * @brief Wait for immediate data from remote peer
   * @param imm_data Expected immediate data value (must be > 0)
   * @return Coroutine that completes when immediate data is received
   */
  [[nodiscard]] Coro<> WaitImmdata(uint64_t imm_data) {
    if (imm_data == 0) [[unlikely]]
      throw std::invalid_argument("imm_data must be greater than 0");
    co_return co_await ImmdataAwaiter{imm_data};
  }

  /**
   * @brief Build RMA IOV from memory region
   * @param data Pointer to data
   * @param size Size in bytes
   * @param mr Memory region
   * @return RMA IOV structure
   */
  [[nodiscard]] static RmaIov MakeRmaIov(void* data, size_t size, MR* mr) noexcept { return Backend::MakeRmaIov(data, size, mr); }

 protected:
  Buffer(std::vector<EFA>& efas, std::vector<std::vector<Channel>>& channels, size_t size) : efas_{efas}, channels_{channels}, size_{size} {
    ASSERT(size > 0 && !efas_.empty());
  }

  /**
   * @brief Align pointer to specified boundary
   * @param ptr Pointer to align
   * @param align Alignment boundary (must be power of 2)
   * @return Aligned pointer
   */
  [[nodiscard]] static constexpr void* Align(void* ptr, size_t align) noexcept { return (void*)(((uintptr_t)ptr + align - 1) & ~(align - 1)); }

  std::vector<EFA>& efas_;                       ///< Reference to Peer's EFAs [num_channels]
  std::vector<std::vector<Channel>>& channels_;  ///< Reference to Peer's channels[world_size][num_channels]
  std::vector<MR*> mrs_;                         ///< Memory regions [num_channels] - one MR per EFA
  size_t size_ = 0;                              ///< Buffer size in bytes
  void* raw_ = nullptr;                          ///< Raw allocated pointer
  void* data_ = nullptr;                         ///< Aligned data pointer
};

}  // namespace rdma
