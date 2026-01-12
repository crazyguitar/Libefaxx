/**
 * @file buffer.h
 * @brief Buffer management for RDMA operations
 */
#pragma once

#include <cuda.h>
#include <io/common.h>
#include <io/coro.h>
#include <rdma/buffer.h>
#include <rdma/fabric/channel.h>
#include <rdma/fabric/selector.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <array>
#include <device/common.cuh>
#include <limits>
#include <unordered_map>
#include <vector>

namespace fi {

using ImmdataAwaiter = rdma::ImmdataAwaiter<FabricSelector, ImmContext>;

/**
 * @brief Base buffer class for RDMA channel operations
 *
 * Abstract base class managing memory registration and RDMA operations.
 * References 2D channel structure: channels[world_size][num_channels]
 */
class Buffer : private NoCopy {
  static constexpr size_t kAlign = 128;

 public:
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
   * @return Pointer to memory registered with libfabric
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
   * @param buffer Buffer to send
   * @param len Bytes to send
   * @param rank Target rank
   * @param ch Channel index
   * @return Bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Send(void* __restrict__ buffer, size_t len, int rank, size_t ch) {
    ASSERT(buffer && len > 0 && ch < channels_[rank].size() && ch < mrs_.size());
    auto rc = co_await channels_[rank][ch].Send(buffer, len, mrs_[ch]);
    if (rc < 0) [[unlikely]]
      throw std::runtime_error(fmt::format("fi_sendmsg fail. error({}): {}", rc, fi_strerror(-rc)));
    co_return rc;
  }

  /**
   * @brief Receive data through specified channel
   * @param buffer Buffer to receive into
   * @param len Maximum bytes to receive
   * @param rank Source rank
   * @param ch Channel index
   * @return Bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recv(void* __restrict__ buffer, size_t len, int rank, size_t ch) {
    ASSERT(buffer && len > 0 && ch < channels_[rank].size() && ch < mrs_.size());
    auto rc = co_await channels_[rank][ch].Recv(buffer, len, mrs_[ch]);
    if (rc < 0) [[unlikely]]
      throw std::runtime_error(fmt::format("fi_recvmsg fail. error({}): {}", rc, fi_strerror(-rc)));
    co_return rc;
  }

  /**
   * @brief Write data to remote memory
   * @param buffer Buffer to write
   * @param len Bytes to write
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param rank Target rank
   * @param ch Channel index
   * @return Bytes written
   */
  [[nodiscard]] Coro<ssize_t> Write(void* __restrict__ buffer, size_t len, uint64_t addr, uint64_t key, uint64_t imm_data, int rank, size_t ch) {
    ASSERT(buffer && len > 0 && ch < channels_[rank].size() && ch < mrs_.size());
    auto rc = co_await channels_[rank][ch].Write(buffer, len, mrs_[ch], addr, key, imm_data);
    if (rc < 0) [[unlikely]]
      throw std::runtime_error(fmt::format("fi_writemsg fail. error({}): {}", rc, fi_strerror(-rc)));
    co_return rc;
  }

  /**
   * @brief Send all data to remote peer
   * @param buffer Buffer to send
   * @param len Total bytes to send
   * @param rank Target rank
   * @param ch Channel index
   * @return Total bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Sendall(void* __restrict__ buffer, size_t len, int rank, size_t ch) {
    ASSERT(buffer && len > 0 && ch < channels_[rank].size() && ch < mrs_.size());
    auto rc = co_await channels_[rank][ch].Sendall(buffer, len, mrs_[ch]);
    if (rc < 0) [[unlikely]]
      throw std::runtime_error(fmt::format("fi_sendmsg fail. error({}): {}", rc, fi_strerror(-rc)));
    co_return rc;
  }

  /**
   * @brief Receive all data from remote peer
   * @param buffer Buffer to receive into
   * @param len Total bytes to receive
   * @param rank Source rank
   * @param ch Channel index
   * @return Total bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recvall(void* __restrict__ buffer, size_t len, int rank, size_t ch) {
    ASSERT(buffer && len > 0 && ch < channels_[rank].size() && ch < mrs_.size());
    auto rc = co_await channels_[rank][ch].Recvall(buffer, len, mrs_[ch]);
    if (rc < 0) [[unlikely]]
      throw std::runtime_error(fmt::format("fi_recvmsg fail. error({}): {}", rc, fi_strerror(-rc)));
    co_return rc;
  }

  /**
   * @brief Write all data to remote memory
   * @param buffer Buffer to write
   * @param len Bytes to write
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param rank Target rank
   * @param ch Channel index
   * @return Total bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(void* __restrict__ buffer, size_t len, uint64_t addr, uint64_t key, uint64_t imm_data, int rank, size_t ch) {
    ASSERT(buffer && len > 0 && ch < channels_[rank].size() && ch < mrs_.size());
    auto rc = co_await channels_[rank][ch].Writeall(buffer, len, mrs_[ch], addr, key, imm_data);
    if (rc < 0) [[unlikely]]
      throw std::runtime_error(fmt::format("fi_writemsg fail. error({}): {}", rc, fi_strerror(-rc)));
    co_return rc;
  }

  /**
   * @brief Send from internal buffer
   * @param len Bytes to send
   * @param rank Target rank
   * @param ch Channel index
   * @return Bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Send(size_t len, int rank, size_t ch) { co_return co_await Send(data_, len, rank, ch); }

  /**
   * @brief Receive into internal buffer
   * @param len Maximum bytes to receive
   * @param rank Source rank
   * @param ch Channel index
   * @return Bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recv(size_t len, int rank, size_t ch) { co_return co_await Recv(data_, len, rank, ch); }

  /**
   * @brief Write from internal buffer to remote memory
   * @param len Bytes to write
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param rank Target rank
   * @param ch Channel index
   * @return Bytes written
   */
  [[nodiscard]] Coro<ssize_t> Write(size_t len, uint64_t addr, uint64_t key, uint64_t imm_data, int rank, size_t ch) {
    co_return co_await Write(data_, len, addr, key, imm_data, rank, ch);
  }

  /**
   * @brief Send entire internal buffer
   * @param rank Target rank
   * @param ch Channel index
   * @return Bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Send(int rank, size_t ch) { co_return co_await Send(data_, size_, rank, ch); }

  /**
   * @brief Receive into entire internal buffer
   * @param rank Source rank
   * @param ch Channel index
   * @return Bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recv(int rank, size_t ch) { co_return co_await Recv(data_, size_, rank, ch); }

  /**
   * @brief Write entire internal buffer to remote memory
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param rank Target rank
   * @param ch Channel index
   * @return Bytes written
   */
  [[nodiscard]] Coro<ssize_t> Write(uint64_t addr, uint64_t key, uint64_t imm_data, int rank, size_t ch) {
    co_return co_await Write(data_, size_, addr, key, imm_data, rank, ch);
  }

  /**
   * @brief Send all from internal buffer
   * @param len Bytes to send
   * @param rank Target rank
   * @param ch Channel index
   * @return Total bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Sendall(size_t len, int rank, size_t ch) { co_return co_await Sendall(data_, len, rank, ch); }

  /**
   * @brief Receive all into internal buffer
   * @param len Bytes to receive
   * @param rank Source rank
   * @param ch Channel index
   * @return Total bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recvall(size_t len, int rank, size_t ch) { co_return co_await Recvall(data_, len, rank, ch); }

  /**
   * @brief Write all from internal buffer to remote memory
   * @param len Bytes to write
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param rank Target rank
   * @param ch Channel index
   * @return Total bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(size_t len, uint64_t addr, uint64_t key, uint64_t imm_data, int rank, size_t ch) {
    co_return co_await Writeall(data_, len, addr, key, imm_data, rank, ch);
  }

  /**
   * @brief Send all of entire internal buffer
   * @param rank Target rank
   * @param ch Channel index
   * @return Total bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Sendall(int rank, size_t ch) { co_return co_await Sendall(data_, size_, rank, ch); }

  /**
   * @brief Receive all into entire internal buffer
   * @param rank Source rank
   * @param ch Channel index
   * @return Total bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recvall(int rank, size_t ch) { co_return co_await Recvall(data_, size_, rank, ch); }

  /**
   * @brief Write all of entire internal buffer to remote memory
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param rank Target rank
   * @param ch Channel index
   * @return Total bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(uint64_t addr, uint64_t key, uint64_t imm_data, int rank, size_t ch) {
    co_return co_await Writeall(data_, size_, addr, key, imm_data, rank, ch);
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

 protected:
  Buffer(std::vector<EFA>& efas, std::vector<std::vector<Channel>>& channels, size_t size) : efas_{efas}, channels_{channels}, size_{size} {
    ASSERT(size > 0 && !efas_.empty());
  }

  /**
   * @brief Align pointer to specified boundary
   */
  [[nodiscard]] static constexpr void* Align(void* ptr, size_t align) noexcept { return (void*)(((uintptr_t)ptr + align - 1) & ~(align - 1)); }

  /**
   * @brief Build RMA IOV from memory region
   */
  [[nodiscard]] static fi_rma_iov MakeRmaIov(void* data, size_t size, fid_mr* mr) noexcept {
    return {reinterpret_cast<uint64_t>(data), size, mr->key};
  }

  std::vector<EFA>& efas_;                       ///< Reference to Peer's EFAs [num_channels]
  std::vector<std::vector<Channel>>& channels_;  ///< Reference to Peer's channels[world_size][num_channels]
  std::vector<struct fid_mr*> mrs_;              ///< Memory regions [num_channels] - one MR per EFA
  size_t size_ = 0;
  void* raw_ = nullptr;
  void* data_ = nullptr;
};

/**
 * @brief Host memory buffer for RDMA operations
 */
class HostBuffer : public Buffer {
 public:
  static constexpr size_t kAlign = 128;

  /**
   * @brief Construct HostBuffer with host memory allocation
   * @param efas EFA endpoints for memory registration
   * @param channels 2D channel array [world_size][num_channels]
   * @param device CUDA device ID (unused for host buffer)
   * @param size Buffer size in bytes
   * @param align Memory alignment (default: 128)
   */
  HostBuffer(std::vector<EFA>& efas, std::vector<std::vector<Channel>>& channels, int /*device*/, size_t size, size_t align = kAlign)
      : Buffer(efas, channels, size) {
    ASSERT(align > 0 && (align & (align - 1)) == 0);
    raw_ = malloc(size + align - 1);
    ASSERT(raw_);
    data_ = Align(raw_, align);
    mrs_ = Register(efas_, data_, size_);
  }

  ~HostBuffer() override {
    for (auto mr : mrs_) fi_close((fid_t)mr);
    mrs_.clear();
    if (raw_) {
      free(raw_);
      raw_ = nullptr;
    }
  }

 protected:
  static struct fid_mr* Register(EFA& efa, void* __restrict__ data, size_t size) {
    struct fid_mr* mr;
    struct fi_mr_attr mr_attr = {};
    struct iovec iov = {.iov_base = data, .iov_len = size};
    mr_attr.mr_iov = &iov;
    mr_attr.iov_count = 1;
    mr_attr.access = FI_SEND | FI_RECV | FI_REMOTE_WRITE | FI_REMOTE_READ | FI_WRITE | FI_READ;
    FI_CHECK(fi_mr_regattr(efa.GetDomain(), &mr_attr, 0, &mr));
    return mr;
  }

  static std::vector<struct fid_mr*> Register(std::vector<EFA>& efas, void* __restrict__ data, size_t size) {
    std::vector<struct fid_mr*> mrs;
    mrs.reserve(efas.size());
    for (auto& efa : efas) mrs.push_back(Register(efa, data, size));
    return mrs;
  }
};

/**
 * @brief GPU device memory buffer for RDMA operations using DMABUF
 */
class DeviceDMABuffer : public Buffer {
 public:
  static constexpr size_t kAlign = 4096;

  /**
   * @brief Construct DeviceDMABuffer with GPU memory allocation and dmabuf export
   * @param efas EFA endpoints for memory registration
   * @param channels 2D channel array [world_size][num_channels]
   * @param device CUDA device ID
   * @param size Buffer size in bytes
   * @param align Memory alignment (default: 4096)
   */
  DeviceDMABuffer(std::vector<EFA>& efas, std::vector<std::vector<Channel>>& channels, int device, size_t size, size_t align = kAlign)
      : Buffer(efas, channels, size), device_{device} {
    ASSERT(align > 0 && (align & (align - 1)) == 0);
    const size_t page_size = sysconf(_SC_PAGESIZE);
    const size_t effective_align = std::max(align, page_size);
    const size_t alloc_size = ((size + page_size - 1) / page_size) * page_size;
    CUDA_CHECK(cudaMalloc(&raw_, alloc_size + effective_align - 1));
    data_ = Align(raw_, effective_align);
    const size_t dmabuf_size = ((size + page_size - 1) / page_size) * page_size;
    CU_CHECK(cuMemGetHandleForAddressRange(&dmabuf_fd_, (CUdeviceptr)data_, dmabuf_size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
    ASSERT(dmabuf_fd_ != -1);
    try {
      mrs_ = Register(efas_, data_, size_, dmabuf_fd_, device_);
    } catch (...) {
      close(dmabuf_fd_);
      cudaFree(raw_);
      throw;
    }
  }

  ~DeviceDMABuffer() override {
    for (auto mr : mrs_) fi_close((fid_t)mr);
    mrs_.clear();
    if (dmabuf_fd_ >= 0) {
      close(dmabuf_fd_);
      dmabuf_fd_ = -1;
    }
    if (raw_) {
      cudaFree(raw_);
      raw_ = nullptr;
    }
  }

 protected:
  static struct fid_mr* Register(EFA& efa, void* __restrict__ data, size_t size, int dmabuf_fd, int device) {
    struct fid_mr* mr;
    struct fi_mr_attr mr_attr = {};
    struct fi_mr_dmabuf dmabuf = {};
    dmabuf.fd = dmabuf_fd;
    dmabuf.offset = 0;
    dmabuf.len = size;
    dmabuf.base_addr = data;
    mr_attr.iov_count = 1;
    mr_attr.access = FI_SEND | FI_RECV | FI_REMOTE_WRITE | FI_REMOTE_READ | FI_WRITE | FI_READ;
    mr_attr.iface = FI_HMEM_CUDA;
    mr_attr.device.cuda = device;
    mr_attr.dmabuf = &dmabuf;
    FI_CHECK(fi_mr_regattr(efa.GetDomain(), &mr_attr, FI_MR_DMABUF, &mr));
    return mr;
  }

  static std::vector<struct fid_mr*> Register(std::vector<EFA>& efas, void* __restrict__ data, size_t size, int dmabuf_fd, int device) {
    std::vector<struct fid_mr*> mrs;
    mrs.reserve(efas.size());
    for (auto& efa : efas) mrs.push_back(Register(efa, data, size, dmabuf_fd, device));
    return mrs;
  }

  int device_ = -1;
  int dmabuf_fd_ = -1;
};

/**
 * @brief GPU buffer using pinned host memory for RDMA operations
 */
class DevicePinBuffer : public Buffer {
 public:
  static constexpr size_t kAlign = 4096;

  /**
   * @brief Construct DevicePinBuffer with pinned host memory
   * @param efas EFA endpoints for memory registration
   * @param channels 2D channel array [world_size][num_channels]
   * @param device CUDA device ID
   * @param size Buffer size in bytes
   * @param align Memory alignment (default: 4096)
   */
  DevicePinBuffer(std::vector<EFA>& efas, std::vector<std::vector<Channel>>& channels, int device, size_t size, size_t align = kAlign)
      : Buffer(efas, channels, size), device_{device} {
    try {
      Alloc(size, align);
    } catch (...) {
      Dealloc();
      throw;
    }
  }

  ~DevicePinBuffer() override { Dealloc(); }

  /** @brief Get CPU-accessible mapped pointer */
  [[nodiscard]] void* MappedData() noexcept { return mapped_data_; }

  /** @brief Get pointer to RDMA-registered data */
  [[nodiscard]] void* RdmaData() noexcept override { return mapped_data_; }

  /**
   * @brief Send data through specified channel
   * @param len Bytes to send
   * @param rank Target rank
   * @param ch Channel index
   * @return Bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Send(size_t len, int rank, size_t ch) { co_return co_await Buffer::Send(mapped_data_, len, rank, ch); }

  /**
   * @brief Receive data through specified channel
   * @param len Maximum bytes to receive
   * @param rank Source rank
   * @param ch Channel index
   * @return Bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recv(size_t len, int rank, size_t ch) { co_return co_await Buffer::Recv(mapped_data_, len, rank, ch); }

  /**
   * @brief Write data to remote memory
   * @param len Bytes to write
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param rank Target rank
   * @param ch Channel index
   * @return Bytes written
   */
  [[nodiscard]] Coro<ssize_t> Write(size_t len, uint64_t addr, uint64_t key, uint64_t imm_data, int rank, size_t ch) {
    co_return co_await Buffer::Write(mapped_data_, len, addr, key, imm_data, rank, ch);
  }

  /**
   * @brief Send entire buffer
   * @param rank Target rank
   * @param ch Channel index
   * @return Bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Send(int rank, size_t ch) { co_return co_await Buffer::Send(mapped_data_, size_, rank, ch); }

  /**
   * @brief Receive into entire buffer
   * @param rank Source rank
   * @param ch Channel index
   * @return Bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recv(int rank, size_t ch) { co_return co_await Buffer::Recv(mapped_data_, size_, rank, ch); }

  /**
   * @brief Write entire buffer to remote memory
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param rank Target rank
   * @param ch Channel index
   * @return Bytes written
   */
  [[nodiscard]] Coro<ssize_t> Write(uint64_t addr, uint64_t key, uint64_t imm_data, int rank, size_t ch) {
    co_return co_await Buffer::Write(mapped_data_, size_, addr, key, imm_data, rank, ch);
  }

  /**
   * @brief Send all data
   * @param len Bytes to send
   * @param rank Target rank
   * @param ch Channel index
   * @return Total bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Sendall(size_t len, int rank, size_t ch) { co_return co_await Buffer::Sendall(mapped_data_, len, rank, ch); }

  /**
   * @brief Receive all data
   * @param len Bytes to receive
   * @param rank Source rank
   * @param ch Channel index
   * @return Total bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recvall(size_t len, int rank, size_t ch) { co_return co_await Buffer::Recvall(mapped_data_, len, rank, ch); }

  /**
   * @brief Write all data to remote memory
   * @param len Bytes to write
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param rank Target rank
   * @param ch Channel index
   * @return Total bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(void* __restrict__, size_t len, uint64_t addr, uint64_t key, uint64_t imm_data, int rank, size_t ch) {
    co_return co_await Buffer::Writeall(mapped_data_, len, addr, key, imm_data, rank, ch);
  }

  /**
   * @brief Write all data to remote memory
   * @param len Bytes to write
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param rank Target rank
   * @param ch Channel index
   * @return Total bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(size_t len, uint64_t addr, uint64_t key, uint64_t imm_data, int rank, size_t ch) {
    co_return co_await Buffer::Writeall(mapped_data_, len, addr, key, imm_data, rank, ch);
  }

  /**
   * @brief Send all of entire buffer
   * @param rank Target rank
   * @param ch Channel index
   * @return Total bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Sendall(int rank, size_t ch) { co_return co_await Buffer::Sendall(mapped_data_, size_, rank, ch); }

  /**
   * @brief Receive all into entire buffer
   * @param rank Source rank
   * @param ch Channel index
   * @return Total bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recvall(int rank, size_t ch) { co_return co_await Buffer::Recvall(mapped_data_, size_, rank, ch); }

  /**
   * @brief Write all of entire buffer to remote memory
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param rank Target rank
   * @param ch Channel index
   * @return Total bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(uint64_t addr, uint64_t key, uint64_t imm_data, int rank, size_t ch) {
    co_return co_await Buffer::Writeall(mapped_data_, size_, addr, key, imm_data, rank, ch);
  }

 protected:
  static struct fid_mr* Register(EFA& efa, void* __restrict__ data, size_t size) {
    struct fid_mr* mr;
    struct fi_mr_attr mr_attr = {};
    struct iovec iov = {.iov_base = data, .iov_len = size};
    mr_attr.mr_iov = &iov;
    mr_attr.iov_count = 1;
    mr_attr.access = FI_SEND | FI_RECV | FI_REMOTE_WRITE | FI_REMOTE_READ | FI_WRITE | FI_READ;
    mr_attr.iface = FI_HMEM_SYSTEM;
    FI_CHECK(fi_mr_regattr(efa.GetDomain(), &mr_attr, 0, &mr));
    return mr;
  }

  static std::vector<struct fid_mr*> Register(std::vector<EFA>& efas, void* __restrict__ data, size_t size) {
    std::vector<struct fid_mr*> mrs;
    mrs.reserve(efas.size());
    for (auto& efa : efas) mrs.push_back(Register(efa, data, size));
    return mrs;
  }

 private:
  void Alloc(size_t size, size_t align) {
    ASSERT(align > 0 && (align & (align - 1)) == 0);
    const size_t page_size = sysconf(_SC_PAGESIZE);
    const size_t effective_align = std::max(align, page_size);
    const size_t alloc_size = ((size + page_size - 1) / page_size) * page_size;

    void* host_raw = nullptr;
    int rc = posix_memalign(&host_raw, effective_align, alloc_size + effective_align - 1);
    if (rc != 0 || !host_raw) throw std::runtime_error("DevicePinBuffer: posix_memalign failed");
    raw_ = host_raw;
    mapped_data_ = Align(raw_, effective_align);

    CUDA_CHECK(cudaHostRegister(mapped_data_, alloc_size, cudaHostRegisterMapped | cudaHostRegisterPortable));
    CUDA_CHECK(cudaHostGetDevicePointer(&data_, mapped_data_, 0));

    mrs_ = Register(efas_, mapped_data_, size_);
  }

  void Dealloc() {
    for (auto mr : mrs_) fi_close((fid_t)mr);
    mrs_.clear();
    if (mapped_data_) {
      cudaHostUnregister(mapped_data_);
      mapped_data_ = nullptr;
    }
    if (raw_) {
      free(raw_);
      raw_ = nullptr;
    }
  }

  int device_ = -1;
  void* mapped_data_ = nullptr;
};

}  // namespace fi
