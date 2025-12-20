/**
 * @file buffer.h
 * @brief Buffer management for RDMA operations
 */
#pragma once

#include <cuda.h>
#include <io/common.h>
#include <io/coro.h>
#include <rdma/fabric/channel.h>
#include <rdma/fabric/selector.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <array>
#include <device/common.cuh>
#include <limits>
#include <vector>

/**
 * @brief Base buffer class for RDMA channel operations
 *
 * Abstract base class managing memory registration and RDMA operations.
 * Subclasses handle specific memory allocation (host vs device).
 */
class Buffer : private NoCopy {
  static constexpr size_t kAlign = 128;

 public:
  /**
   * @brief Awaiter for immediate data operations
   *
   * Coroutine awaiter that suspends until immediate data is received.
   */
  struct ImmdataAwaiter {
    uint64_t imm_data{0};   ///< Expected immediate data value
    ImmContext context{0};  ///< Context for the operation

    constexpr bool await_ready() const noexcept { return false; }

    inline void await_resume() noexcept {
      // Cleanup: remove context from selector
      IO::Get().Quit<FabricSelector>(context);
    }

    template <typename Promise>
    bool await_suspend(std::coroutine_handle<Promise> coroutine) {
      if (imm_data == 0) [[unlikely]]
        return false;
      context.handle = &coroutine.promise();
      context.imm_data = imm_data;
      // Join returns true if completion already arrived (don't suspend)
      if (IO::Get().Join<FabricSelector>(context)) {
        return false;  // Don't suspend, completion already arrived
      }
      coroutine.promise().SetState(Handle::kSuspend);
      return true;
    }
  };

  Buffer() = delete;
  Buffer(Buffer&& other) = delete;
  Buffer& operator=(Buffer&& other) = delete;
  virtual ~Buffer() = default;

  /**
   * @brief Get pointer to buffer data
   * @return Pointer to aligned memory
   */
  [[nodiscard]] inline void* Data() noexcept { return data_; }

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
  [[nodiscard]] inline size_t Size() const noexcept { return size_; }

  /**
   * @brief Send data through specified channel
   * @param buffer Buffer to send
   * @param len Bytes to send
   * @param ch Channel index
   * @return Bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Send(void* __restrict__ buffer, size_t len, size_t ch) {
    ASSERT(buffer && len > 0 && ch < channels_.size() && mrs_[ch]);
    auto rc = co_await channels_[ch].Send(buffer, len, mrs_[ch]);
    if (rc < 0) [[unlikely]] {
      throw std::runtime_error(fmt::format("fi_sendmsg fail. error({}): {}", rc, fi_strerror(-rc)));
    }
    co_return rc;
  }

  /**
   * @brief Receive data through specified channel
   * @param buffer Buffer to receive into
   * @param len Maximum bytes to receive
   * @param ch Channel index
   * @return Bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recv(void* __restrict__ buffer, size_t len, size_t ch) {
    ASSERT(buffer && len > 0 && ch < channels_.size() && mrs_[ch]);
    auto rc = co_await channels_[ch].Recv(buffer, len, mrs_[ch]);
    if (rc < 0) [[unlikely]] {
      throw std::runtime_error(fmt::format("fi_recvmsg fail. error({}): {}", rc, fi_strerror(-rc)));
    }
    co_return rc;
  }

  /**
   * @brief Write data to remote memory
   * @param buffer Buffer to write
   * @param len Bytes to write
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Bytes written
   */
  [[nodiscard]] Coro<ssize_t> Write(void* __restrict__ buffer, size_t len, uint64_t addr, uint64_t key, uint64_t imm_data, size_t ch) {
    ASSERT(buffer && len > 0 && ch < channels_.size() && mrs_[ch]);
    auto rc = co_await channels_[ch].Write(buffer, len, mrs_[ch], addr, key, imm_data);
    if (rc < 0) [[unlikely]] {
      throw std::runtime_error(fmt::format("fi_writemsg fail. error({}): {}", rc, fi_strerror(-rc)));
    }
    co_return rc;
  }

  /**
   * @brief Send all data to remote peer
   * @param buffer Buffer to send
   * @param len Total bytes to send
   * @param ch Channel index
   * @return Total bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Sendall(void* __restrict__ buffer, size_t len, size_t ch) {
    ASSERT(buffer && len > 0 && ch < channels_.size() && mrs_[ch]);
    auto rc = co_await channels_[ch].Sendall(buffer, len, mrs_[ch]);
    if (rc < 0) [[unlikely]] {
      throw std::runtime_error(fmt::format("fi_sendmsg fail. error({}): {}", rc, fi_strerror(-rc)));
    }
    co_return rc;
  }

  /**
   * @brief Receive all data from remote peer
   * @param buffer Buffer to receive into
   * @param len Total bytes to receive
   * @param ch Channel index
   * @return Total bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recvall(void* __restrict__ buffer, size_t len, size_t ch) {
    ASSERT(buffer && len > 0 && ch < channels_.size() && mrs_[ch]);
    auto rc = co_await channels_[ch].Recvall(buffer, len, mrs_[ch]);
    if (rc < 0) [[unlikely]] {
      throw std::runtime_error(fmt::format("fi_recvmsg fail. error({}): {}", rc, fi_strerror(-rc)));
    }
    co_return rc;
  }

  /**
   * @brief Write all data to remote memory
   * @param buffer Buffer to write
   * @param len Bytes to write
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Total bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(void* __restrict__ buffer, size_t len, uint64_t addr, uint64_t key, uint64_t imm_data, size_t ch) {
    ASSERT(buffer && len > 0 && ch < channels_.size() && mrs_[ch]);
    auto rc = co_await channels_[ch].Writeall(buffer, len, mrs_[ch], addr, key, imm_data);
    if (rc < 0) [[unlikely]] {
      throw std::runtime_error(fmt::format("fi_writemsg fail. error({}): {}", rc, fi_strerror(-rc)));
    }
    co_return rc;
  }

  /**
   * @brief Send buffer data through specified channel
   * @param len Bytes to send from buffer
   * @param ch Channel index
   * @return Bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Send(size_t len, size_t ch) { co_return co_await Send(data_, len, ch); }

  /**
   * @brief Receive into buffer through specified channel
   * @param len Maximum bytes to receive
   * @param ch Channel index
   * @return Bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recv(size_t len, size_t ch) { co_return co_await Recv(data_, len, ch); }

  /**
   * @brief Write buffer data to remote memory
   * @param len Bytes to write from buffer
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Bytes written
   */
  [[nodiscard]] Coro<ssize_t> Write(size_t len, uint64_t addr, uint64_t key, uint64_t imm_data, size_t ch) {
    co_return co_await Write(data_, len, addr, key, imm_data, ch);
  }

  /**
   * @brief Send entire buffer through specified channel
   * @param ch Channel index
   * @return Bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Send(size_t ch) { co_return co_await Send(data_, size_, ch); }

  /**
   * @brief Receive into entire buffer through specified channel
   * @param ch Channel index
   * @return Bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recv(size_t ch) { co_return co_await Recv(data_, size_, ch); }

  /**
   * @brief Write entire buffer to remote memory
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Bytes written
   */
  [[nodiscard]] Coro<ssize_t> Write(uint64_t addr, uint64_t key, uint64_t imm_data, size_t ch) {
    co_return co_await Write(data_, size_, addr, key, imm_data, ch);
  }

  /**
   * @brief Send specified length from internal buffer
   * @param len Bytes to send
   * @param ch Channel index
   * @return Total bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Sendall(size_t len, size_t ch) { co_return co_await Sendall(data_, len, ch); }

  /**
   * @brief Receive specified length into internal buffer
   * @param len Bytes to receive
   * @param ch Channel index
   * @return Total bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recvall(size_t len, size_t ch) { co_return co_await Recvall(data_, len, ch); }

  /**
   * @brief Write specified length from buffer to remote memory
   * @param len Bytes to write
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Total bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(size_t len, uint64_t addr, uint64_t key, uint64_t imm_data, size_t ch) {
    co_return co_await Writeall(data_, len, addr, key, imm_data, ch);
  }

  /**
   * @brief Send entire internal buffer
   * @param ch Channel index
   * @return Total bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Sendall(size_t ch) { co_return co_await Sendall(data_, size_, ch); }

  /**
   * @brief Receive into entire internal buffer
   * @param ch Channel index
   * @return Total bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recvall(size_t ch) { co_return co_await Recvall(data_, size_, ch); }

  /**
   * @brief Write entire buffer to remote memory
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Total bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(uint64_t addr, uint64_t key, uint64_t imm_data, size_t ch) {
    co_return co_await Writeall(data_, size_, addr, key, imm_data, ch);
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
  Buffer(std::vector<Channel>& channels, size_t size) : channels_{channels}, size_{size} { ASSERT(size > 0); }

  /**
   * @brief Align pointer to specified boundary
   * @param ptr Pointer to align
   * @param align Alignment boundary (must be power of 2)
   * @return Aligned pointer
   */
  [[nodiscard]] static constexpr void* Align(void* ptr, size_t align) noexcept { return (void*)(((uintptr_t)ptr + align - 1) & ~(align - 1)); }

  /**
   * @brief Build RMA IOV from memory region
   * @param data Buffer data pointer
   * @param size Buffer size
   * @param mr Memory region handle
   * @return RMA IOV structure
   */
  [[nodiscard]] static fi_rma_iov MakeRmaIov(void* data, size_t size, fid_mr* mr) noexcept {
    return {reinterpret_cast<uint64_t>(data), size, mr->key};
  }

  std::vector<Channel>& channels_;
  std::vector<struct fid_mr*> mrs_;
  size_t size_ = 0;
  void* raw_ = nullptr;
  void* data_ = nullptr;
};

/**
 * @brief Host memory buffer for RDMA operations
 *
 * Allocates host memory using malloc and registers with RDMA domain.
 */
class HostBuffer : public Buffer {
 public:
  static constexpr size_t kAlign = 128;

  /**
   * @brief Construct HostBuffer with host memory allocation
   * @param channels Channels for transferring data
   * @param device CUDA device ID (ignored for host buffer)
   * @param size Buffer size in bytes
   * @param align Memory alignment in bytes (default: 128)
   */
  HostBuffer(std::vector<Channel>& channels, int /*device*/, size_t size, size_t align = kAlign) : Buffer(channels, size) {
    ASSERT(align > 0 && (align & (align - 1)) == 0);
    raw_ = malloc(size + align - 1);
    ASSERT(raw_);
    data_ = Align(raw_, align);
    mrs_ = Register(channels_, data_, size_);
  }

  /**
   * @brief Construct HostBuffer without device parameter (legacy)
   * @param channels Channels for transferring data
   * @param size Buffer size in bytes
   * @param align Memory alignment in bytes (default: 128)
   */
  HostBuffer(std::vector<Channel>& channels, size_t size, size_t align = kAlign) : HostBuffer(channels, 0, size, align) {}

  ~HostBuffer() override {
    for (auto mr : mrs_) fi_close((fid_t)mr);
    mrs_.clear();
    if (raw_) {
      free(raw_);
      raw_ = nullptr;
    }
  }

 protected:
  /**
   * @brief Register host buffer with RDMA domain
   * @param channel Channel to register with
   * @param data Buffer data pointer
   * @param size Buffer size in bytes
   * @return Memory region handle
   */
  inline static struct fid_mr* Register(Channel& channel, void* __restrict__ data, size_t size) {
    struct fid_mr* mr;
    struct fi_mr_attr mr_attr = {};
    struct iovec iov = {.iov_base = data, .iov_len = size};
    mr_attr.mr_iov = &iov;
    mr_attr.iov_count = 1;
    mr_attr.access = FI_SEND | FI_RECV | FI_REMOTE_WRITE | FI_REMOTE_READ | FI_WRITE | FI_READ;
    auto efa = channel.GetEFA();
    auto domain = efa->GetDomain();
    constexpr uint64_t flags = 0;
    FI_CHECK(fi_mr_regattr(domain, &mr_attr, flags, &mr));
    return mr;
  }

  /**
   * @brief Register host buffer with multiple channels
   * @param channels Vector of channels to register with
   * @param data Buffer data pointer
   * @param size Buffer size in bytes
   * @return Vector of memory region handles
   */
  inline static std::vector<struct fid_mr*> Register(std::vector<Channel>& channels, void* __restrict__ data, size_t size) {
    const size_t n = channels.size();
    std::vector<struct fid_mr*> mrs;
    mrs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      auto mr = Register(channels[i], data, size);
      ASSERT(!!mr);
      mrs.push_back(mr);
    }
    return mrs;
  }
};

/**
 * @brief GPU device memory buffer for RDMA operations using DMABUF
 *
 * Allocates GPU memory and exports dmabuf for zero-copy RDMA.
 */
class DeviceDMABuffer : public Buffer {
 public:
  /**
   * @brief Construct DeviceDMABuffer with GPU memory allocation and dmabuf export
   * @param channels Channels for transferring data
   * @param device CUDA device ID
   * @param size Buffer size in bytes
   * @param align Memory alignment in bytes (default: 128)
   */
  DeviceDMABuffer(std::vector<Channel>& channels, int device, size_t size, size_t align = kAlign) : Buffer(channels, size), device_{device} {
    ASSERT(align > 0 && (align & (align - 1)) == 0);
    struct cudaPointerAttributes attrs = {};
    const size_t page_size = sysconf(_SC_PAGESIZE);
    const size_t effective_align = std::max(align, page_size);
    const size_t alloc_size = ((size + page_size - 1) / page_size) * page_size;
    ASSERT(alloc_size >= size);
    CUDA_CHECK(cudaMalloc(&raw_, alloc_size + effective_align - 1));
    CUDA_CHECK(cudaPointerGetAttributes(&attrs, raw_));
    ASSERT(attrs.type == cudaMemoryTypeDevice);
    data_ = Align(raw_, effective_align);
    const size_t offset = (uintptr_t)data_ - (uintptr_t)raw_;
    const size_t remaining = alloc_size + effective_align - 1 - offset;
    ASSERT(remaining >= alloc_size);
    const size_t dmabuf_size = ((size + page_size - 1) / page_size) * page_size;
    // cuMemGetHandleForAddressRange requires both dptr and size to be aligned to host page size
    CU_CHECK(cuMemGetHandleForAddressRange(&dmabuf_fd_, (CUdeviceptr)data_, dmabuf_size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
    ASSERT(dmabuf_fd_ != -1);
    try {
      mrs_ = Register(channels_, data_, size_, dmabuf_fd_, device_);
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
  static constexpr size_t kAlign = 4096;  // DMABUF requires page alignment

  /**
   * @brief Register GPU memory with libfabric using dmabuf
   * @param channel Channel to register with
   * @param data GPU memory buffer pointer
   * @param size Buffer size in bytes
   * @param dmabuf_fd CUDA dmabuf file descriptor
   * @param device CUDA device ID
   * @return Memory region handle
   */
  inline static struct fid_mr* Register(Channel& channel, void* __restrict__ data, size_t size, int dmabuf_fd, int device) {
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
    auto efa = channel.GetEFA();
    auto domain = efa->GetDomain();
    constexpr uint64_t flags = FI_MR_DMABUF;
    FI_CHECK(fi_mr_regattr(domain, &mr_attr, flags, &mr));
    return mr;
  }

  /**
   * @brief Register GPU memory with multiple channels
   * @param channels Vector of channels to register with
   * @param data GPU memory buffer pointer
   * @param size Buffer size in bytes
   * @param dmabuf_fd CUDA dmabuf file descriptor
   * @param device CUDA device ID
   * @return Vector of memory region handles
   */
  inline static std::vector<struct fid_mr*>
  Register(std::vector<Channel>& channels, void* __restrict__ data, size_t size, int dmabuf_fd, int device) {
    const size_t n = channels.size();
    std::vector<struct fid_mr*> mrs;
    mrs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      auto mr = Register(channels[i], data, size, dmabuf_fd, device);
      ASSERT(!!mr);
      mrs.push_back(mr);
    }
    return mrs;
  }

  int device_ = -1;
  int dmabuf_fd_ = -1;
};

/**
 * @brief GPU buffer using pinned host memory for RDMA operations
 *
 * Allocates page-aligned host memory, pins/maps it with CUDA, registers the
 * host pointer for RDMA, and exposes the mapped device pointer for kernels.
 */
class DevicePinBuffer : public Buffer {
 public:
  /**
   * @brief Construct DevicePinBuffer with GPU memory allocation and CUDA mapping
   * @param channels Channels for transferring data
   * @param device CUDA device ID
   * @param size Buffer size in bytes
   * @param align Memory alignment in bytes (default: 128)
   */
  DevicePinBuffer(std::vector<Channel>& channels, int device, size_t size, size_t align = kAlign) : Buffer(channels, size), device_{device} {
    try {
      Alloc(size, align);
    } catch (...) {
      Dealloc();
      throw;
    }
  }

  ~DevicePinBuffer() override { Dealloc(); }

  /**
   * @brief Get CPU-accessible mapped pointer
   * @return Pointer to CPU-accessible mapped memory
   */
  [[nodiscard]] inline void* MappedData() noexcept { return mapped_data_; }

  /**
   * @brief Get pointer to RDMA-registered data
   * @return Pointer to pinned host memory registered with libfabric (FI_HMEM_SYSTEM)
   */
  [[nodiscard]] void* RdmaData() noexcept override { return mapped_data_; }

  /**
   * @brief Send data through specified channel
   * @param len Bytes to send
   * @param ch Channel index
   * @return Bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Send(size_t len, size_t ch) { co_return co_await Buffer::Send(mapped_data_, len, ch); }

  /**
   * @brief Receive data through specified channel
   * @param len Maximum bytes to receive
   * @param ch Channel index
   * @return Bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recv(size_t len, size_t ch) { co_return co_await Buffer::Recv(mapped_data_, len, ch); }

  /**
   * @brief Write data to remote memory
   * @param len Bytes to write
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Bytes written
   */
  [[nodiscard]] Coro<ssize_t> Write(size_t len, uint64_t addr, uint64_t key, uint64_t imm_data, size_t ch) {
    co_return co_await Buffer::Write(mapped_data_, len, addr, key, imm_data, ch);
  }

  /**
   * @brief Send entire buffer through specified channel
   * @param ch Channel index
   * @return Bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Send(size_t ch) { co_return co_await Buffer::Send(mapped_data_, size_, ch); }

  /**
   * @brief Receive into entire buffer through specified channel
   * @param ch Channel index
   * @return Bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recv(size_t ch) { co_return co_await Buffer::Recv(mapped_data_, size_, ch); }

  /**
   * @brief Write entire buffer to remote memory
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Bytes written
   */
  [[nodiscard]] Coro<ssize_t> Write(uint64_t addr, uint64_t key, uint64_t imm_data, size_t ch) {
    co_return co_await Buffer::Write(mapped_data_, size_, addr, key, imm_data, ch);
  }

  /**
   * @brief Send all data through specified channel
   * @param len Bytes to send
   * @param ch Channel index
   * @return Bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Sendall(size_t len, size_t ch) { co_return co_await Buffer::Sendall(mapped_data_, len, ch); }

  /**
   * @brief Receive all data through specified channel
   * @param len Bytes to receive
   * @param ch Channel index
   * @return Bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recvall(size_t len, size_t ch) { co_return co_await Buffer::Recvall(mapped_data_, len, ch); }

  /**
   * @brief Write all data to remote memory
   * @param len Bytes to write
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Bytes written
   */
  /**
   * @brief Write all data to remote memory (overloaded for compatibility)
   * @param buffer Buffer to write (ignored, uses mapped_data_)
   * @param len Bytes to write
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Total bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(void* __restrict__ buffer, size_t len, uint64_t addr, uint64_t key, uint64_t imm_data, size_t ch) {
    co_return co_await Buffer::Writeall(mapped_data_, len, addr, key, imm_data, ch);
  }

  /**
   * @brief Write all data from buffer to remote memory
   * @param len Bytes to write
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Total bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(size_t len, uint64_t addr, uint64_t key, uint64_t imm_data, size_t ch) {
    co_return co_await Buffer::Writeall(mapped_data_, len, addr, key, imm_data, ch);
  }

  /**
   * @brief Send entire buffer through specified channel (using mapped data)
   * @param ch Channel index
   * @return Total bytes sent
   */
  [[nodiscard]] Coro<ssize_t> Sendall(size_t ch) { co_return co_await Buffer::Sendall(mapped_data_, size_, ch); }

  /**
   * @brief Receive into entire buffer through specified channel (using mapped data)
   * @param ch Channel index
   * @return Total bytes received
   */
  [[nodiscard]] Coro<ssize_t> Recvall(size_t ch) { co_return co_await Buffer::Recvall(mapped_data_, size_, ch); }

  /**
   * @brief Write entire buffer to remote memory (using mapped data)
   * @param addr Remote memory address
   * @param key Remote memory key
   * @param imm_data Immediate data
   * @param ch Channel index
   * @return Total bytes written
   */
  [[nodiscard]] Coro<ssize_t> Writeall(uint64_t addr, uint64_t key, uint64_t imm_data, size_t ch) {
    co_return co_await Buffer::Writeall(mapped_data_, size_, addr, key, imm_data, ch);
  }

 protected:
  static constexpr size_t kAlign = 4096;  // Requires page alignment

  /**
   * @brief Register pinned CUDA host memory with libfabric
   * @param channel Channel to register with
   * @param data CPU-accessible pinned host memory pointer
   * @param size Buffer size in bytes
   * @param device CUDA device ID
   * @return Memory region handle
   */
  inline static struct fid_mr* Register(Channel& channel, void* __restrict__ data, size_t size, int device) {
    struct fid_mr* mr;
    struct fi_mr_attr mr_attr = {};
    struct iovec iov = {.iov_base = data, .iov_len = size};
    mr_attr.mr_iov = &iov;
    mr_attr.iov_count = 1;
    mr_attr.access = FI_SEND | FI_RECV | FI_REMOTE_WRITE | FI_REMOTE_READ | FI_WRITE | FI_READ;
    mr_attr.iface = FI_HMEM_SYSTEM;
    (void)device;
    auto efa = channel.GetEFA();
    auto domain = efa->GetDomain();
    FI_CHECK(fi_mr_regattr(domain, &mr_attr, 0, &mr));
    return mr;
  }

  /**
   * @brief Register pinned CUDA host memory with multiple channels
   * @param channels Vector of channels to register with
   * @param data CPU-accessible pinned host memory pointer
   * @param size Buffer size in bytes
   * @param device CUDA device ID
   * @return Vector of memory region handles
   */
  inline static std::vector<struct fid_mr*> Register(std::vector<Channel>& channels, void* __restrict__ data, size_t size, int device) {
    std::vector<struct fid_mr*> mrs;
    mrs.reserve(channels.size());
    for (auto& ch : channels) {
      auto mr = Register(ch, data, size, device);
      ASSERT(!!mr);
      mrs.push_back(mr);
    }
    return mrs;
  }

 private:
  /**
   * @brief Allocate GPU memory and set up CUDA mapping
   * @param size Buffer size in bytes
   * @param align Memory alignment in bytes
   */
  void Alloc(size_t size, size_t align) {
    ASSERT(align > 0 && (align & (align - 1)) == 0);
    const size_t page_size = sysconf(_SC_PAGESIZE);
    const size_t effective_align = std::max(align, page_size);
    const size_t alloc_size = ((size + page_size - 1) / page_size) * page_size;

    // Page-aligned host allocation for registration and mapping
    void* host_raw = nullptr;
    int rc = posix_memalign(&host_raw, effective_align, alloc_size + effective_align - 1);
    if (rc != 0 || !host_raw) {
      throw std::runtime_error("DevicePinBuffer: posix_memalign failed");
    }
    raw_ = host_raw;
    mapped_data_ = Align(raw_, effective_align);

    CUDA_CHECK(cudaHostRegister(mapped_data_, alloc_size, cudaHostRegisterMapped | cudaHostRegisterPortable));
    CUDA_CHECK(cudaHostGetDevicePointer(&data_, mapped_data_, 0));

    mrs_ = Register(channels_, mapped_data_, size_, device_);
  }

  /**
   * @brief Clean up CUDA resources
   */
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

  int device_ = -1;              ///< CUDA device ID
  void* mapped_data_ = nullptr;  ///< Host pinned pointer registered for RDMA
};
