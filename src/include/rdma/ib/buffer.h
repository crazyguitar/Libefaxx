/**
 * @file buffer.h
 * @brief Buffer management for RDMA operations using ibverbs
 *
 * Buffer references 2D channel structure from Peer: channels[world_size][num_channels]
 */
#pragma once

#include <cuda.h>
#include <io/common.h>
#include <io/coro.h>
#include <rdma/buffer.h>
#include <rdma/ib/channel.h>
#include <rdma/ib/selector.h>
#include <unistd.h>

#include <device/common.cuh>
#include <vector>

namespace ib {

using ImmdataAwaiter = rdma::ImmdataAwaiter<IBSelector, ImmContext>;

/**
 * @brief Base buffer class for RDMA channel operations using ibverbs
 *
 * References 2D channel structure: channels[world_size][num_channels]
 */
class Buffer : private NoCopy {
 public:
  Buffer() = delete;
  Buffer(Buffer&& other) = delete;
  Buffer& operator=(Buffer&& other) = delete;
  virtual ~Buffer() = default;

  [[nodiscard]] void* Data() noexcept { return data_; }
  [[nodiscard]] virtual void* RdmaData() noexcept { return data_; }
  [[nodiscard]] size_t Size() const noexcept { return size_; }

  [[nodiscard]] Coro<ssize_t> Write(int rank, void* __restrict__ buffer, size_t len, uint64_t addr, uint32_t key, uint64_t imm_data, size_t ch) {
    ASSERT(buffer && len > 0 && ch < channels_[rank].size() && ch < mrs_.size());
    auto* mr = mrs_[ch];
    ASSERT(mr);
    auto rc = co_await channels_[rank][ch].Write(buffer, len, mr, addr, key, imm_data);
    if (rc < 0) [[unlikely]]
      throw std::runtime_error(fmt::format("ib_writemsg fail. error({}): {}", rc, strerror(-rc)));
    co_return rc;
  }

  [[nodiscard]] Coro<ssize_t> Writeall(int rank, void* __restrict__ buffer, size_t len, uint64_t addr, uint32_t key, uint64_t imm_data, size_t ch) {
    ASSERT(buffer && len > 0 && ch < channels_[rank].size() && ch < mrs_.size());
    auto* mr = mrs_[ch];
    ASSERT(mr);
    auto rc = co_await channels_[rank][ch].Writeall(buffer, len, mr, addr, key, imm_data);
    if (rc < 0) [[unlikely]]
      throw std::runtime_error(fmt::format("ib_writemsg fail. error({}): {}", rc, strerror(-rc)));
    co_return rc;
  }

  [[nodiscard]] Coro<ssize_t> Write(int rank, size_t len, uint64_t addr, uint32_t key, uint64_t imm_data, size_t ch) {
    co_return co_await Write(rank, data_, len, addr, key, imm_data, ch);
  }

  [[nodiscard]] Coro<ssize_t> Write(int rank, uint64_t addr, uint32_t key, uint64_t imm_data, size_t ch) {
    co_return co_await Write(rank, data_, size_, addr, key, imm_data, ch);
  }

  [[nodiscard]] Coro<ssize_t> Writeall(int rank, size_t len, uint64_t addr, uint32_t key, uint64_t imm_data, size_t ch) {
    co_return co_await Writeall(rank, data_, len, addr, key, imm_data, ch);
  }

  [[nodiscard]] Coro<ssize_t> Writeall(int rank, uint64_t addr, uint32_t key, uint64_t imm_data, size_t ch) {
    co_return co_await Writeall(rank, data_, size_, addr, key, imm_data, ch);
  }

  [[nodiscard]] Coro<ssize_t> Sendall(int rank, void* __restrict__ buffer, size_t len, uint64_t addr, uint32_t key, uint64_t imm_data, size_t ch) {
    ASSERT(buffer && len > 0 && ch < channels_[rank].size() && ch < mrs_.size());
    auto* mr = mrs_[ch];
    ASSERT(mr);
    co_return co_await channels_[rank][ch].Sendall(buffer, len, mr, addr, key, imm_data);
  }

  [[nodiscard]] Coro<ssize_t> Sendall(int rank, uint64_t addr, uint32_t key, uint64_t imm_data, size_t ch) {
    co_return co_await Sendall(rank, data_, size_, addr, key, imm_data, ch);
  }

  [[nodiscard]] Coro<ssize_t> Recvall(int rank, size_t len, uint64_t imm_data, size_t ch) {
    ASSERT(len > 0 && ch < channels_[rank].size() && ch < mrs_.size());
    auto* mr = mrs_[ch];
    ASSERT(mr);
    co_return co_await channels_[rank][ch].Recvall(data_, len, mr, imm_data);
  }

  [[nodiscard]] Coro<ssize_t> Recvall(int rank, uint64_t imm_data, size_t ch) { co_return co_await Recvall(rank, size_, imm_data, ch); }

  [[nodiscard]] Coro<> WaitImmdata(uint64_t imm_data) {
    if (imm_data == 0) [[unlikely]]
      throw std::invalid_argument("imm_data must be greater than 0");
    co_return co_await ImmdataAwaiter{imm_data};
  }

 protected:
  Buffer(std::vector<std::vector<Channel>>& channels, size_t size) : channels_{channels}, size_{size} { ASSERT(size > 0); }

  [[nodiscard]] static constexpr void* Align(void* ptr, size_t align) noexcept { return (void*)(((uintptr_t)ptr + align - 1) & ~(align - 1)); }

  [[nodiscard]] static ib_rma_iov MakeRmaIov(void* data, size_t size, ib_mr* mr) noexcept {
    return {reinterpret_cast<uint64_t>(data), size, ib_mr_rkey(mr)};
  }

  std::vector<std::vector<Channel>>& channels_;  // Reference to Peer's channels[world_size][num_channels]
  std::vector<ib_mr*> mrs_;                      // [num_channels] - one MR per channel/EFA
  size_t size_ = 0;
  void* raw_ = nullptr;
  void* data_ = nullptr;
};

/**
 * @brief GPU device memory buffer for RDMA operations using DMABUF and ibverbs
 */
class DeviceDMABuffer : public Buffer {
 public:
  static constexpr size_t kAlign = 4096;

  DeviceDMABuffer(std::vector<std::vector<Channel>>& channels, int device, size_t size, size_t align = kAlign)
      : Buffer(channels, size), device_{device} {
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
      mrs_ = Register(channels_, data_, size_, dmabuf_fd_);
    } catch (...) {
      close(dmabuf_fd_);
      cudaFree(raw_);
      throw;
    }
  }

  ~DeviceDMABuffer() override {
    for (auto* mr : mrs_) ib_mr_close(mr);
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

  [[nodiscard]] cudaIpcMemHandle_t GetIPCHandle() const {
    cudaIpcMemHandle_t handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&handle, data_));
    return handle;
  }

 protected:
  /**
   * @brief Register GPU memory with one MR per EFA/channel
   *
   * Uses first non-empty channel vector to get EFA references
   */
  static std::vector<ib_mr*> Register(std::vector<std::vector<Channel>>& channels, void* __restrict__ data, size_t size, int dmabuf_fd) {
    // Find first non-empty channel vector to determine num_channels
    size_t num_channels = 0;
    for (auto& ch_vec : channels) {
      if (!ch_vec.empty()) {
        num_channels = ch_vec.size();
        break;
      }
    }
    ASSERT(num_channels > 0);

    std::vector<ib_mr*> mrs;
    mrs.reserve(num_channels);

    // Register with each EFA (one MR per channel)
    for (auto& ch_vec : channels) {
      if (ch_vec.empty()) continue;
      for (size_t ch = 0; ch < ch_vec.size(); ++ch) {
        if (mrs.size() > ch) continue;  // Already registered for this channel
        auto* efa = ch_vec[ch].GetEFA();
        auto* domain = efa->GetDomain();
        ibv_mr* ibv_mr_ptr = ibv_reg_dmabuf_mr(
            domain->pd, 0, size, reinterpret_cast<uint64_t>(data), dmabuf_fd,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ
        );
        if (!ibv_mr_ptr) throw std::runtime_error(fmt::format("ibv_reg_dmabuf_mr failed: {}", strerror(errno)));
        mrs.push_back(new ib_mr{ibv_mr_ptr, domain});
      }
      break;  // Only need to register once per EFA
    }
    return mrs;
  }

  int device_ = -1;
  int dmabuf_fd_ = -1;
};

/**
 * @brief Host memory buffer for RDMA operations using ibverbs
 */
class HostBuffer : public Buffer {
 public:
  static constexpr size_t kAlign = 128;

  HostBuffer(std::vector<std::vector<Channel>>& channels, int /*device*/, size_t size, size_t align = kAlign) : Buffer(channels, size) {
    ASSERT(align > 0 && (align & (align - 1)) == 0);
    raw_ = malloc(size + align - 1);
    ASSERT(raw_);
    data_ = Align(raw_, align);
    mrs_ = Register(channels_, data_, size_);
  }

  HostBuffer(std::vector<std::vector<Channel>>& channels, size_t size, size_t align = kAlign) : HostBuffer(channels, 0, size, align) {}

  ~HostBuffer() override {
    for (auto* mr : mrs_) ib_mr_close(mr);
    mrs_.clear();
    if (raw_) {
      free(raw_);
      raw_ = nullptr;
    }
  }

 protected:
  static std::vector<ib_mr*> Register(std::vector<std::vector<Channel>>& channels, void* __restrict__ data, size_t size) {
    size_t num_channels = 0;
    for (auto& ch_vec : channels) {
      if (!ch_vec.empty()) {
        num_channels = ch_vec.size();
        break;
      }
    }
    ASSERT(num_channels > 0);

    std::vector<ib_mr*> mrs;
    mrs.reserve(num_channels);

    for (auto& ch_vec : channels) {
      if (ch_vec.empty()) continue;
      for (size_t ch = 0; ch < ch_vec.size(); ++ch) {
        if (mrs.size() > ch) continue;
        auto* efa = ch_vec[ch].GetEFA();
        auto* domain = efa->GetDomain();
        ib_mr* mr = nullptr;
        int rc = ib_mr_reg(domain, data, size, IB_MR_LOCAL_READ | IB_MR_REMOTE_WRITE | IB_MR_REMOTE_READ, &mr);
        if (rc) throw std::runtime_error(fmt::format("ib_mr_reg failed: {}", strerror(-rc)));
        mrs.push_back(mr);
      }
      break;
    }
    return mrs;
  }
};

}  // namespace ib
