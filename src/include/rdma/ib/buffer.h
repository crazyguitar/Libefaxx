/**
 * @file buffer.h
 * @brief Buffer management for RDMA operations using ibverbs
 */
#pragma once

#include <cuda.h>
#include <rdma/buffer.h>
#include <rdma/ib/channel.h>
#include <rdma/ib/selector.h>
#include <unistd.h>

#include <device/common.cuh>
#include <vector>

namespace ib {

/**
 * @brief Backend traits for ibverbs buffer
 */
struct BufferBackend {
  using EFA = ib::EFA;
  using Channel = ib::Channel;
  using MR = ib_mr;
  using RmaIov = ib_rma_iov;
  using KeyType = uint32_t;
  using ImmdataAwaiter = rdma::ImmdataAwaiter<IBSelector, ImmContext>;

  static constexpr bool kSupportsSendRecv = false;
  static constexpr const char* kWriteOp = "ib_writemsg";

  static RmaIov MakeRmaIov(void* data, size_t size, MR* mr) noexcept { return {reinterpret_cast<uint64_t>(data), size, ib_mr_rkey(mr)}; }

  static std::string FormatError(ssize_t rc) { return strerror(-rc); }
};

/**
 * @brief Buffer class for RDMA channel operations using ibverbs
 */
class Buffer : public rdma::Buffer<BufferBackend> {
  using Base = rdma::Buffer<BufferBackend>;

 protected:
  Buffer(std::vector<EFA>& efas, std::vector<std::vector<Channel>>& channels, size_t size) : Base(efas, channels, size) {}
};

/**
 * @brief GPU device memory buffer for RDMA operations using DMABUF and ibverbs
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
      mrs_ = Register(efas_, data_, size_, dmabuf_fd_);
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
  static std::vector<ib_mr*> Register(std::vector<EFA>& efas, void* __restrict__ data, size_t size, int dmabuf_fd) {
    std::vector<ib_mr*> mrs;
    mrs.reserve(efas.size());
    for (auto& efa : efas) {
      auto* domain = efa.GetDomain();
      ibv_mr* ibv_mr_ptr = ibv_reg_dmabuf_mr(
          domain->pd, 0, size, reinterpret_cast<uint64_t>(data), dmabuf_fd, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ
      );
      if (!ibv_mr_ptr) throw std::runtime_error(fmt::format("ibv_reg_dmabuf_mr failed: {}", strerror(errno)));
      mrs.push_back(new ib_mr{ibv_mr_ptr, domain});
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
    for (auto* mr : mrs_) ib_mr_close(mr);
    mrs_.clear();
    if (raw_) {
      free(raw_);
      raw_ = nullptr;
    }
  }

 protected:
  static std::vector<ib_mr*> Register(std::vector<EFA>& efas, void* __restrict__ data, size_t size) {
    std::vector<ib_mr*> mrs;
    mrs.reserve(efas.size());
    for (auto& efa : efas) {
      auto* domain = efa.GetDomain();
      ib_mr* mr = nullptr;
      int rc = ib_mr_reg(domain, data, size, IB_MR_LOCAL_READ | IB_MR_REMOTE_WRITE | IB_MR_REMOTE_READ, &mr);
      if (rc) throw std::runtime_error(fmt::format("ib_mr_reg failed: {}", BufferBackend::FormatError(rc)));
      mrs.push_back(mr);
    }
    return mrs;
  }
};

}  // namespace ib
