/**
 * @file bench.cuh
 * @brief Unified RDMA benchmark framework for Fabric and IB backends
 */
#pragma once

#include <io/progress.h>
#include <mpi.h>

#include <chrono>
#include <device/common.cuh>

/**
 * @brief Result from a single benchmark run
 */
struct BenchResult {
  size_t size;
  double time_us;
  double bw_gbps;
  double bus_bw;
};

__global__ void InitBufferKernel(int* __restrict__ data, size_t len, int value) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) data[idx] = value;
}

__global__ void RandInitKernel(int* __restrict__ data, size_t len, unsigned seed) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) data[idx] = (seed * 1103515245 + 12345 + idx) & 0x7fffffff;
}

inline size_t VerifyBufferData(const std::vector<int>& buf, int expected, int rank, int peer) noexcept {
  size_t errors = 0;
  for (size_t j = 0; j < buf.size(); ++j) {
    if (buf[j] != expected) {
      if (errors == 0) SPDLOG_ERROR("Rank {}: Peer {} mismatch at [{}]: expected {}, got {}", rank, peer, j, expected, buf[j]);
      errors++;
    }
  }
  return errors;
}

struct VerifyGPU {
  template <typename P, typename Buffers>
  void operator()(P& peer, Buffers& bufs) const {
    const auto world_size = peer.mpi.GetWorldSize();
    const auto rank = peer.mpi.GetWorldRank();
    const size_t buf_size = bufs[rank == 0 ? 1 : 0]->Size();
    std::vector<int> host_buf(buf_size / sizeof(int));
    for (int i = 0; i < world_size; ++i) {
      if (i == rank) continue;
      CUDA_CHECK(cudaMemcpy(host_buf.data(), bufs[i]->Data(), buf_size, cudaMemcpyDeviceToHost));
      if (VerifyBufferData(host_buf, i, rank, i) > 0) throw std::runtime_error("Verification failed");
    }
  }
};

struct VerifyCPU {
  template <typename P, typename Buffers>
  void operator()(P& peer, Buffers& bufs) const {
    const auto world_size = peer.mpi.GetWorldSize();
    const auto rank = peer.mpi.GetWorldRank();
    const size_t buf_size = bufs[rank == 0 ? 1 : 0]->Size();
    std::vector<int> host_buf(buf_size / sizeof(int));
    for (int i = 0; i < world_size; ++i) {
      if (i == rank) continue;
      std::memcpy(host_buf.data(), bufs[i]->Data(), buf_size);
      if (VerifyBufferData(host_buf, i, rank, i) > 0) throw std::runtime_error("Verification failed");
    }
  }
};

struct NoVerify {
  template <typename P, typename Buffers>
  void operator()(P&, Buffers&) const {}
};

/**
 * @brief Traits for buffer initialization
 */
template <typename T>
struct BufferTraits {
  static constexpr bool is_host = false;
};

/**
 * @brief Initialize GPU buffer
 */
template <typename T>
void InitBuffer(T* buf, size_t num_ints, int value, cudaStream_t stream) {
  if constexpr (BufferTraits<T>::is_host) {
    int* data = reinterpret_cast<int*>(buf->Data());
    for (size_t i = 0; i < num_ints; ++i) data[i] = value;
  } else {
    cudaLaunchConfig_t cfg{};
    cfg.blockDim = dim3(256, 1, 1);
    cfg.gridDim = dim3((num_ints + 255) / 256, 1, 1);
    cfg.stream = stream;
    LAUNCH_KERNEL(&cfg, InitBufferKernel, reinterpret_cast<int*>(buf->Data()), num_ints, value);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
}

template <typename T>
void RandInit(T* buf, size_t num_ints, cudaStream_t stream) {
  unsigned seed = static_cast<unsigned>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  if constexpr (BufferTraits<T>::is_host) {
    int* data = reinterpret_cast<int*>(buf->Data());
    for (size_t i = 0; i < num_ints; ++i) data[i] = (seed * 1103515245 + 12345 + i) & 0x7fffffff;
  } else {
    cudaLaunchConfig_t cfg{};
    cfg.blockDim = dim3(256, 1, 1);
    cfg.gridDim = dim3((num_ints + 255) / 256, 1, 1);
    cfg.stream = stream;
    LAUNCH_KERNEL(&cfg, RandInitKernel, reinterpret_cast<int*>(buf->Data()), num_ints, seed);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
}

/**
 * @brief Unified benchmark class template
 * @tparam PeerBase The base peer class (fi::Peer or ib::Peer)
 * @tparam Traits Backend-specific traits for buffer creation
 */
template <typename PeerBase, typename Traits>
class BenchBase : public PeerBase {
 public:
  cudaStream_t stream;

  BenchBase() : PeerBase() {
    CUDA_CHECK(cudaSetDevice(this->device));
    CUDA_CHECK(cudaStreamCreate(&stream));
  }

  ~BenchBase() { CUDA_CHECK(cudaStreamDestroy(stream)); }

  template <typename T>
  typename PeerBase::template Buffers<T> Alloc(size_t size, int init_value) {
    const auto world_size = this->mpi.GetWorldSize();
    const auto rank = this->mpi.GetWorldRank();
    typename PeerBase::template Buffers<T> buffers(world_size);
    const size_t num_ints = size / sizeof(int);
    int value = (init_value == -1) ? rank : init_value;
    for (int i = 0; i < world_size; ++i) {
      if (i == rank) continue;
      buffers[i] = Traits::template MakeBuffer<T>(this->channels, i, this->device, size, world_size);
      InitBuffer(buffers[i].get(), num_ints, value, stream);
    }
    return buffers;
  }

  template <typename T>
  typename PeerBase::template Buffers<T> AllocIPC(size_t size, int init_value = 0) {
    const auto world_size = this->mpi.GetWorldSize();
    const auto rank = this->mpi.GetWorldRank();
    typename PeerBase::template Buffers<T> buffers(world_size);
    buffers[rank] = Traits::template MakeBuffer<T>(this->channels, rank, this->device, size, world_size);
    const size_t num_ints = size / sizeof(int);
    int value = (init_value == -1) ? rank : init_value;
    InitBuffer(buffers[rank].get(), num_ints, value, stream);
    return buffers;
  }

  template <typename T>
  std::pair<typename PeerBase::template Buffers<T>, typename PeerBase::template Buffers<T>> AllocPair(size_t size) {
    return {Alloc<T>(size, this->mpi.GetWorldRank()), Alloc<T>(size, -1)};
  }

  template <typename T, typename F, typename V>
  void Warmup(typename PeerBase::template Buffers<T>& a, typename PeerBase::template Buffers<T>& b, F&& func, V&& verify, int iters = 8) {
    for (int i = 0; i < iters; ++i) {
      func(*this, a, b);
      MPI_Barrier(MPI_COMM_WORLD);
      verify(*this, b);
    }
  }

  template <typename T, typename F, typename V>
  BenchResult Bench(
      std::string_view name,
      typename PeerBase::template Buffers<T>& a,
      typename PeerBase::template Buffers<T>& b,
      F&& func,
      V&& verify,
      int iters,
      size_t progress_bytes = 0,
      size_t progress_bw = 0
  ) {
    const auto rank = this->mpi.GetWorldRank();
    const auto world_size = this->mpi.GetWorldSize();
    size_t buf_size = 0;
    for (int i = 0; i < world_size && buf_size == 0; ++i) {
      if (a[i]) buf_size = a[i]->Size();
    }
    const size_t bytes = progress_bytes > 0 ? progress_bytes : buf_size;
    const size_t bw = progress_bw > 0 ? progress_bw : this->GetBandwidth(0);
    Progress progress(iters, bw, name);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
      func(*this, a, b);
      if (rank == 0 && i % Progress::kPrintFreq == 0) progress.Print(std::chrono::high_resolution_clock::now(), bytes, i + 1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us = elapsed_us / iters;
    double bw_gbps = (buf_size * 8.0) / (avg_us * 1000.0);
    double bus_bw = (bw_gbps * 1e9 / 8.0) / bw;
    verify(*this, b);
    return {buf_size, avg_us, bw_gbps, bus_bw};
  }

  template <size_t N>
  static void Print(
      const char* title,
      int nranks,
      int warmup,
      int iters,
      double link_bw,
      const char* pattern,
      const std::vector<std::string>& columns,
      const std::vector<std::array<BenchResult, N>>& results
  ) {
    printf("#\n# %s\n#\n", title);
    printf("# nranks: %d\n", nranks);
    printf("# warmup iters: %d\n", warmup);
    printf("# bench iters: %d\n", iters);
    printf("# link bandwidth: %.0f Gbps\n#\n", link_bw);
    if (pattern) printf("# Pattern: %s\n#\n", pattern);
    printf("# BusBW: Percentage of theoretical link bandwidth achieved\n#\n");
    printf("%12s %12s", "size", "count");
    for (const auto& col : columns) printf(" %14s %10s %10s", col.c_str(), "BusBW(%)", "Lat(us)");
    printf("\n");
    for (const auto& r : results) {
      printf("%12zu %12zu", r[0].size, r[0].size / sizeof(float));
      for (const auto& v : r) printf(" %14.2f %10.1f %10.2f", v.bw_gbps, v.bus_bw, v.time_us);
      printf("\n");
    }
    printf("#\n# Benchmark complete.\n");
  }
};
