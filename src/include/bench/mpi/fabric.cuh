/**
 * @file fabric.cuh
 * @brief RDMA fabric peer with CUDA benchmarking support
 */
#pragma once

#include <bootstrap/mpi/fabric.h>
#include <io/progress.h>
#include <rdma/fabric/memory.h>

#include <chrono>
#include <device/common.cuh>

/**
 * @brief Result from a single benchmark run
 */
struct BenchResult {
  size_t size;     ///< Buffer size in bytes
  double time_us;  ///< Average time per iteration in microseconds
  double bw_gbps;  ///< Achieved bandwidth in Gbps
  double bus_bw;   ///< Bus bandwidth utilization ratio
};

/**
 * @brief CUDA kernel to initialize buffer with a value
 * @param data Pointer to buffer data
 * @param len Number of elements
 * @param value Value to set
 */
__global__ void InitBufferKernel(int* __restrict__ data, size_t len, int value) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) data[idx] = value;
}

/**
 * @brief Create buffer for SymmetricMemory types
 * @param c Channel vector for registration
 * @param device CUDA device ID
 * @param size Buffer size in bytes
 * @param world_size Number of ranks
 * @return Unique pointer to created buffer
 */
template <typename T>
std::unique_ptr<T> MakeBuffer(std::vector<Channel>& c, int device, size_t size, int world_size) {
  return std::make_unique<T>(c, size, world_size, device);
}

/** @brief Create buffer specialization for raw HostBuffer */
template <>
inline std::unique_ptr<HostBuffer> MakeBuffer(std::vector<Channel>& c, int device, size_t size, int) {
  return std::make_unique<HostBuffer>(c, device, size);
}

/** @brief Create buffer specialization for raw DeviceDMABuffer */
template <>
inline std::unique_ptr<DeviceDMABuffer> MakeBuffer(std::vector<Channel>& c, int device, size_t size, int) {
  return std::make_unique<DeviceDMABuffer>(c, device, size);
}

/** @brief Create buffer specialization for raw DevicePinBuffer */
template <>
inline std::unique_ptr<DevicePinBuffer> MakeBuffer(std::vector<Channel>& c, int device, size_t size, int) {
  return std::make_unique<DevicePinBuffer>(c, device, size);
}

/**
 * @brief Initialize buffer using CUDA kernel
 * @param buf Buffer to initialize
 * @param num_ints Number of integers to set
 * @param value Value to set
 * @param stream CUDA stream for async execution
 */
template <typename T>
void InitBuffer(T* buf, size_t num_ints, int value, cudaStream_t stream) {
  cudaLaunchConfig_t cfg{};
  cfg.blockDim = dim3(256, 1, 1);
  cfg.gridDim = dim3((num_ints + 255) / 256, 1, 1);
  cfg.stream = stream;
  LAUNCH_KERNEL(&cfg, InitBufferKernel, reinterpret_cast<int*>(buf->Data()), num_ints, value);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

/** @brief Initialize buffer specialization for HostBuffer (CPU loop) */
template <>
inline void InitBuffer(HostBuffer* buf, size_t num_ints, int value, cudaStream_t) {
  int* data = reinterpret_cast<int*>(buf->Data());
  for (size_t i = 0; i < num_ints; ++i) data[i] = value;
}

/** @brief Initialize buffer specialization for SymmetricHostMemory (CPU loop) */
template <>
inline void InitBuffer(SymmetricHostMemory* buf, size_t num_ints, int value, cudaStream_t) {
  int* data = reinterpret_cast<int*>(buf->Data());
  for (size_t i = 0; i < num_ints; ++i) data[i] = value;
}

/** @brief CUDA kernel for random initialization */
__global__ void RandInitKernel(int* __restrict__ data, size_t len, unsigned seed) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) data[idx] = (seed * 1103515245 + 12345 + idx) & 0x7fffffff;
}

/** @brief Random initialize GPU buffer */
template <typename T>
void RandInit(T* buf, size_t num_ints, cudaStream_t stream) {
  cudaLaunchConfig_t cfg{};
  cfg.blockDim = dim3(256, 1, 1);
  cfg.gridDim = dim3((num_ints + 255) / 256, 1, 1);
  cfg.stream = stream;
  unsigned seed = static_cast<unsigned>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  LAUNCH_KERNEL(&cfg, RandInitKernel, reinterpret_cast<int*>(buf->Data()), num_ints, seed);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

/** @brief Random initialize CPU buffer */
template <>
inline void RandInit(HostBuffer* buf, size_t num_ints, cudaStream_t) {
  int* data = reinterpret_cast<int*>(buf->Data());
  unsigned seed = static_cast<unsigned>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  for (size_t i = 0; i < num_ints; ++i) data[i] = (seed * 1103515245 + 12345 + i) & 0x7fffffff;
}

/**
 * @brief Verify buffer contents match expected value
 * @param buf Host buffer to verify
 * @param expected Expected value
 * @param rank Current MPI rank (for logging)
 * @param peer Peer rank (for logging)
 * @return Number of mismatched elements
 */
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

/** @brief Verification functor for GPU buffers (cudaMemcpy to host then verify) */
struct VerifyGPU {
  template <typename P, typename Buffers>
  void operator()(P& peer, Buffers& bufs) const {
    const auto world_size = peer.mpi.GetWorldSize();
    const auto rank = peer.mpi.GetWorldRank();
    const size_t buf_size = bufs[rank == 0 ? 1 : 0]->Size();
    const size_t num_ints = buf_size / sizeof(int);
    std::vector<int> host_buf(num_ints);
    for (int i = 0; i < world_size; ++i) {
      if (i == rank) continue;
      CUDA_CHECK(cudaMemcpy(host_buf.data(), bufs[i]->Data(), buf_size, cudaMemcpyDeviceToHost));
      if (VerifyBufferData(host_buf, i, rank, i) > 0) throw std::runtime_error("Verification failed");
    }
    SPDLOG_DEBUG("Rank {}: Verification passed", rank);
  }
};

/** @brief Verification functor for CPU buffers (direct memcpy) */
struct VerifyCPU {
  template <typename P, typename Buffers>
  void operator()(P& peer, Buffers& bufs) const {
    const auto world_size = peer.mpi.GetWorldSize();
    const auto rank = peer.mpi.GetWorldRank();
    const size_t buf_size = bufs[rank == 0 ? 1 : 0]->Size();
    const size_t num_ints = buf_size / sizeof(int);
    std::vector<int> host_buf(num_ints);
    for (int i = 0; i < world_size; ++i) {
      if (i == rank) continue;
      std::memcpy(host_buf.data(), bufs[i]->Data(), buf_size);
      if (VerifyBufferData(host_buf, i, rank, i) > 0) throw std::runtime_error("Verification failed");
    }
    SPDLOG_DEBUG("Rank {}: Verification passed", rank);
  }
};

/** @brief No-op verification functor */
struct NoVerify {
  template <typename P, typename Buffers>
  void operator()(P&, Buffers&) const {}
};

/**
 * @brief RDMA peer with CUDA stream and benchmarking support
 *
 * Extends Peer with buffer allocation, warmup, and timed benchmarking.
 * @tparam T Buffer type (e.g., SymmetricDMAMemory, HostBuffer)
 * @tparam F Communication functor (e.g., All2all)
 * @tparam V Verification functor (VerifyGPU or VerifyCPU)
 */
class FabricBench : public Peer {
 public:
  cudaStream_t stream;

  FabricBench() : Peer() {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaStreamCreate(&stream));
  }

  ~FabricBench() { CUDA_CHECK(cudaStreamDestroy(stream)); }

  /**
   * @brief Allocate buffers for all peers with initial value
   * @param size Buffer size in bytes
   * @param init_value Initial value (-1 uses rank as value)
   * @return Vector of allocated buffers
   */
  template <typename T>
  Buffers<T> Alloc(size_t size, int init_value) {
    const auto world_size = mpi.GetWorldSize();
    const auto rank = mpi.GetWorldRank();
    Buffers<T> buffers(world_size);
    const size_t num_ints = size / sizeof(int);
    int value = (init_value == -1) ? rank : init_value;
    for (int i = 0; i < world_size; ++i) {
      if (i == rank) continue;
      buffers[i] = MakeBuffer<T>(channels[i], device, size, world_size);
      InitBuffer(buffers[i].get(), num_ints, value, stream);
    }
    return buffers;
  }

  /**
   * @brief Allocate send/recv buffer pair
   * @param size Buffer size in bytes
   * @return Pair of (send buffers initialized with rank, recv buffers initialized with 0)
   */
  template <typename T>
  std::pair<Buffers<T>, Buffers<T>> AllocPair(size_t size) {
    return {Alloc<T>(size, mpi.GetWorldRank()), Alloc<T>(size, -1)};
  }

  /**
   * @brief Run warmup iterations with verification
   * @param a First buffer set
   * @param b Second buffer set
   * @param func Communication functor
   * @param verify Verification functor
   * @param iters Number of warmup iterations
   */
  template <typename T, typename F, typename V>
  void Warmup(Buffers<T>& a, Buffers<T>& b, F&& func, V&& verify, int iters = 8) {
    for (int i = 0; i < iters; ++i) {
      func(*this, a, b);
      // run barrier to ensure all bench done
      MPI_Barrier(MPI_COMM_WORLD);
      verify(*this, b);
    }
  }

  /**
   * @brief Run timed benchmark and return results
   * @param a First buffer set
   * @param b Second buffer set
   * @param func Communication functor
   * @param verify Verification functor
   * @param iters Number of benchmark iterations
   * @return Benchmark results
   */
  template <typename T, typename F, typename V>
  BenchResult Bench(Buffers<T>& a, Buffers<T>& b, F&& func, V&& verify, int iters) {
    const auto rank = mpi.GetWorldRank();
    const size_t buf_size = a[rank == 0 ? 1 : 0]->Size();
    const size_t bw = GetBandwidth(0);
    Progress progress(iters, bw);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
      func(*this, a, b);
      if (rank == 0 && i % 10 == 0) progress.Print(std::chrono::high_resolution_clock::now(), buf_size, i + 1);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us = elapsed_us / iters;
    double bw_gbps = (buf_size * 8.0) / (avg_us * 1000.0);
    double bus_bw = (bw_gbps * 1e9 / 8.0) / bw;
    verify(*this, b);
    return {buf_size, avg_us, bw_gbps, bus_bw};
  }

  /**
   * @brief Print complete benchmark summary
   * @tparam N Number of results per row
   * @param title Benchmark title/name
   * @param nranks Number of MPI ranks
   * @param warmup Number of warmup iterations
   * @param iters Number of benchmark iterations
   * @param link_bw Theoretical link bandwidth in Gbps
   * @param pattern Description of communication pattern
   * @param columns Column names for results
   * @param results Vector of result arrays to print
   */
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
    for (const auto& col : columns) printf(" %14s %10s", col.c_str(), "BusBW(%)");
    printf("\n");
    for (const auto& r : results) {
      printf("%12zu %12zu", r[0].size, r[0].size / sizeof(float));
      for (const auto& v : r) printf(" %14.2f %10.1f", v.bw_gbps, v.bus_bw);
      printf("\n");
    }
    printf("#\n# Benchmark complete.\n");
  }
};
