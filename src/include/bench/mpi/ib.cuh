/**
 * @file ib.cuh
 * @brief RDMA fabric peer with CUDA benchmarking support using ibverbs
 */
#pragma once

#include <bootstrap/mpi/ib.h>
#include <io/progress.h>
#include <rdma/ib/memory.h>

#include <bench/mpi/fabric.cuh>  // Reuse BenchResult, InitBuffer, VerifyBufferData
#include <chrono>
#include <device/common.cuh>

/**
 * @brief CUDA kernel to initialize buffer with a value
 */
__global__ void IBInitBufferKernel(int* __restrict__ data, size_t len, int value) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) data[idx] = value;
}

/**
 * @brief Create buffer for ib::SymmetricMemory types
 */
template <typename T>
std::unique_ptr<T> IBMakeBuffer(std::vector<ib::Channel>& c, int device, size_t size, int world_size) {
  return std::make_unique<T>(c, size, world_size, device);
}

/**
 * @brief Initialize buffer using CUDA kernel
 */
template <typename T>
void IBInitBuffer(T* buf, size_t num_ints, int value, cudaStream_t stream) {
  cudaLaunchConfig_t cfg{};
  cfg.blockDim = dim3(256, 1, 1);
  cfg.gridDim = dim3((num_ints + 255) / 256, 1, 1);
  cfg.stream = stream;
  LAUNCH_KERNEL(&cfg, IBInitBufferKernel, reinterpret_cast<int*>(buf->Data()), num_ints, value);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

/** @brief Initialize buffer specialization for ib::SymmetricHostMemory (CPU loop) */
template <>
inline void IBInitBuffer(ib::SymmetricHostMemory* buf, size_t num_ints, int value, cudaStream_t) {
  int* data = reinterpret_cast<int*>(buf->Data());
  for (size_t i = 0; i < num_ints; ++i) data[i] = value;
}

/**
 * @brief RDMA peer with CUDA stream and benchmarking support using ibverbs
 */
class IBBench : public ib::Peer {
 public:
  cudaStream_t stream;

  IBBench() : ib::Peer() {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaStreamCreate(&stream));
  }

  ~IBBench() { CUDA_CHECK(cudaStreamDestroy(stream)); }

  template <typename T>
  Buffers<T> Alloc(size_t size, int init_value) {
    const auto world_size = mpi.GetWorldSize();
    const auto rank = mpi.GetWorldRank();
    Buffers<T> buffers(world_size);
    const size_t num_ints = size / sizeof(int);
    int value = (init_value == -1) ? rank : init_value;
    for (int i = 0; i < world_size; ++i) {
      if (i == rank) continue;
      buffers[i] = IBMakeBuffer<T>(channels[i], device, size, world_size);
      IBInitBuffer(buffers[i].get(), num_ints, value, stream);
    }
    return buffers;
  }

  template <typename T>
  std::pair<Buffers<T>, Buffers<T>> AllocPair(size_t size) {
    return {Alloc<T>(size, mpi.GetWorldRank()), Alloc<T>(size, -1)};
  }

  template <typename T, typename F, typename V>
  void Warmup(Buffers<T>& a, Buffers<T>& b, F&& func, V&& verify, int iters = 8) {
    for (int i = 0; i < iters; ++i) {
      func(*this, a, b);
      MPI_Barrier(MPI_COMM_WORLD);
      verify(*this, b);
    }
  }

  template <typename T, typename F, typename V>
  BenchResult
  Bench(std::string_view name, Buffers<T>& a, Buffers<T>& b, F&& func, V&& verify, int iters, size_t progress_bytes = 0, size_t progress_bw = 0) {
    const auto rank = mpi.GetWorldRank();
    const auto world_size = mpi.GetWorldSize();
    size_t buf_size = 0;
    for (int i = 0; i < world_size && buf_size == 0; ++i) {
      if (a[i]) buf_size = a[i]->Size();
    }
    const size_t bytes = progress_bytes > 0 ? progress_bytes : buf_size;
    const size_t bw = progress_bw > 0 ? progress_bw : GetBandwidth(0);
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
    printf("#\n# %s (ibverbs)\n#\n", title);
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
