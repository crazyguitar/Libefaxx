/**
 * @file main.cu
 * @brief EFA Proxy Write Benchmark - GPU-initiated RDMA write
 *
 * Measures RDMA write bandwidth with GPU kernel pushing requests to queue.
 * Pattern: rank0 -> rank_k (k=1..N-1), results averaged across all pairs.
 */
#include <bench/arguments.h>
#include <io/runner.h>
#include <rdma/fabric/memory.h>
#include <rdma/fabric/selector.h>

#include <bench/mpi/fabric.cuh>

/** @brief Device function: rank 0 writes data and pushes request */
__device__ __forceinline__ void ProxyWrite(DeviceContext ctx, int target, size_t len, int* data, uint64_t imm) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) data[idx] = target + idx;
  __syncthreads();

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    __threadfence_system();
    DeviceRequest req{
        .type = static_cast<uint64_t>(DeviceRequestType::kPut),
        .rank = static_cast<uint64_t>(target),
        .size = len * sizeof(int),
        .addr = reinterpret_cast<uint64_t>(data),
        .imm = imm
    };
    ctx.queue->Push(req);
    atomicAdd(reinterpret_cast<unsigned long long*>(ctx.posted), 1ULL);
    Fence();
    Quiet(ctx.posted, ctx.completed);
  }
  __syncthreads();
}

/** @brief Device function: rank 1..n waits for data from rank 0 */
__device__ __forceinline__ void ProxyWait(DeviceContext ctx) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    Quiet(ctx.posted, ctx.completed);
  }
  __syncthreads();
}

/** @brief Device function: verify received data */
__device__ __forceinline__ bool ProxyVerify(int expected, size_t len, int* data) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  bool ok = true;
  if (idx < len) {
    if (data[idx] != expected + idx) ok = false;
  }
  return __syncthreads_and(ok);
}

/** @brief Kernel for rank 0: write to targets */
__global__ void ProxyWriteKernel(DeviceContext ctx, int target, size_t len, int* data, uint64_t imm, int iters) {
  for (int i = 0; i < iters; ++i) {
    ProxyWrite(ctx, target, len, data, imm);
  }
}

/** @brief Kernel for rank 1..n: wait and verify */
__global__ void ProxyWaitKernel(DeviceContext ctx, int sender, size_t len, int* data, int iters, int* result) {
  for (int i = 0; i < iters; ++i) ProxyWait(ctx);
  bool ok = ProxyVerify(sender, len, data);
  if (threadIdx.x == 0 && blockIdx.x == 0) *result = ok ? 1 : 0;
}

/**
 * @brief Proxy benchmark - GPU kernel initiates RDMA writes
 */
template <typename Peer>
struct ProxyBench {
  int target;
  int iters;

  template <typename T>
  BenchResult Write(Peer& peer, typename Peer::template Buffers<T>& write) {
    auto ctx = write[target]->GetContext();
    int* data = reinterpret_cast<int*>(write[target]->Data());
    size_t size = write[target]->Size();
    size_t len = size / sizeof(int);
    auto& prop = peer.loc.GetGPUAffinity()[peer.device].prop;

    cudaLaunchConfig_t cfg{};
    cfg.blockDim = dim3(prop.maxThreadsPerBlock, 1, 1);
    cfg.gridDim = dim3((len + cfg.blockDim.x - 1) / cfg.blockDim.x, 1, 1);
    cfg.stream = peer.stream;

    auto start = std::chrono::high_resolution_clock::now();
    LAUNCH_KERNEL(&cfg, ProxyWriteKernel, ctx, target, len, data, 1ULL, iters);
    for (auto& efa : peer.efas) IO::Get().Join<FabricSelector>(efa);

    ::Run([&]() -> Coro<> {
      int completed = 0;
      DeviceRequest req;
      while (completed < iters) {
        if (ctx.queue->Pop(req)) {
          co_await write[target]->Writeall(static_cast<int>(req.rank), req.imm);
          write[target]->Complete();
          fmt::print("\r  Progress: {}/{}", ++completed, iters);
        }
        co_await std::suspend_always{};
      }
      fmt::print("\n");
      for (auto& efa : peer.efas) IO::Get().Quit<FabricSelector>(efa);
    }());

    CUDA_CHECK(cudaStreamSynchronize(peer.stream));
    auto end = std::chrono::high_resolution_clock::now();

    double time_us = std::chrono::duration<double, std::micro>(end - start).count();
    double bw_gbps = (static_cast<double>(size) * iters * 8.0) / (time_us * 1000.0);

    return {size, time_us / iters, bw_gbps, 0.0};
  }

  template <typename T>
  void Read(Peer& peer, typename Peer::template Buffers<T>& read) {
    auto ctx = read[0]->GetContext();
    int* data = reinterpret_cast<int*>(read[0]->Data());
    size_t len = read[0]->Size() / sizeof(int);
    auto& prop = peer.loc.GetGPUAffinity()[peer.device].prop;

    int* result;
    CUDA_CHECK(cudaMallocManaged(&result, sizeof(int)));
    *result = 0;

    cudaLaunchConfig_t cfg{};
    cfg.blockDim = dim3(prop.maxThreadsPerBlock, 1, 1);
    cfg.gridDim = dim3((len + cfg.blockDim.x - 1) / cfg.blockDim.x, 1, 1);
    cfg.stream = peer.stream;

    LAUNCH_KERNEL(&cfg, ProxyWaitKernel, ctx, 0, len, data, iters, result);

    for (auto& efa : peer.efas) IO::Get().Join<FabricSelector>(efa);
    ::Run([&]() -> Coro<> {
      for (int i = 0; i < iters; ++i) {
        co_await read[0]->WaitallImmdata(1);
        read[0]->Complete();
      }
      for (auto& efa : peer.efas) IO::Get().Quit<FabricSelector>(efa);
    }());

    CUDA_CHECK(cudaStreamSynchronize(peer.stream));
    if (*result != 1) SPDLOG_ERROR("Verification failed on rank {}", peer.mpi.GetWorldRank());
    CUDA_CHECK(cudaFree(result));
  }

  template <typename T>
  BenchResult Run(Peer& peer, typename Peer::template Buffers<T>& write, typename Peer::template Buffers<T>& read) {
    int rank = peer.mpi.GetWorldRank();
    if (rank == 0) return Write(peer, write);
    if (rank == target) Read(peer, read);
    return {write[target]->Size(), 0, 0, 0};
  }
};

/**
 * @brief Test runner for proxy benchmark
 */
template <typename BufType>
struct Test {
  static BenchResult Run(size_t size, const Options& opts, double total_bw) {
    FabricBench peer;
    peer.Exchange();
    peer.Connect();
    int rank = peer.mpi.GetWorldRank();
    int world = peer.mpi.GetWorldSize();

    auto [write, read] = peer.AllocPair<BufType>(size);
    peer.Handshake(write, read);

    double sum_bw = 0;
    double sum_time = 0;
    for (int t = 1; t < world; ++t) {
      auto r = ProxyBench<FabricBench>{t, opts.repeat}.Run(peer, write, read);
      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == 0) {
        sum_bw += r.bw_gbps;
        sum_time += r.time_us;
      }
    }
    int npairs = world - 1;
    double avg_bw = sum_bw / npairs;
    double bus_bw = (total_bw > 0) ? (avg_bw / total_bw) * 100.0 : 0;
    return {size, sum_time / npairs, avg_bw, bus_bw};
  }
};

using ProxyDMA = Test<SymmetricDMAMemory>;

int main(int argc, char* argv[]) {
  try {
    auto opts = parse_args(argc, argv);
    auto sizes = generate_sizes(opts);
    int rank = MPI::Get().GetWorldRank();
    int nranks = MPI::Get().GetWorldSize();

    FabricBench peer;
    double total_bw = peer.GetTotalBandwidth() / 1e9;

    std::vector<BenchResult> results;
    for (auto size : sizes) {
      results.push_back(ProxyDMA::Run(size, opts, total_bw));
      MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
      fmt::print("\n");
      fmt::print("EFA Proxy Write Benchmark\n");
      fmt::print("  Ranks: {}, Iterations: {}\n", nranks, opts.repeat);
      fmt::print("  Total BW: {:.2f} Gbps\n\n", total_bw);
      fmt::print("{:>12} {:>12} {:>12} {:>10}\n", "Size", "Time(us)", "BW(Gbps)", "BusBW(%)");
      for (auto& r : results) {
        fmt::print("{:>12} {:>12.2f} {:>12.2f} {:>10.1f}\n", r.size, r.time_us, r.bw_gbps, r.bus_bw);
      }
    }
    return 0;
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Fatal error: {}", e.what());
    return 1;
  }
}
