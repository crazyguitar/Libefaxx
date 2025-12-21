/**
 * @file main.cu
 * @brief EFA Proxy Write Benchmark - GPU-initiated RDMA write
 *
 * Measures RDMA write bandwidth with GPU kernel pushing requests to queue.
 * Pattern: rank0 -> rank_k (k=1..N-1), results averaged across all pairs.
 */
#include <bench/arguments.h>
#include <io/awaiter.h>
#include <io/runner.h>
#include <rdma/fabric/memory.h>
#include <rdma/fabric/selector.h>

#include <bench/mpi/fabric.cuh>

/** @brief Device function: rank 0 writes data and pushes request */
__device__ __forceinline__ void DeviceWrite(DeviceContext ctx, int target, size_t len, int* data, uint64_t imm) {
  for (size_t idx = threadIdx.x; idx < len; idx += blockDim.x) {
    data[idx] = target + static_cast<int>(idx);
  }
  __threadfence_system();

  if (threadIdx.x == 0) {
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
__device__ __forceinline__ void DeviceWait(DeviceContext ctx, int expected_ops) {
  if (threadIdx.x == 0) {
    while (*ctx.completed < expected_ops) __threadfence_system();
  }
  __syncthreads();
}

/** @brief Device function: verify received data */
__device__ __forceinline__ bool DeviceVerify(int expected, size_t len, int* data) {
  bool ok = true;
  for (size_t idx = threadIdx.x; idx < len; idx += blockDim.x) {
    if (data[idx] != expected + static_cast<int>(idx)) ok = false;
  }
  return ok;
}

/** @brief Kernel for rank 0: write to all targets */
__global__ void ProxyWriteKernel(DeviceContext ctx, int world_size, size_t len, int* data, uint64_t imm, int iters) {
  for (int i = 0; i < iters; ++i) {
    for (int t = 1; t < world_size; ++t) {
      DeviceWrite(ctx, t, len, data, imm);
    }
  }
}

/** @brief Kernel for rank 1..n: wait and verify */
__global__ void ProxyWaitKernel(DeviceContext ctx, int rank, size_t len, int* data, int iters, int* result) {
  DeviceWait(ctx, iters);
  bool ok = DeviceVerify(rank, len, data);
  if (threadIdx.x == 0) *result = ok ? 1 : 0;
}

/**
 * @brief Rank 0: launch write kernel and run proxy loop
 */
template <typename Peer>
struct ProxyWrite {
  int iters;

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& write) {
    if (peer.mpi.GetWorldRank() != 0) return;

    int world_size = peer.mpi.GetWorldSize();
    auto ctx = write[1]->GetContext();
    int* data = reinterpret_cast<int*>(write[1]->Data());
    size_t size = write[1]->Size();
    size_t len = size / sizeof(int);
    auto& prop = peer.loc.GetGPUAffinity()[peer.device].prop;

    // Single block with max threads to fit kernel on one SM for efficient __syncthreads()
    cudaLaunchConfig_t cfg{};
    cfg.blockDim = dim3(prop.maxThreadsPerBlock, 1, 1);
    cfg.gridDim = dim3(1, 1, 1);
    cfg.stream = peer.stream;

    LAUNCH_KERNEL(&cfg, ProxyWriteKernel, ctx, world_size, len, data, 1ULL, iters);

    int total_ops = iters * (world_size - 1);
    size_t total_bw = peer.GetTotalBandwidth();
    Progress progress(total_ops, total_bw);

    ::Run([&]() -> Coro<> {
      int completed = 0;
      DeviceRequest req;
      while (completed < total_ops) {
        if (ctx.queue->Pop(req)) {
          co_await write[req.rank]->Writeall(static_cast<int>(req.rank), req.imm);
          write[req.rank]->Complete();
          ++completed;
          if (completed % 10 == 0) progress.Print(std::chrono::high_resolution_clock::now(), size, completed);
        }
        co_await YieldAwaiter{};
      }
      for (auto& efa : peer.efas) IO::Get().Quit<FabricSelector>(efa);
    }());

    CUDA_CHECK(cudaStreamSynchronize(peer.stream));
  }
};

/**
 * @brief Rank 1..n: launch wait kernel and receive immdata
 */
template <typename Peer>
struct ProxyRead {
  int iters;

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& read) {
    if (peer.mpi.GetWorldRank() == 0) return;

    auto ctx = read[0]->GetContext();
    int* data = reinterpret_cast<int*>(read[0]->Data());
    size_t len = read[0]->Size() / sizeof(int);
    auto& prop = peer.loc.GetGPUAffinity()[peer.device].prop;

    int* result;
    CUDA_CHECK(cudaMallocManaged(&result, sizeof(int)));
    *result = 1;

    // Single block with max threads to fit kernel on one SM for efficient __syncthreads()
    cudaLaunchConfig_t cfg{};
    cfg.blockDim = dim3(prop.maxThreadsPerBlock, 1, 1);
    cfg.gridDim = dim3(1, 1, 1);
    cfg.stream = peer.stream;

    LAUNCH_KERNEL(&cfg, ProxyWaitKernel, ctx, peer.mpi.GetWorldRank(), len, data, iters, result);

    ::Run([&]() -> Coro<> {
      for (int i = 0; i < iters; ++i) {
        co_await read[0]->WaitallImmdata(1);
        read[0]->Complete();
      }
      for (auto& efa : peer.efas) IO::Get().Quit<FabricSelector>(efa);
    }());

    CUDA_CHECK(cudaStreamSynchronize(peer.stream));
    if (*result != 1) throw std::runtime_error(fmt::format("Verification failed on rank {}", peer.mpi.GetWorldRank()));
    CUDA_CHECK(cudaFree(result));
  }
};

/**
 * @brief Combined proxy benchmark
 */
template <typename Peer>
struct ProxyBench {
  int iters;

  template <typename T>
  BenchResult Run(Peer& peer, typename Peer::template Buffers<T>& write, typename Peer::template Buffers<T>& read) {
    int rank = peer.mpi.GetWorldRank();
    int world_size = peer.mpi.GetWorldSize();
    size_t size = (rank == 0) ? write[1]->Size() : read[0]->Size();

    for (auto& efa : peer.efas) IO::Get().Join<FabricSelector>(efa);

    auto start = std::chrono::high_resolution_clock::now();
    ProxyWrite<Peer>{iters}.template operator()<T>(peer, write);
    ProxyRead<Peer>{iters}.template operator()<T>(peer, read);
    auto end = std::chrono::high_resolution_clock::now();

    double time_us = std::chrono::duration<double, std::micro>(end - start).count();
    double bw_gbps = (rank == 0) ? (static_cast<double>(size) * iters * (world_size - 1) * 8.0) / (time_us * 1000.0) : 0;

    return {size, time_us / iters, bw_gbps, 0.0};
  }
};

/**
 * @brief Test runner for proxy benchmark
 */
template <typename BufType>
struct Test {
  static BenchResult Run(size_t size, const Options& opts, double single_bw, double total_bw) {
    FabricBench peer;
    peer.Exchange();
    peer.Connect();

    auto [write, read] = peer.AllocPair<BufType>(size);
    peer.Handshake(write, read);

    auto r = ProxyBench<FabricBench>{opts.repeat}.Run(peer, write, read);
    MPI_Barrier(MPI_COMM_WORLD);

    double bus_bw = (total_bw > 0) ? (r.bw_gbps / total_bw) * 100.0 : 0;
    return {size, r.time_us, r.bw_gbps, bus_bw};
  }
};

template <typename... Tests>
std::array<BenchResult, sizeof...(Tests)> RunTests(size_t size, const Options& opts, double single_bw, double total_bw) {
  std::array<BenchResult, sizeof...(Tests)> results;
  size_t i = 0;
  ((results[i++] = Tests::Run(size, opts, single_bw, total_bw), MPI_Barrier(MPI_COMM_WORLD)), ...);
  return results;
}

using ProxyDMA = Test<SymmetricDMAMemory>;

int main(int argc, char* argv[]) {
  try {
    auto opts = parse_args(argc, argv);
    auto sizes = generate_sizes(opts);
    int rank = MPI::Get().GetWorldRank();
    int nranks = MPI::Get().GetWorldSize();

    FabricBench peer;
    double single_bw = peer.GetBandwidth(0) / 1e9;
    double total_bw = peer.GetTotalBandwidth() / 1e9;

    std::vector<std::array<BenchResult, 1>> results;
    for (auto size : sizes) {
      results.push_back(RunTests<ProxyDMA>(size, opts, single_bw, total_bw));
    }

    if (rank == 0) {
      FabricBench::Print(
          "EFA Proxy Write Benchmark", nranks, opts.warmup, opts.repeat, total_bw, "GPU kernel -> Queue -> RDMA write, rank0 -> rank_k", {"ProxyDMA"},
          results
      );
    }
    return 0;
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Fatal error: {}", e.what());
    return 1;
  }
}
