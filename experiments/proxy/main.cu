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

/** @brief CUDA kernel to push write request to queue */
__global__ void ProxyKernel(Queue<DeviceRequest>* queue, int rank, size_t size, void* addr, uint64_t imm) {
  DeviceRequest req{
      .type = static_cast<uint64_t>(DeviceRequestType::kPut),
      .rank = static_cast<uint64_t>(rank),
      .size = size,
      .addr = reinterpret_cast<uint64_t>(addr),
      .imm = imm
  };
  queue->Push(req);
}

/**
 * @brief Proxy write functor (single channel) - GPU kernel initiates write
 */
template <typename Peer>
struct ProxyWrite {
  int target;
  int channel;

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& write, typename Peer::template Buffers<T>& read) {
    int rank = peer.mpi.GetWorldRank();

    if (rank == 0) {
      auto* queue = write[target]->GetQueue();
      void* addr = write[target]->Data();
      size_t size = write[target]->Size();
      ProxyKernel<<<1, 1>>>(queue, target, size, addr, 1);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (auto& efa : peer.efas) IO::Get().Join<FabricSelector>(efa);
    Run([&]() -> Coro<> {
      if (rank == 0) {
        DeviceRequest req;
        while (write[target]->GetQueue()->Pop(req)) co_await write[target]->Write(static_cast<int>(req.rank), req.imm, channel);
      } else if (rank == target) {
        co_await read[0]->WaitImmdata(1);
      }
      for (auto& efa : peer.efas) IO::Get().Quit<FabricSelector>(efa);
    }());
  }
};

/**
 * @brief Proxy write functor (multi-channel) - GPU kernel initiates write
 */
template <typename Peer>
struct ProxyWriteMulti {
  int target;

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& write, typename Peer::template Buffers<T>& read) {
    int rank = peer.mpi.GetWorldRank();

    if (rank == 0) {
      auto* queue = write[target]->GetQueue();
      void* addr = write[target]->Data();
      size_t size = write[target]->Size();
      ProxyKernel<<<1, 1>>>(queue, target, size, addr, 1);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (auto& efa : peer.efas) IO::Get().Join<FabricSelector>(efa);
    Run([&]() -> Coro<> {
      if (rank == 0) {
        DeviceRequest req;
        while (write[target]->GetQueue()->Pop(req)) co_await write[target]->Writeall(static_cast<int>(req.rank), req.imm);
      } else if (rank == target) {
        co_await read[0]->WaitallImmdata(1);
      }
      for (auto& efa : peer.efas) IO::Get().Quit<FabricSelector>(efa);
    }());
  }
};

/** @brief Bandwidth type tags */
struct SingleLinkBW {};
struct TotalLinkBW {};

/**
 * @brief Test configuration for single channel
 */
template <typename BufType, typename BWType = SingleLinkBW>
struct Test {
  static BenchResult Run(size_t size, const Options& opts, double single_bw, double total_bw) {
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
      peer.Warmup(write, read, ProxyWrite<FabricBench>{t, 0}, NoVerify{}, opts.warmup);
      auto r = peer.Bench(write, read, ProxyWrite<FabricBench>{t, 0}, NoVerify{}, opts.repeat);
      sum_bw += r.bw_gbps;
      sum_time += r.time_us;
    }
    int npairs = world - 1;
    double avg_bw = sum_bw / npairs;
    double link_bw = std::is_same_v<BWType, TotalLinkBW> ? total_bw : single_bw;
    double bus_bw = (link_bw > 0) ? (avg_bw / link_bw) * 100.0 : 0;
    return {size, sum_time / npairs, avg_bw, bus_bw};
  }
};

/**
 * @brief Test configuration for multi-channel
 */
template <typename BufType, typename BWType = TotalLinkBW>
struct TestMulti {
  static BenchResult Run(size_t size, const Options& opts, double single_bw, double total_bw) {
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
      peer.Warmup(write, read, ProxyWriteMulti<FabricBench>{t}, NoVerify{}, opts.warmup);
      auto r = peer.Bench(write, read, ProxyWriteMulti<FabricBench>{t}, NoVerify{}, opts.repeat);
      sum_bw += r.bw_gbps;
      sum_time += r.time_us;
    }
    int npairs = world - 1;
    double avg_bw = sum_bw / npairs;
    double link_bw = std::is_same_v<BWType, TotalLinkBW> ? total_bw : single_bw;
    double bus_bw = (link_bw > 0) ? (avg_bw / link_bw) * 100.0 : 0;
    return {size, sum_time / npairs, avg_bw, bus_bw};
  }
};

template <typename... Tests>
std::array<BenchResult, sizeof...(Tests)> RunTests(size_t size, const Options& opts, double single_bw, double total_bw) {
  std::array<BenchResult, sizeof...(Tests)> results;
  size_t i = 0;
  ((results[i++] = Tests::Run(size, opts, single_bw, total_bw), MPI_Barrier(MPI_COMM_WORLD)), ...);
  return results;
}

using SingleDMA = Test<SymmetricDMAMemory, SingleLinkBW>;
using MultiDMA = TestMulti<SymmetricDMAMemory, TotalLinkBW>;

int main(int argc, char* argv[]) {
  try {
    auto opts = parse_args(argc, argv);
    auto sizes = generate_sizes(opts);
    int rank = MPI::Get().GetWorldRank();
    int nranks = MPI::Get().GetWorldSize();

    FabricBench peer;
    double single_bw = peer.GetBandwidth(0) / 1e9;
    double total_bw = peer.GetTotalBandwidth() / 1e9;

    std::vector<std::array<BenchResult, 2>> results;
    for (auto size : sizes) {
      results.push_back(RunTests<SingleDMA, MultiDMA>(size, opts, single_bw, total_bw));
    }

    if (rank == 0) {
      FabricBench::Print(
          "EFA Proxy Write Benchmark", nranks, opts.warmup, opts.repeat, single_bw, "GPU kernel -> Queue -> RDMA write, rank0 -> rank_k",
          {"SingleDMA", "MultiDMA"}, results
      );
    }
    return 0;
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Fatal error: {}", e.what());
    return 1;
  }
}
