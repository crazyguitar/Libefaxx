/**
 * @file main.cu
 * @brief EFA RDMA Write Benchmark - All-to-all bandwidth measurement
 *
 * Measures RDMA write bandwidth using all-to-all communication pattern.
 * Tests single/multi channel with DMA and pinned memory.
 */
#include <bench/arguments.h>
#include <rdma/fabric/memory.h>
#include <rdma/proxy.h>

#include <bench/modules/all2all.cuh>
#include <bench/mpi/fabric.cuh>

/** @brief Bandwidth type tags */
struct SingleLinkBW {};
struct TotalLinkBW {};

/// All2all RDMA write (single channel)
struct All2all {
  int channel = 0;
  template <typename T>
  void operator()(FabricBench& peer, FabricBench::Buffers<T>& write, FabricBench::Buffers<T>& read) {
    for (auto& efa : peer.efas) IO::Get().Join<FabricProxy>(efa);
    Run([&]() -> Coro<> {
      co_await RunAll2allWrite(write, read, channel, peer.mpi.GetWorldSize(), peer.mpi.GetWorldRank());
      for (auto& efa : peer.efas) IO::Get().Quit<FabricProxy>(efa);
    }());
  }
};

/// All2all RDMA write (multi-channel)
struct All2allMulti {
  template <typename T>
  void operator()(FabricBench& peer, FabricBench::Buffers<T>& write, FabricBench::Buffers<T>& read) {
    for (auto& efa : peer.efas) IO::Get().Join<FabricProxy>(efa);
    Run([&]() -> Coro<> {
      co_await RunAll2allWriteMultiChannel(write, read, peer.mpi.GetWorldSize(), peer.mpi.GetWorldRank());
      for (auto& efa : peer.efas) IO::Get().Quit<FabricProxy>(efa);
    }());
  }
};

/// Test configuration with bandwidth type
template <typename BufType, typename Func, typename BWType = SingleLinkBW, typename Verify = VerifyGPU>
struct Test {
  static BenchResult Run(size_t size, const Options& opts, double single_bw, double total_bw) {
    FabricBench peer;
    peer.Exchange();
    peer.Connect();
    auto [write, read] = peer.AllocPair<BufType>(size);
    peer.Handshake(write, read);
    peer.Warmup(write, read, Func{}, Verify{}, opts.warmup);
    auto r = peer.Bench(write, read, Func{}, Verify{}, opts.repeat);
    double link_bw = std::is_same_v<BWType, TotalLinkBW> ? total_bw : single_bw;
    r.bus_bw = (link_bw > 0) ? (r.bw_gbps / link_bw) * 100.0 : 0;
    return r;
  }
};

/// Run multiple test configurations
template <typename... Tests>
std::array<BenchResult, sizeof...(Tests)> RunTests(size_t size, const Options& opts, double single_bw, double total_bw) {
  std::array<BenchResult, sizeof...(Tests)> results;
  size_t i = 0;
  ((results[i++] = Tests::Run(size, opts, single_bw, total_bw), MPI_Barrier(MPI_COMM_WORLD)), ...);
  return results;
}

using SingleDMA = Test<DeviceDMAMemory, All2all, SingleLinkBW>;
using MultiDMA = Test<DeviceDMAMemory, All2allMulti, TotalLinkBW>;
using SinglePin = Test<DevicePinMemory, All2all, SingleLinkBW>;
using MultiPin = Test<DevicePinMemory, All2allMulti, TotalLinkBW>;

int main(int argc, char* argv[]) {
  try {
    auto opts = parse_args(argc, argv);
    auto sizes = generate_sizes(opts);
    int rank = MPI::Get().GetWorldRank();
    int nranks = MPI::Get().GetWorldSize();

    FabricBench peer;
    double single_bw = peer.GetBandwidth(0) / 1e9;
    double total_bw = peer.GetTotalBandwidth() / 1e9;

    std::vector<std::array<BenchResult, 4>> results;
    for (auto size : sizes) {
      results.push_back(RunTests<SingleDMA, MultiDMA, SinglePin, MultiPin>(size, opts, single_bw, total_bw));
    }

    if (rank == 0) {
      FabricBench::Print(
          "EFA RDMA Write Benchmark", nranks, opts.warmup, opts.repeat, single_bw, "all-to-all RDMA write",
          {"SingleDMA", "MultiDMA", "SinglePin", "MultiPin"}, results
      );
    }
    return 0;
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Fatal error: {}", e.what());
    return 1;
  }
}
