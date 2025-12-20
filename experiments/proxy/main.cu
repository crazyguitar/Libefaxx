/**
 * @file main.cu
 * @brief EFA Write Benchmark - Point-to-point RDMA write bandwidth measurement
 *
 * Measures RDMA write bandwidth between rank 0 and all other ranks.
 * Pattern: rank0 -> rank_k (k=1..N-1), results averaged across all pairs.
 */
#include <bench/arguments.h>

#include <write/write.cuh>

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
    size_t num_ints = size / sizeof(int);

    auto [write, read] = peer.AllocPair<BufType>(size);
    peer.Handshake(write, read);

    double sum_bw = 0;
    double sum_time = 0;
    for (int t = 1; t < world; ++t) {
      if (rank == 0) RandInit(write[t].get(), num_ints, peer.stream);
      peer.Warmup(write, read, PairWrite{t, 0}, NoVerify{}, opts.warmup);
      auto r = peer.Bench(write, read, PairWrite{t, 0}, NoVerify{}, opts.repeat);
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
    size_t num_ints = size / sizeof(int);

    auto [write, read] = peer.AllocPair<BufType>(size);
    peer.Handshake(write, read);

    double sum_bw = 0;
    double sum_time = 0;
    for (int t = 1; t < world; ++t) {
      if (rank == 0) RandInit(write[t].get(), num_ints, peer.stream);
      peer.Warmup(write, read, PairWriteMulti{t}, NoVerify{}, opts.warmup);
      auto r = peer.Bench(write, read, PairWriteMulti{t}, NoVerify{}, opts.repeat);
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

using SinglePin = Test<DevicePinMemory, SingleLinkBW>;
using SingleDMA = Test<DeviceDMAMemory, SingleLinkBW>;
using MultiDMA = TestMulti<DeviceDMAMemory, TotalLinkBW>;

int main(int argc, char* argv[]) {
  try {
    auto opts = parse_args(argc, argv);
    auto sizes = generate_sizes(opts);
    int rank = MPI::Get().GetWorldRank();
    int nranks = MPI::Get().GetWorldSize();

    FabricBench peer;
    double single_bw = peer.GetBandwidth(0) / 1e9;
    double total_bw = peer.GetTotalBandwidth() / 1e9;

    std::vector<std::array<BenchResult, 3>> results;
    for (auto size : sizes) {
      results.push_back(RunTests<SinglePin, SingleDMA, MultiDMA>(size, opts, single_bw, total_bw));
    }

    if (rank == 0) {
      FabricBench::Print(
          "EFA Write Benchmark", nranks, opts.warmup, opts.repeat, single_bw, "rank0 -> rank_k (k=1..N-1), averaged across all pairs",
          {"SinglePin", "SingleDMA", "MultiDMA"}, results
      );
    }
    return 0;
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Fatal error: {}", e.what());
    return 1;
  }
}
