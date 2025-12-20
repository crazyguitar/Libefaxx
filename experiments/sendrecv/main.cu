/**
 * @file main.cu
 * @brief EFA SendRecv Benchmark - Point-to-point bandwidth measurement
 *
 * Measures send/recv bandwidth between rank 0 and all other ranks.
 * Pattern: rank0 <-> rank_k (k=1..N-1), results averaged across all pairs.
 */
#include <bench/arguments.h>

#include <bench/modules/sendrecv.cuh>
#include <bench/mpi/fabric.cuh>

/**
 * @brief Test configuration for a specific buffer type
 * @tparam BufType Buffer type (DeviceDMABuffer or HostBuffer)
 */
template <typename BufType>
struct Test {
  /**
   * @brief Run benchmark for given size
   * @param size Buffer size in bytes
   * @param opts Benchmark options (warmup, repeat iterations)
   * @param link_bw Theoretical link bandwidth in Gbps
   * @return BenchResult with averaged metrics across all pairs
   */
  static BenchResult Run(size_t size, const Options& opts, double link_bw) {
    FabricBench peer;
    peer.Exchange();
    peer.Connect();
    int rank = peer.mpi.GetWorldRank();
    int world = peer.mpi.GetWorldSize();
    size_t num_ints = size / sizeof(int);

    auto send = peer.Alloc<BufType>(size, rank);
    auto recv = peer.Alloc<BufType>(size, -1);

    double total_bw = 0;
    double total_time = 0;
    for (int t = 1; t < world; ++t) {
      if (rank == 0)
        RandInit(send[t].get(), num_ints, peer.stream);
      else if (rank == t)
        RandInit(send[0].get(), num_ints, peer.stream);

      peer.Warmup(send, recv, PairBench<FabricBench>{t}, NoVerify{}, opts.warmup);
      auto r = peer.Bench(send, recv, PairBench<FabricBench>{t}, NoVerify{}, opts.repeat);
      total_bw += r.bw_gbps;
      total_time += r.time_us;
    }
    int npairs = world - 1;
    double avg_bw = total_bw / npairs;
    double bus_bw = (link_bw > 0) ? (avg_bw / link_bw) * 100.0 : 0;
    return {size, total_time / npairs, avg_bw, bus_bw};
  }
};

/**
 * @brief Run multiple test configurations sequentially
 * @tparam Tests Test types to run
 * @param size Buffer size in bytes
 * @param opts Benchmark options
 * @param link_bw Theoretical link bandwidth in Gbps
 * @return Array of BenchResult for each test type
 */
template <typename... Tests>
std::array<BenchResult, sizeof...(Tests)> RunTests(size_t size, const Options& opts, double link_bw) {
  std::array<BenchResult, sizeof...(Tests)> results;
  size_t i = 0;
  ((results[i++] = Tests::Run(size, opts, link_bw), MPI_Barrier(MPI_COMM_WORLD)), ...);
  return results;
}

using SingleDevice = Test<DeviceDMABuffer>;  ///< GPU DMA buffer test
using SingleHost = Test<HostBuffer>;         ///< Host pinned buffer test

int main(int argc, char* argv[]) {
  try {
    auto opts = parse_args(argc, argv);
    auto sizes = generate_sizes(opts);
    int rank = MPI::Get().GetWorldRank();
    int nranks = MPI::Get().GetWorldSize();

    // link_attr->speed is in bits/sec
    FabricBench peer;
    double link_bw = peer.GetBandwidth(0) / 1e9;

    // Run all benchmarks
    std::vector<std::array<BenchResult, 2>> results;
    for (auto size : sizes) {
      results.push_back(RunTests<SingleDevice, SingleHost>(size, opts, link_bw));
    }

    // Print summary at the end (rank 0 only)
    if (rank == 0) {
      FabricBench::Print(
          "EFA SendRecv Benchmark", nranks, opts.warmup, opts.repeat, link_bw, "rank0 <-> rank_k (k=1..N-1), averaged across all pairs",
          {"Device(Gbps)", "Host(Gbps)"}, results
      );
    }
    return 0;
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Fatal error: {}", e.what());
    return 1;
  }
}
