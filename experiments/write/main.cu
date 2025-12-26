/**
 * @file main.cu
 * @brief EFA Write Benchmark - Point-to-point RDMA write bandwidth measurement
 *
 * Measures RDMA write bandwidth between rank 0 and all other ranks.
 * Pattern: rank0 -> rank_k (k=1..N-1), results averaged across all pairs.
 */
#include <bench/arguments.h>
#include <rdma/fabric/memory.h>

#include <bench/modules/write.cuh>
#include <bench/mpi/fabric.cuh>

/** @brief Bandwidth type tags */
struct SingleLinkBW {};
struct TotalLinkBW {};

/** @brief Pair-aware verification for RDMA write (only target receives) */
struct WriteVerifyGPU {
  int target;
  template <typename P, typename Buffers>
  void operator()(P& peer, Buffers& read) const {
    const auto rank = peer.mpi.GetWorldRank();
    if (rank != target) return;  // Only target verifies
    const size_t buf_size = read[0]->Size();
    const size_t num_ints = buf_size / sizeof(int);
    std::vector<int> host_buf(num_ints);
    CUDA_CHECK(cudaMemcpy(host_buf.data(), read[0]->Data(), buf_size, cudaMemcpyDeviceToHost));
    if (VerifyBufferData(host_buf, 0, rank, 0) > 0) throw std::runtime_error("Verification failed");
  }
};

/**
 * @brief Test configuration for single channel
 */
template <const char* Name, typename BufType, typename BWType = SingleLinkBW>
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
    size_t progress_bw = static_cast<size_t>(single_bw * 1e9);
    for (int t = 1; t < world; ++t) {
      peer.Warmup(write, read, PairWrite<FabricBench>{t, 0}, WriteVerifyGPU{t}, opts.warmup);
      auto r = peer.Bench(Name, write, read, PairWrite<FabricBench>{t, 0}, WriteVerifyGPU{t}, opts.repeat, 0, progress_bw);
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
template <const char* Name, typename BufType, typename BWType = TotalLinkBW>
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
    size_t progress_bw = static_cast<size_t>(total_bw * 1e9);  // Convert Gbps to bits/sec
    for (int t = 1; t < world; ++t) {
      peer.Warmup(write, read, PairWriteMulti<FabricBench>{t}, WriteVerifyGPU{t}, opts.warmup);
      auto r = peer.Bench(Name, write, read, PairWriteMulti<FabricBench>{t}, WriteVerifyGPU{t}, opts.repeat, 0, progress_bw);
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
 * @brief Test configuration for round-robin (all targets in parallel)
 */
template <const char* Name, typename BufType, typename BWType = TotalLinkBW>
struct TestRoundRobin {
  static BenchResult Run(size_t size, const Options& opts, double single_bw, double total_bw) {
    FabricBench peer;
    peer.Exchange();
    peer.Connect();
    int world = peer.mpi.GetWorldSize();

    auto [write, read] = peer.AllocPair<BufType>(size);
    peer.Handshake(write, read);

    size_t num_channels = peer.efas.size();
    auto func = [&](FabricBench& p, FabricBench::Buffers<BufType>& w, FabricBench::Buffers<BufType>& r) {
      for (auto& efa : p.efas) IO::Get().Join<FabricSelector>(efa);
      ::Run([&]() -> Coro<> {
        co_await RunWriteRoundRobin(w, r, num_channels, p.mpi.GetWorldSize(), p.mpi.GetWorldRank());
        for (auto& efa : p.efas) IO::Get().Quit<FabricSelector>(efa);
      }());
    };
    auto noop = [](auto&, auto&) {};

    peer.Warmup(write, read, func, noop, opts.warmup);
    size_t total_bytes = size * (world - 1);
    size_t progress_bw = static_cast<size_t>(total_bw * 1e9);  // Convert Gbps to bits/sec
    auto r = peer.Bench(Name, write, read, func, noop, opts.repeat, total_bytes, progress_bw);

    // Total bandwidth = size * (world-1) targets / time
    double bw = (total_bytes * 8) / (r.time_us * 1e3);  // Gbps
    double link_bw = std::is_same_v<BWType, TotalLinkBW> ? total_bw : single_bw;
    double bus_bw = (link_bw > 0) ? (bw / link_bw) * 100.0 : 0;
    return {size, r.time_us, bw, bus_bw};
  }
};

template <typename... Tests>
std::array<BenchResult, sizeof...(Tests)> RunTests(size_t size, const Options& opts, double single_bw, double total_bw) {
  std::array<BenchResult, sizeof...(Tests)> results;
  size_t i = 0;
  ((results[i++] = Tests::Run(size, opts, single_bw, total_bw), MPI_Barrier(MPI_COMM_WORLD)), ...);
  return results;
}

inline constexpr char kSinglePin[] = "SinglePin";
inline constexpr char kSingleDMA[] = "SingleDMA";
inline constexpr char kMultiDMA[] = "MultiDMA";
inline constexpr char kRoundRobinDMA[] = "RoundRobinDMA";

using SinglePin = Test<kSinglePin, SymmetricPinMemory, SingleLinkBW>;
using SingleDMA = Test<kSingleDMA, SymmetricDMAMemory, SingleLinkBW>;
using MultiDMA = TestMulti<kMultiDMA, SymmetricDMAMemory, TotalLinkBW>;
using RoundRobinDMA = TestRoundRobin<kRoundRobinDMA, SymmetricDMAMemory, TotalLinkBW>;

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
      results.push_back(RunTests<SinglePin, SingleDMA, MultiDMA, RoundRobinDMA>(size, opts, single_bw, total_bw));
    }

    if (rank == 0) {
      FabricBench::Print(
          "EFA Write Benchmark", nranks, opts.warmup, opts.repeat, single_bw, "rank0 -> rank_k (k=1..N-1), averaged across all pairs",
          {"SinglePin", "SingleDMA", "MultiDMA", "RoundRobinDMA"}, results
      );
    }
    return 0;
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Fatal error: {}", e.what());
    return 1;
  }
}
