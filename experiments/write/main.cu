/**
 * @file main.cu
 * @brief EFA Write Benchmark - Point-to-point RDMA write bandwidth measurement
 *
 * Measures RDMA write bandwidth between rank 0 and all other ranks using both
 * libfabric and ibverbs implementations.
 * Pattern: rank0 -> rank_k (k=1..N-1), results averaged across all pairs.
 */
#include <bench/arguments.h>
#include <io/runner.h>
#include <rdma/fabric/memory.h>
#include <rdma/fabric/selector.h>
#include <rdma/ib/memory.h>
#include <rdma/ib/selector.h>

#include <bench/modules/write.cuh>
#include <bench/mpi/fabric.cuh>
#include <bench/mpi/ib.cuh>

/** @brief Bandwidth type tags */
struct SingleLinkBW {};
struct TotalLinkBW {};

/** @brief Pair-aware verification for RDMA write (only target receives) */
struct WriteVerifyGPU {
  int target;
  template <typename P, typename Buffers>
  void operator()(P& peer, Buffers& read) const {
    const auto rank = peer.mpi.GetWorldRank();
    if (rank != target) return;
    const size_t buf_size = read[0]->Size();
    const size_t num_ints = buf_size / sizeof(int);
    std::vector<int> host_buf(num_ints);
    CUDA_CHECK(cudaMemcpy(host_buf.data(), read[0]->Data(), buf_size, cudaMemcpyDeviceToHost));
    if (VerifyBufferData(host_buf, 0, rank, 0) > 0) throw std::runtime_error("Verification failed");
  }
};

/** @brief Pair-aware verification for CPU buffers */
struct WriteVerifyCPU {
  int target;
  template <typename P, typename Buffers>
  void operator()(P& peer, Buffers& read) const {
    const auto rank = peer.mpi.GetWorldRank();
    if (rank != target) return;
    const size_t buf_size = read[0]->Size();
    const size_t num_ints = buf_size / sizeof(int);
    std::vector<int> host_buf(num_ints);
    std::memcpy(host_buf.data(), read[0]->Data(), buf_size);
    if (VerifyBufferData(host_buf, 0, rank, 0) > 0) throw std::runtime_error("Verification failed");
  }
};

/** @brief Unified test configuration template */
template <const char* Name, typename Peer, typename Selector, typename BufType, typename Verify, typename BWType = SingleLinkBW>
struct Test {
  static BenchResult Run(size_t size, const Options& opts) {
    Peer peer;
    peer.Exchange();
    peer.Connect();
    int rank = peer.mpi.GetWorldRank();
    int world = peer.mpi.GetWorldSize();
    double single_bw = peer.GetBandwidth(0) / 1e9;
    double total_bw = peer.GetTotalBandwidth() / 1e9;

    auto [write, read] = peer.template AllocPair<BufType>(size);
    peer.Handshake(write, read);

    double sum_bw = 0, sum_time = 0;
    size_t progress_bw = static_cast<size_t>(single_bw * 1e9);
    for (int t = 1; t < world; ++t) {
      peer.Warmup(write, read, PairWrite<Peer, Selector>{t, 0}, Verify{t}, opts.warmup);
      auto r = peer.Bench(Name, write, read, PairWrite<Peer, Selector>{t, 0}, Verify{t}, opts.repeat, 0, progress_bw);
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

/** @brief Multi-channel test configuration (uses all EFA channels) */
template <const char* Name, typename Peer, typename Selector, typename BufType, typename Verify, typename BWType = TotalLinkBW>
struct TestMulti {
  static BenchResult Run(size_t size, const Options& opts) {
    Peer peer;
    peer.Exchange();
    peer.Connect();
    int rank = peer.mpi.GetWorldRank();
    int world = peer.mpi.GetWorldSize();
    double single_bw = peer.GetBandwidth(0) / 1e9;
    double total_bw = peer.GetTotalBandwidth() / 1e9;

    auto [write, read] = peer.template AllocPair<BufType>(size);
    peer.Handshake(write, read);

    double sum_bw = 0, sum_time = 0;
    size_t progress_bw = static_cast<size_t>(total_bw * 1e9);
    for (int t = 1; t < world; ++t) {
      peer.Warmup(write, read, PairWriteMulti<Peer, Selector>{t}, Verify{t}, opts.warmup);
      auto r = peer.Bench(Name, write, read, PairWriteMulti<Peer, Selector>{t}, Verify{t}, opts.repeat, 0, progress_bw);
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

/** @brief Round-robin channel test configuration (parallel to all targets) */
template <const char* Name, typename Peer, typename Selector, typename BufType, typename Verify, typename BWType = TotalLinkBW>
struct TestRoundRobin {
  static BenchResult Run(size_t size, const Options& opts) {
    Peer peer;
    peer.Exchange();
    peer.Connect();
    int rank = peer.mpi.GetWorldRank();
    int world = peer.mpi.GetWorldSize();
    double single_bw = peer.GetBandwidth(0) / 1e9;
    double total_bw = peer.GetTotalBandwidth() / 1e9;

    auto [write, read] = peer.template AllocPair<BufType>(size);
    peer.Handshake(write, read);

    auto noop = [](auto&, auto&) {};
    size_t total_bytes = size * (world - 1);
    size_t progress_bw = static_cast<size_t>(total_bw * 1e9);
    peer.Warmup(write, read, PairWriteRoundRobinAll<Peer, Selector>{}, noop, opts.warmup);
    auto r = peer.Bench(Name, write, read, PairWriteRoundRobinAll<Peer, Selector>{}, noop, opts.repeat, total_bytes, progress_bw);

    double bw = (total_bytes * 8) / (r.time_us * 1e3);  // Gbps
    double link_bw = std::is_same_v<BWType, TotalLinkBW> ? total_bw : single_bw;
    double bus_bw = (link_bw > 0) ? (bw / link_bw) * 100.0 : 0;
    return {size, r.time_us, bw, bus_bw};
  }
};

template <typename... Tests>
std::array<BenchResult, sizeof...(Tests)> RunTests(size_t size, const Options& opts) {
  std::array<BenchResult, sizeof...(Tests)> results;
  size_t i = 0;
  ((results[i++] = Tests::Run(size, opts), MPI_Barrier(MPI_COMM_WORLD)), ...);
  return results;
}

// Test name constants
inline constexpr char kFabricDMA[] = "FabricDMA";
inline constexpr char kFabricHost[] = "FabricHost";
inline constexpr char kFabricMultiDMA[] = "FabricMultiDMA";
inline constexpr char kFabricRoundRobin[] = "FabricRoundRobin";
inline constexpr char kIBDMA[] = "IBDMA";
inline constexpr char kIBHost[] = "IBHost";
inline constexpr char kIBMultiDMA[] = "IBMultiDMA";
inline constexpr char kIBRoundRobin[] = "IBRoundRobin";

// Test type aliases - Single channel
using FabricDMA = Test<kFabricDMA, FabricBench, fi::FabricSelector, fi::SymmetricDMAMemory, WriteVerifyGPU>;
using FabricHost = Test<kFabricHost, FabricBench, fi::FabricSelector, fi::SymmetricHostMemory, WriteVerifyCPU>;
using IBDMA = Test<kIBDMA, IBBench, ib::IBSelector, ib::SymmetricDMAMemory, WriteVerifyGPU>;
using IBHost = Test<kIBHost, IBBench, ib::IBSelector, ib::SymmetricHostMemory, WriteVerifyCPU>;

// Test type aliases - Multi channel
using FabricMultiDMA = TestMulti<kFabricMultiDMA, FabricBench, fi::FabricSelector, fi::SymmetricDMAMemory, WriteVerifyGPU>;
using IBMultiDMA = TestMulti<kIBMultiDMA, IBBench, ib::IBSelector, ib::SymmetricDMAMemory, WriteVerifyGPU>;

// Test type aliases - Round-robin channel
using FabricRoundRobin = TestRoundRobin<kFabricRoundRobin, FabricBench, fi::FabricSelector, fi::SymmetricDMAMemory, WriteVerifyGPU>;
using IBRoundRobin = TestRoundRobin<kIBRoundRobin, IBBench, ib::IBSelector, ib::SymmetricDMAMemory, WriteVerifyGPU>;

int main(int argc, char* argv[]) {
  try {
    auto opts = parse_args(argc, argv);
    auto sizes = generate_sizes(opts);
    int rank = MPI::Get().GetWorldRank();
    int nranks = MPI::Get().GetWorldSize();

    // Run Fabric tests
    std::vector<std::array<BenchResult, 4>> fabric_results;
    for (auto size : sizes) {
      fabric_results.push_back(RunTests<FabricDMA, FabricHost, FabricMultiDMA, FabricRoundRobin>(size, opts));
    }

    // Run IB tests
    std::vector<std::array<BenchResult, 4>> ib_results;
    for (auto size : sizes) {
      ib_results.push_back(RunTests<IBDMA, IBHost, IBMultiDMA, IBRoundRobin>(size, opts));
    }

    if (rank == 0) {
      FabricBench::Print(
          "EFA Write Benchmark (Fabric)", nranks, opts.warmup, opts.repeat, "rank0 -> rank_k (k=1..N-1), averaged across all pairs",
          {"FabricDMA", "FabricHost", "FabricMultiDMA", "FabricRoundRobin"}, fabric_results
      );
      IBBench::Print(
          "EFA Write Benchmark (IB)", nranks, opts.warmup, opts.repeat, "rank0 -> rank_k (k=1..N-1), averaged across all pairs",
          {"IBDMA", "IBHost", "IBMultiDMA", "IBRoundRobin"}, ib_results
      );
    }
    return 0;
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Fatal error: {}", e.what());
    return 1;
  }
}
