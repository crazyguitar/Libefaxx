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
#include <rdma/ib/memory.h>

#include <bench/mpi/fabric.cuh>
#include <bench/mpi/ib.cuh>

/** @brief Bandwidth type tags */
struct SingleLinkBW {};
struct TotalLinkBW {};

// ============================================================================
// Verification Functors
// ============================================================================

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

// ============================================================================
// Libfabric Write Functors
// ============================================================================

/** @brief Rank 0 write functor for libfabric */
struct FabricWrite {
  int target;
  int channel;

  template <typename T>
  Coro<> operator()(FabricBench& peer, typename FabricBench::Buffers<T>& write) {
    if (peer.mpi.GetWorldRank() != 0) co_return;
    co_await write[target]->Write(target, 1, channel);
  }
};

/** @brief Target rank wait for immediate data (libfabric) */
struct FabricRead {
  int target;

  template <typename T>
  Coro<> operator()(FabricBench& peer, typename FabricBench::Buffers<T>& read) {
    if (peer.mpi.GetWorldRank() != target) co_return;
    co_await read[0]->WaitImmdata(1);
  }
};

/** @brief Combined pair benchmark functor for libfabric */
struct FabricPairWrite {
  int target;
  int channel;

  template <typename T>
  void operator()(FabricBench& peer, typename FabricBench::Buffers<T>& write, typename FabricBench::Buffers<T>& read) {
    for (auto& efa : peer.efas) IO::Get().Join<fi::FabricSelector>(efa);
    Run([&]() -> Coro<> {
      co_await FabricWrite{target, channel}.template operator()<T>(peer, write);
      co_await FabricRead{target}.template operator()<T>(peer, read);
      for (auto& efa : peer.efas) IO::Get().Quit<fi::FabricSelector>(efa);
    }());
  }
};

// ============================================================================
// IBVerbs Write Functors
// ============================================================================

/** @brief Rank 0 write functor for ibverbs */
struct IBWrite {
  int target;
  int channel;

  template <typename T>
  Coro<> operator()(IBBench& peer, typename IBBench::Buffers<T>& write) {
    if (peer.mpi.GetWorldRank() != 0) co_return;
    co_await write[target]->Write(target, 1, channel);
  }
};

/** @brief Target rank wait for immediate data (ibverbs) */
struct IBRead {
  int target;

  template <typename T>
  Coro<> operator()(IBBench& peer, typename IBBench::Buffers<T>& read) {
    if (peer.mpi.GetWorldRank() != target) co_return;
    co_await read[0]->WaitImmdata(1);
  }
};

/** @brief Combined pair benchmark functor for ibverbs */
struct IBPairWrite {
  int target;
  int channel;

  template <typename T>
  void operator()(IBBench& peer, typename IBBench::Buffers<T>& write, typename IBBench::Buffers<T>& read) {
    for (auto& efa : peer.efas) IO::Get().Join<ib::IBSelector>(efa);
    Run([&]() -> Coro<> {
      co_await IBWrite{target, channel}.template operator()<T>(peer, write);
      co_await IBRead{target}.template operator()<T>(peer, read);
      for (auto& efa : peer.efas) IO::Get().Quit<ib::IBSelector>(efa);
    }());
  }
};

// ============================================================================
// Test Configurations
// ============================================================================

/** @brief Libfabric test configuration */
template <const char* Name, typename BufType, typename Verify, typename BWType = SingleLinkBW>
struct FabricTest {
  static BenchResult Run(size_t size, const Options& opts, double single_bw, double total_bw) {
    FabricBench peer;
    peer.Exchange();
    peer.Connect();
    int rank = peer.mpi.GetWorldRank();
    int world = peer.mpi.GetWorldSize();

    auto [write, read] = peer.AllocPair<BufType>(size);
    peer.Handshake(write, read);

    double sum_bw = 0, sum_time = 0;
    size_t progress_bw = static_cast<size_t>(single_bw * 1e9);
    for (int t = 1; t < world; ++t) {
      peer.Warmup(write, read, FabricPairWrite{t, 0}, Verify{t}, opts.warmup);
      auto r = peer.Bench(Name, write, read, FabricPairWrite{t, 0}, Verify{t}, opts.repeat, 0, progress_bw);
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

/** @brief IBVerbs test configuration */
template <const char* Name, typename BufType, typename Verify, typename BWType = SingleLinkBW>
struct IBTest {
  static BenchResult Run(size_t size, const Options& opts, double single_bw, double total_bw) {
    IBBench peer;
    peer.Exchange();
    peer.Connect();
    int rank = peer.mpi.GetWorldRank();
    int world = peer.mpi.GetWorldSize();

    auto [write, read] = peer.AllocPair<BufType>(size);
    peer.Handshake(write, read);

    double sum_bw = 0, sum_time = 0;
    size_t progress_bw = static_cast<size_t>(single_bw * 1e9);
    for (int t = 1; t < world; ++t) {
      peer.Warmup(write, read, IBPairWrite{t, 0}, Verify{t}, opts.warmup);
      auto r = peer.Bench(Name, write, read, IBPairWrite{t, 0}, Verify{t}, opts.repeat, 0, progress_bw);
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

// Test name constants
inline constexpr char kFabricDMA[] = "FabricDMA";
inline constexpr char kFabricHost[] = "FabricHost";
inline constexpr char kIBDMA[] = "IBDMA";
inline constexpr char kIBHost[] = "IBHost";

// Test type aliases
using FabricDMA = FabricTest<kFabricDMA, fi::SymmetricDMAMemory, WriteVerifyGPU>;
using FabricHost = FabricTest<kFabricHost, fi::SymmetricHostMemory, WriteVerifyCPU>;
using IBDMA = IBTest<kIBDMA, ib::SymmetricDMAMemory, WriteVerifyGPU>;
using IBHost = IBTest<kIBHost, ib::SymmetricHostMemory, WriteVerifyCPU>;

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
      results.push_back(RunTests<FabricDMA, FabricHost, IBDMA, IBHost>(size, opts, single_bw, total_bw));
    }

    if (rank == 0) {
      FabricBench::Print(
          "EFA Write Benchmark", nranks, opts.warmup, opts.repeat, single_bw, "rank0 -> rank_k (k=1..N-1), averaged across all pairs",
          {"FabricDMA", "FabricHost", "IBDMA", "IBHost"}, results
      );
    }
    return 0;
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Fatal error: {}", e.what());
    return 1;
  }
}
