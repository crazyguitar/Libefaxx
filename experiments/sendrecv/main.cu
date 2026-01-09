/**
 * @file main.cu
 * @brief EFA SendRecv Benchmark - Point-to-point bandwidth measurement
 *
 * Measures send/recv bandwidth between rank 0 and all other ranks.
 * Pattern: rank0 <-> rank_k (k=1..N-1), results averaged across all pairs.
 */
#include <bench/arguments.h>
#include <rdma/fabric/memory.h>
#include <rdma/fabric/selector.h>
#include <rdma/ib/memory.h>
#include <rdma/ib/selector.h>

#include <bench/modules/sendrecv.cuh>
#include <bench/mpi/fabric.cuh>
#include <bench/mpi/ib.cuh>

/** @brief Pair-aware verification for GPU buffers */
struct PairVerifyGPU {
  int target;
  template <typename P, typename Buffers>
  void operator()(P& peer, Buffers& recv) const {
    const auto rank = peer.mpi.GetWorldRank();
    if (rank != 0 && rank != target) return;  // Only participating ranks verify
    const int peer_rank = (rank == 0) ? target : 0;
    const size_t buf_size = recv[peer_rank]->Size();
    const size_t num_ints = buf_size / sizeof(int);
    std::vector<int> host_buf(num_ints);
    CUDA_CHECK(cudaMemcpy(host_buf.data(), recv[peer_rank]->Data(), buf_size, cudaMemcpyDeviceToHost));
    if (VerifyBufferData(host_buf, peer_rank, rank, peer_rank) > 0) throw std::runtime_error("Verification failed");
  }
};

/** @brief Pair-aware verification for CPU buffers */
struct PairVerifyCPU {
  int target;
  template <typename P, typename Buffers>
  void operator()(P& peer, Buffers& recv) const {
    const auto rank = peer.mpi.GetWorldRank();
    if (rank != 0 && rank != target) return;  // Only participating ranks verify
    const int peer_rank = (rank == 0) ? target : 0;
    const size_t buf_size = recv[peer_rank]->Size();
    const size_t num_ints = buf_size / sizeof(int);
    std::vector<int> host_buf(num_ints);
    std::memcpy(host_buf.data(), recv[peer_rank]->Data(), buf_size);
    if (VerifyBufferData(host_buf, peer_rank, rank, peer_rank) > 0) throw std::runtime_error("Verification failed");
  }
};

/**
 * @brief Test configuration for a specific buffer type
 */
template <const char* Name, typename Peer, typename Selector, typename BufType, typename PairVerify = PairVerifyGPU>
struct Test {
  static BenchResult Run(size_t size, const Options& opts) {
    Peer peer;
    peer.Exchange();
    peer.Connect();
    int rank = peer.mpi.GetWorldRank();
    int world = peer.mpi.GetWorldSize();
    double link_bw = peer.GetBandwidth(0) / 1e9;

    auto send = peer.template Alloc<BufType>(size, rank);
    auto recv = peer.template Alloc<BufType>(size, -1);
    peer.Handshake(send, recv);

    double total_bw = 0;
    double total_time = 0;
    size_t progress_bw = static_cast<size_t>(link_bw * 1e9);
    for (int t = 1; t < world; ++t) {
      peer.Warmup(send, recv, PairBench<Peer, Selector>{t}, PairVerify{t}, opts.warmup);
      auto r = peer.Bench(Name, send, recv, PairBench<Peer, Selector>{t}, PairVerify{t}, opts.repeat, 0, progress_bw);
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
 * @return Array of BenchResult for each test type
 */
template <typename... Tests>
std::array<BenchResult, sizeof...(Tests)> RunTests(size_t size, const Options& opts) {
  std::array<BenchResult, sizeof...(Tests)> results;
  size_t i = 0;
  ((results[i++] = Tests::Run(size, opts), MPI_Barrier(MPI_COMM_WORLD)), ...);
  return results;
}

inline constexpr char kFabricDMA[] = "FabricDMA";
inline constexpr char kFabricHost[] = "FabricHost";
inline constexpr char kIBDMA[] = "IBDMA";
inline constexpr char kIBHost[] = "IBHost";

using FabricDMA = Test<kFabricDMA, FabricBench, fi::FabricSelector, fi::SymmetricDMAMemory, PairVerifyGPU>;
using FabricHost = Test<kFabricHost, FabricBench, fi::FabricSelector, fi::SymmetricHostMemory, PairVerifyCPU>;
using IBDMA = Test<kIBDMA, IBBench, ib::IBSelector, ib::SymmetricDMAMemory, PairVerifyGPU>;
using IBHost = Test<kIBHost, IBBench, ib::IBSelector, ib::SymmetricHostMemory, PairVerifyCPU>;

int main(int argc, char* argv[]) {
  try {
    auto opts = parse_args(argc, argv);
    auto sizes = generate_sizes(opts);
    int rank = MPI::Get().GetWorldRank();
    int nranks = MPI::Get().GetWorldSize();

    std::vector<std::array<BenchResult, 4>> results;
    for (auto size : sizes) {
      results.push_back(RunTests<FabricDMA, FabricHost, IBDMA, IBHost>(size, opts));
    }

    if (rank == 0) {
      FabricBench::Print(
          "EFA SendRecv Benchmark", nranks, opts.warmup, opts.repeat, "rank0 <-> rank_k (k=1..N-1), averaged across all pairs",
          {"FabricDMA", "FabricHost", "IBDMA", "IBHost"}, results
      );
    }
    return 0;
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Fatal error: {}", e.what());
    return 1;
  }
}
