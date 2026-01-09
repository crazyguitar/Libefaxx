/**
 * @file main.cu
 * @brief EFA Proxy Write Benchmark - GPU-initiated RDMA write via CPU proxy
 *
 * Benchmarks three queue types for GPU→CPU communication:
 *   - Queue(Managed):  cudaMallocManaged - unified memory with page migration
 *   - PinnedQueue:     cudaHostAlloc - GPU writes over PCIe to host DRAM
 *   - GdrQueue:        GDRCopy - GPU writes to VRAM, CPU reads via BAR1
 */
#include <bench/arguments.h>
#include <rdma/fabric/memory.h>

#include <bench/modules/proxy.cuh>
#include <bench/mpi/fabric.cuh>

template <
    typename Peer,
    typename Selector,
    bool MultiChannel,
    template <typename, typename, bool, template <typename, typename> class> class Writer = ProxyWrite,
    template <typename, typename> class Launcher = KernelBlocking>
struct ProxyBench {
  int iters, target;

  template <typename T>
  BenchResult Run(Peer& peer, typename Peer::template Buffers<T>& write, typename Peer::template Buffers<T>& read) {
    int rank = peer.mpi.GetWorldRank();
    size_t size = (rank == 0) ? write[target]->Size() : read[0]->Size();
    if (rank != 0 && rank != target) return {size, 0, 0, 0};

    auto start = std::chrono::high_resolution_clock::now();
    Writer<Peer, Selector, MultiChannel, Launcher>{iters, target}(peer, write);
    ProxyRead<Peer, Selector, MultiChannel>{iters}(peer, read);
    auto end = std::chrono::high_resolution_clock::now();

    double time_us = std::chrono::duration<double, std::micro>(end - start).count();
    double lat_us = time_us / iters;
    double bw_gbps = (rank == 0) ? (static_cast<double>(size) * iters * 8.0) / (time_us * 1000.0) : 0;
    return {size, lat_us, bw_gbps, 0.0};
  }
};

template <typename BufType, bool MultiChannel, template <typename, typename> class Launcher = KernelBlocking>
struct Test {
  static BenchResult Run(size_t size, const Options& opts, double single_bw, double total_bw) {
    FabricBench peer;
    peer.Exchange();
    peer.Connect();

    auto [write, read] = peer.AllocPair<BufType>(size);
    peer.Handshake(write, read);

    double sum_bw = 0, sum_time = 0;
    int world = peer.mpi.GetWorldSize();
    for (int t = 1; t < world; ++t) {
      auto r = ProxyBench<FabricBench, fi::FabricSelector, MultiChannel, ProxyWrite, Launcher>{opts.repeat, t}.Run(peer, write, read);
      sum_bw += r.bw_gbps;
      sum_time += r.time_us;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int npairs = world - 1;
    double avg_bw = sum_bw / npairs;
    double link_bw = MultiChannel ? total_bw : single_bw;
    double avg_lat = sum_time / npairs;
    return {size, avg_lat, avg_bw, (link_bw > 0) ? (avg_bw / link_bw) * 100.0 : 0};
  }
};

template <typename... Tests>
std::array<BenchResult, sizeof...(Tests)> RunTests(size_t size, const Options& opts, double single_bw, double total_bw) {
  std::array<BenchResult, sizeof...(Tests)> results;
  size_t i = 0;
  ((results[i++] = Tests::Run(size, opts, single_bw, total_bw), MPI_Barrier(MPI_COMM_WORLD)), ...);
  return results;
}

// Queue type aliases
using ManagedMem = fi::SymmetricDMAMemoryT<Queue<rdma::DeviceRequest>>;
using PinnedMem = fi::SymmetricDMAMemoryT<PinnedQueue<rdma::DeviceRequest>>;
using GdrMem = fi::SymmetricDMAMemoryT<GdrQueue<rdma::DeviceRequest>>;

// Single channel - Blocking mode
using ManagedBlocking = Test<ManagedMem, false, KernelBlocking>;
using PinnedBlocking = Test<PinnedMem, false, KernelBlocking>;
using GdrBlocking = Test<GdrMem, false, KernelBlocking>;

// Single channel - NBI mode
using ManagedNBI = Test<ManagedMem, false, KernelNBI>;
using PinnedNBI = Test<PinnedMem, false, KernelNBI>;
using GdrNBI = Test<GdrMem, false, KernelNBI>;

// Multi channel - Blocking mode
using ManagedMultiBlocking = Test<ManagedMem, true, KernelBlocking>;
using PinnedMultiBlocking = Test<PinnedMem, true, KernelBlocking>;
using GdrMultiBlocking = Test<GdrMem, true, KernelBlocking>;

// Multi channel - NBI mode
using ManagedMultiNBI = Test<ManagedMem, true, KernelNBI>;
using PinnedMultiNBI = Test<PinnedMem, true, KernelNBI>;
using GdrMultiNBI = Test<GdrMem, true, KernelNBI>;

int main(int argc, char* argv[]) {
  try {
    auto opts = parse_args(argc, argv);
    auto sizes = generate_sizes(opts);
    int rank = MPI::Get().GetWorldRank();
    int nranks = MPI::Get().GetWorldSize();

    FabricBench peer;
    double single_bw = peer.GetBandwidth(0) / 1e9;
    double total_bw = peer.GetTotalBandwidth() / 1e9;

    // Single channel - Blocking
    std::vector<std::array<BenchResult, 3>> single_blocking;
    for (auto size : sizes) {
      single_blocking.push_back(RunTests<ManagedBlocking, PinnedBlocking, GdrBlocking>(size, opts, single_bw, total_bw));
    }

    // Single channel - NBI
    std::vector<std::array<BenchResult, 3>> single_nbi;
    for (auto size : sizes) {
      single_nbi.push_back(RunTests<ManagedNBI, PinnedNBI, GdrNBI>(size, opts, single_bw, total_bw));
    }

    // Multi channel - Blocking
    std::vector<std::array<BenchResult, 3>> multi_blocking;
    for (auto size : sizes) {
      multi_blocking.push_back(RunTests<ManagedMultiBlocking, PinnedMultiBlocking, GdrMultiBlocking>(size, opts, single_bw, total_bw));
    }

    // Multi channel - NBI
    std::vector<std::array<BenchResult, 3>> multi_nbi;
    for (auto size : sizes) {
      multi_nbi.push_back(RunTests<ManagedMultiNBI, PinnedMultiNBI, GdrMultiNBI>(size, opts, single_bw, total_bw));
    }

    if (rank == 0) {
      FabricBench::Print(
          "EFA Proxy Write - Single Channel Blocking", nranks, opts.warmup, opts.repeat, single_bw, "GPU -> Queue -> RDMA (sync per op)",
          {"Managed", "Pinned", "GdrQueue"}, single_blocking
      );
      printf("\n");
      FabricBench::Print(
          "EFA Proxy Write - Single Channel NBI", nranks, opts.warmup, opts.repeat, single_bw, "GPU -> Queue -> RDMA (batch)",
          {"Managed", "Pinned", "GdrQueue"}, single_nbi
      );
      printf("\n");
      FabricBench::Print(
          "EFA Proxy Write - Multi Channel Blocking", nranks, opts.warmup, opts.repeat, total_bw, "GPU -> Queue -> RDMA (sync per op, all EFAs)",
          {"Managed", "Pinned", "GdrQueue"}, multi_blocking
      );
      printf("\n");
      FabricBench::Print(
          "EFA Proxy Write - Multi Channel NBI", nranks, opts.warmup, opts.repeat, total_bw, "GPU -> Queue -> RDMA (batch, all EFAs)",
          {"Managed", "Pinned", "GdrQueue"}, multi_nbi
      );
    }
    return 0;
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Fatal error: {}", e.what());
    return 1;
  }
}
