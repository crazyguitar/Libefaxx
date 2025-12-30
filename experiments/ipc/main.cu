/**
 * @file main.cu
 * @brief IPC benchmark - pure GPU-to-GPU communication via CUDA IPC
 *
 * NVLink Bandwidth and Parallelism Analysis:
 * ------------------------------------------
 * This benchmark demonstrates how thread parallelism affects NVLink utilization
 * in intra-node GPU communication. Results show clear scaling patterns:
 *
 * Configuration Performance (at 1GB):
 * - 1×256 threads:   ~350 GB/s (9.2% peak BW) - insufficient parallelism
 * - 1×1024 threads:  ~353 GB/s (9.2% peak BW) - marginal improvement
 * - 16×256 threads:  ~1868 GB/s (48.8% peak BW) - better SM utilization
 * - 128×256 threads: ~2971 GB/s (77.7% peak BW) - near-optimal parallelism
 *
 * Key Insights:
 * 1. Single thread blocks severely underutilize NVLink bandwidth
 * 2. Increasing blocks (not just threads) is critical for performance
 * 3. 128+ blocks achieve >75% of theoretical NVLink bandwidth
 * 4. Latency decreases significantly with higher parallelism
 */
#include <affinity/affinity.h>
#include <bench/arguments.h>

#include <bench/modules/ipc.cuh>
#include <bench/mpi/fabric.cuh>

/**
 * @brief IPC benchmark test with configurable grid/block dimensions
 * @tparam NumBlocks Number of thread blocks (grid dimension)
 * @tparam NumThreads Number of threads per block
 */
template <unsigned int NumBlocks, unsigned int NumThreads>
struct Test {
  static BenchResult Run(size_t size, const Options& opts, FabricBench& peer, std::string_view name) {
    int rank = peer.mpi.GetWorldRank();
    int local_size = peer.mpi.GetLocalSize();

    // Allocate IPC buffer and exchange handles with local ranks
    auto bufs = peer.AllocIPC<SymmetricDMAMemory>(size);
    auto local_world_ranks = peer.Handshake(bufs, std::true_type{});

    auto& affinity = GPUloc::Get().GetGPUAffinity()[peer.device];
    size_t ipc_bw = affinity.mem_support.nvlink_bw * 8;

    // Benchmark rank 0 writing to each local peer via IPC
    double sum_bw = 0, sum_time = 0;
    for (int t = 1; t < local_size; ++t) {
      int target = local_world_ranks[t];
      using Write = IPCWrite<FabricBench, NumBlocks, NumThreads>;
      using Verify = IPCVerify<FabricBench, NumBlocks, NumThreads>;

      peer.Warmup(bufs, bufs, Write{target}, Verify{target}, opts.warmup);
      auto r = peer.Bench(name, bufs, bufs, Write{target}, Verify{target}, opts.repeat, 0, ipc_bw);
      sum_bw += r.bw_gbps;
      sum_time += r.time_us;
    }

    int npairs = local_size - 1;
    double avg_bw = sum_bw / npairs;
    double link_bw = ipc_bw / 1e9;
    return {size, sum_time / npairs, avg_bw, (link_bw > 0) ? (avg_bw / link_bw) * 100.0 : 0};
  }
};

// Test configurations demonstrating parallelism impact on NVLink utilization
using Test1x256 = Test<1, 256>;    // Low parallelism baseline
using Test1x1024 = Test<1, 1024>;  // Single block, max threads
using Test16x256 = Test<16, 256>;  // Multi-block, moderate
using Test128x256 = Test<128, 256>;

int main(int argc, char* argv[]) {
  auto opts = parse_args(argc, argv);
  auto sizes = generate_sizes(opts);

  FabricBench peer;
  peer.Exchange();
  peer.Connect();

  int rank = peer.mpi.GetWorldRank();
  int local_size = peer.mpi.GetLocalSize();

  if (local_size < 2) {
    if (rank == 0) printf("IPC requires at least 2 ranks per node\n");
    return 1;
  }

  std::vector<std::array<BenchResult, 4>> results;
  for (auto size : sizes) {
    results.push_back({
        Test1x256::Run(size, opts, peer, "IPC(1x256)"),
        Test1x1024::Run(size, opts, peer, "IPC(1x1024)"),
        Test16x256::Run(size, opts, peer, "IPC(16x256)"),
        Test128x256::Run(size, opts, peer, "IPC(128x256)"),
    });
    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (rank == 0) {
    FabricBench::Print(
        "IPC Write Benchmark - Parallelism Impact on NVLink Utilization", local_size, opts.warmup, opts.repeat, 0,
        "Demonstrates how grid/block dimensions affect NVLink bandwidth utilization", {"IPC(1x256)", "IPC(1x1024)", "IPC(16x256)", "IPC(128x256)"},
        results
    );
  }
  return 0;
}
