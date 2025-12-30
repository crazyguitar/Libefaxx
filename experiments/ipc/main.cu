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
#include <io/progress.h>

#include <bench/modules/ipc.cuh>
#include <bench/mpi/fabric.cuh>

/**
 * @brief IPC benchmark with configurable parallelism
 * @tparam BufType Buffer type (SymmetricDMAMemory)
 * @tparam NumBlocks Number of thread blocks (grid dimension)
 * @tparam NumThreads Number of threads per block
 */
template <typename BufType, unsigned int NumBlocks, unsigned int NumThreads>
struct TestConfig {
  static BenchResult Run(size_t size, const Options& opts, FabricBench& peer, std::string_view name = "IPCWrite") {
    int rank = peer.mpi.GetWorldRank();
    int world = peer.mpi.GetWorldSize();
    int local_size = peer.mpi.GetLocalSize();
    int local_rank = peer.mpi.GetLocalRank();

    int next = (rank + 1) % world;
    auto mem = std::make_unique<BufType>(peer.channels[next], size, world, peer.device);

    std::vector<cudaIpcMemHandle_t> all_handles(local_size);
    std::vector<int> local_world_ranks(local_size);
    cudaIpcMemHandle_t local_handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&local_handle, mem->Data()));

    MPI_Comm local = peer.mpi.GetLocalComm();
    MPI_Allgather(&rank, 1, MPI_INT, local_world_ranks.data(), 1, MPI_INT, local);
    MPI_Allgather(&local_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, all_handles.data(), sizeof(cudaIpcMemHandle_t), MPI_BYTE, local);
    mem->OpenIPCHandles(all_handles, local_world_ranks, local_rank);

    auto ctx = mem->GetContext();
    size_t len = size / sizeof(int);

    auto& affinity = GPUloc::Get().GetGPUAffinity()[peer.device];
    size_t ipc_bw_bps = affinity.mem_support.nvlink_bw * 8;

    cudaLaunchConfig_t cfg{.gridDim = {NumBlocks, 1, 1}, .blockDim = {NumThreads, 1, 1}, .stream = peer.stream};

    double sum_bw = 0, sum_time = 0;
    for (int t = 1; t < local_size; ++t) {
      int target_rank = local_world_ranks[t];
      Progress progress(opts.repeat, ipc_bw_bps, name);
      MPI_Barrier(MPI_COMM_WORLD);
      auto start = std::chrono::high_resolution_clock::now();

      if (rank == 0) {
        for (int i = 0; i < opts.repeat; ++i) {
          LAUNCH_KERNEL(&cfg, IPCWriteKernel, ctx.ipc_ptrs, target_rank, len, 1);
          if ((i + 1) % Progress::kPrintFreq == 0) {
            CUDA_CHECK(cudaStreamSynchronize(peer.stream));
            progress.Print(std::chrono::high_resolution_clock::now(), size, i + 1);
          }
        }
        CUDA_CHECK(cudaStreamSynchronize(peer.stream));
      }

      MPI_Barrier(MPI_COMM_WORLD);
      auto end = std::chrono::high_resolution_clock::now();

      // GPU-side verification: target rank checks received data
      if (rank == target_rank) {
        int* result;
        CUDA_CHECK(cudaMallocManaged(&result, sizeof(int)));
        *result = 1;
        LAUNCH_KERNEL(&cfg, IPCVerifyKernel, static_cast<int*>(mem->Data()), target_rank, len, result);
        CUDA_CHECK(cudaStreamSynchronize(peer.stream));
        bool ok = (*result == 1);
        CUDA_CHECK(cudaFree(result));
        if (!ok) throw std::runtime_error("IPC verification failed");
      }

      double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
      double avg_us = elapsed_us / opts.repeat;
      double bw_gbps = (size * 8.0) / (avg_us * 1000.0);
      sum_bw += bw_gbps;
      sum_time += avg_us;
    }

    int npairs = local_size - 1;
    double avg_bw = sum_bw / npairs;
    double link_bw = ipc_bw_bps / 1e9;
    double bus_bw = (link_bw > 0) ? (avg_bw / link_bw) * 100.0 : 0;
    return {size, sum_time / npairs, avg_bw, bus_bw};
  }
};

// Test configurations demonstrating parallelism impact on NVLink utilization
using Test1x256 = TestConfig<SymmetricDMAMemory, 1, 256>;      // Low parallelism baseline
using Test1x1024 = TestConfig<SymmetricDMAMemory, 1, 1024>;    // Single block, max threads
using Test16x256 = TestConfig<SymmetricDMAMemory, 16, 256>;    // Multi-block, moderate
using Test128x256 = TestConfig<SymmetricDMAMemory, 128, 256>;  // High parallelism

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
