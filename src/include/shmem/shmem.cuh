/**
 * @file shmem.cuh
 * @brief NVSHMEM-like API over EFA/libfabric
 *
 * Provides shmem_* APIs similar to NVSHMEM for GPU-initiated communication.
 * Uses MPI for bootstrap and libfabric for RDMA transport.
 */
#pragma once

#include <bootstrap/mpi/fabric.h>
#include <rdma/fabric/memory.h>
#include <rdma/fabric/request.h>

#include <unordered_map>

namespace shmem::detail {

inline std::unique_ptr<Peer> g_peer;
inline std::unordered_map<void*, std::unique_ptr<SymmetricDMAMemory>> g_allocs;

}  // namespace shmem::detail

/**
 * @brief Initialize SHMEM with MPI-based bootstrap
 */
inline void shmem_init() {
  shmem::detail::g_peer = std::make_unique<Peer>();
  shmem::detail::g_peer->Exchange();
  shmem::detail::g_peer->Connect();
}

/**
 * @brief Finalize SHMEM and release resources
 */
inline void shmem_finalize() {
  shmem::detail::g_allocs.clear();
  shmem::detail::g_peer.reset();
}

/**
 * @brief Get current PE (processing element) rank
 * @return Current PE index
 */
[[nodiscard]] inline int shmem_my_pe() noexcept { return shmem::detail::g_peer->mpi.GetWorldRank(); }

/**
 * @brief Get total number of PEs
 * @return Number of PEs in the job
 */
[[nodiscard]] inline int shmem_n_pes() noexcept { return shmem::detail::g_peer->mpi.GetWorldSize(); }

/**
 * @brief Allocate symmetric memory on the symmetric heap
 * @param size Allocation size in bytes
 * @return Pointer to allocated symmetric memory
 */
[[nodiscard]] inline void* shmem_malloc(size_t size) {
  auto& peer = *shmem::detail::g_peer;
  int rank = peer.mpi.GetWorldRank();
  int world = peer.mpi.GetWorldSize();

  int target = (rank + 1) % world;
  int source = (rank - 1 + world) % world;
  auto mem = std::make_unique<SymmetricDMAMemory>(peer.channels[target], size, world, peer.device);

  auto local_rma = mem->GetLocalRmaIovs();
  size_t sz = local_rma.size() * sizeof(fi_rma_iov);
  std::vector<fi_rma_iov> target_iovs(local_rma.size());
  MPI_Sendrecv(local_rma.data(), sz, MPI_BYTE, source, 0, target_iovs.data(), sz, MPI_BYTE, target, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  mem->SetRemoteRmaIovs(target, target_iovs);

  void* ptr = mem->Data();
  shmem::detail::g_allocs[ptr] = std::move(mem);
  return ptr;
}

/**
 * @brief Free symmetric memory
 * @param ptr Pointer returned by shmem_malloc
 */
inline void shmem_free(void* ptr) { shmem::detail::g_allocs.erase(ptr); }

/**
 * @brief Get device context for a symmetric allocation
 * @param ptr Pointer returned by shmem_malloc
 * @return DeviceContext for GPU kernel access
 */
[[nodiscard]] inline DeviceContext shmem_ctx(void* ptr) noexcept { return shmem::detail::g_allocs.at(ptr)->GetContext(); }

/**
 * @brief Global barrier across all PEs
 */
inline void shmem_barrier_all() noexcept { MPI_Barrier(MPI_COMM_WORLD); }

/**
 * @brief Get symmetric memory object for a pointer
 * @param ptr Pointer returned by shmem_malloc
 * @return Reference to SymmetricDMAMemory
 */
[[nodiscard]] inline SymmetricDMAMemory& shmem_mem(void* ptr) noexcept { return *shmem::detail::g_allocs.at(ptr); }

/**
 * @brief Get peer object
 * @return Reference to Peer
 */
[[nodiscard]] inline Peer& shmem_peer() noexcept { return *shmem::detail::g_peer; }

#ifdef __CUDACC__

/**
 * @brief Device fence - ensures memory visibility across system
 */
__device__ __forceinline__ void shmem_fence() { Fence(); }

/**
 * @brief Device quiet - wait for all outstanding operations to complete
 * @param ctx Device context from shmem_ctx
 */
__device__ __forceinline__ void shmem_quiet(DeviceContext ctx) { Quiet(ctx.posted, ctx.completed); }

/**
 * @brief Put a single int to remote PE (non-blocking)
 * @param ctx Device context from shmem_ctx
 * @param dest Destination address in symmetric memory
 * @param value Value to write
 * @param pe Target PE index
 */
__device__ __forceinline__ void shmem_int_p_nbi(DeviceContext ctx, int* __restrict__ dest, int value, int pe) {
  *dest = value;
  __threadfence_system();
  if (threadIdx.x == 0) {
    DeviceRequest req{
        .type = static_cast<uint64_t>(DeviceRequestType::kPut),
        .rank = static_cast<uint64_t>(pe),
        .size = sizeof(int),
        .addr = reinterpret_cast<uint64_t>(dest),
        .imm = 1
    };
    while (!ctx.queue->Push(req)) __threadfence_system();
    reinterpret_cast<cuda::std::atomic<uint64_t>*>(ctx.posted)->fetch_add(1ULL, cuda::std::memory_order_relaxed);
    shmem_fence();
  }
  __syncthreads();
}

/**
 * @brief Put a single int to remote PE (blocking)
 * @param ctx Device context from shmem_ctx
 * @param dest Destination address in symmetric memory
 * @param value Value to write
 * @param pe Target PE index
 */
__device__ __forceinline__ void shmem_int_p(DeviceContext ctx, int* dest, int value, int pe) {
  shmem_int_p_nbi(ctx, dest, value, pe);
  shmem_quiet(ctx);
}

#endif  // __CUDACC__
