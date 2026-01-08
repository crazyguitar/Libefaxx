/**
 * @file shmem.cuh
 * @brief NVSHMEM-like API over EFA/libfabric with CUDA IPC support
 *
 * Provides shmem_* APIs similar to NVSHMEM for GPU-initiated communication.
 * Uses MPI for bootstrap, CUDA IPC for intra-node, and libfabric for inter-node.
 */
#pragma once

#include <bootstrap/mpi/fabric.h>
#include <rdma/fabric/memory.h>
#include <rdma/fabric/request.h>

#include <unordered_map>

namespace shmem::detail {

inline std::unique_ptr<fi::Peer> g_peer;
inline std::unordered_map<void*, std::unique_ptr<fi::SymmetricDMAMemory>> g_allocs;

}  // namespace shmem::detail

/**
 * @brief Initialize SHMEM with MPI-based bootstrap
 */
inline void shmem_init() {
  shmem::detail::g_peer = std::make_unique<fi::Peer>();
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

  // Create symmetric memory
  int target = (rank + 1) % world;
  int source = (rank - 1 + world) % world;
  auto mem = std::make_unique<fi::SymmetricDMAMemory>(peer.channels[target], size, world, peer.device);

  // Exchange IPC handles among local ranks
  peer.Handshake(mem);

  // Exchange RDMA keys (always do this for ring pattern)
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
[[nodiscard]] inline auto shmem_ctx(void* ptr) noexcept { return shmem::detail::g_allocs.at(ptr)->GetContext(); }

/**
 * @brief Global barrier across all PEs
 */
inline void shmem_barrier_all() noexcept { MPI_Barrier(MPI_COMM_WORLD); }

/**
 * @brief Get symmetric memory object for a pointer
 * @return Reference to SymmetricDMAMemory
 */
[[nodiscard]] inline fi::SymmetricDMAMemory& shmem_mem(void* ptr) noexcept { return *shmem::detail::g_allocs.at(ptr); }

/**
 * @brief Get peer object
 * @return Reference to fi::Peer
 */
[[nodiscard]] inline fi::Peer& shmem_peer() noexcept { return *shmem::detail::g_peer; }

#ifdef __CUDACC__

/**
 * @brief Device fence - ensures memory visibility across system
 */
__device__ __forceinline__ void shmem_fence() { fi::Fence(); }

/**
 * @brief Device quiet - wait for all outstanding operations to complete
 * @param ctx Device context from shmem_ctx
 */
template <typename Ctx>
__device__ __forceinline__ void shmem_quiet(Ctx ctx) {
  fi::Quiet(ctx.posted, ctx.completed);
}

/**
 * @brief Template put a single value to remote PE (non-blocking)
 * @tparam T Value type
 * @param ctx Device context from shmem_ctx
 * @param dest Destination address in symmetric memory
 * @param value Value to write
 * @param pe Target PE index
 */
template <typename Ctx, typename T>
__device__ __forceinline__ void shmem_p_nbi(const Ctx ctx, T* __restrict__ dest, const T value, const int pe) {
  if (ctx.ipc_ptrs[pe]) {
    // Intra-node: Direct IPC memory access
    static_cast<T*>(ctx.ipc_ptrs[pe])[0] = value;
  } else {
    // Inter-node: RDMA via queue
    *dest = value;
    __threadfence_system();
    fi::DeviceRequest req{
        .type = static_cast<uint64_t>(fi::DeviceRequestType::kPut),
        .rank = static_cast<uint64_t>(pe),
        .size = sizeof(T),
        .addr = reinterpret_cast<uint64_t>(dest),
        .imm = 1
    };
    while (!ctx.queue->Push(req)) __threadfence_system();
    reinterpret_cast<cuda::std::atomic<uint64_t>*>(ctx.posted)->fetch_add(1ULL, cuda::std::memory_order_relaxed);
  }
  __threadfence_system();
}

/**
 * @brief Template put a single value to remote PE (blocking)
 * @tparam T Value type
 * @param ctx Device context from shmem_ctx
 * @param dest Destination address in symmetric memory
 * @param value Value to write
 * @param pe Target PE index
 */
template <typename Ctx, typename T>
__device__ __forceinline__ void shmem_p(Ctx ctx, T* __restrict__ dest, T value, int pe) {
  shmem_p_nbi(ctx, dest, value, pe);
  shmem_quiet(ctx);
}

// Type-specific wrappers via macro
#define SHMEM_TYPE_P_IMPL(NAME, TYPE)                                                             \
  template <typename Ctx>                                                                         \
  __device__ __forceinline__ void shmem_##NAME##_p(Ctx ctx, TYPE* dest, TYPE value, int pe) {     \
    shmem_p(ctx, dest, value, pe);                                                                \
  }                                                                                               \
  template <typename Ctx>                                                                         \
  __device__ __forceinline__ void shmem_##NAME##_p_nbi(Ctx ctx, TYPE* dest, TYPE value, int pe) { \
    shmem_p_nbi(ctx, dest, value, pe);                                                            \
  }

SHMEM_TYPE_P_IMPL(char, char)
SHMEM_TYPE_P_IMPL(schar, signed char)
SHMEM_TYPE_P_IMPL(short, short)
SHMEM_TYPE_P_IMPL(int, int)
SHMEM_TYPE_P_IMPL(long, long)
SHMEM_TYPE_P_IMPL(longlong, long long)
SHMEM_TYPE_P_IMPL(uchar, unsigned char)
SHMEM_TYPE_P_IMPL(ushort, unsigned short)
SHMEM_TYPE_P_IMPL(uint, unsigned int)
SHMEM_TYPE_P_IMPL(ulong, unsigned long)
SHMEM_TYPE_P_IMPL(ulonglong, unsigned long long)
SHMEM_TYPE_P_IMPL(int8, int8_t)
SHMEM_TYPE_P_IMPL(int16, int16_t)
SHMEM_TYPE_P_IMPL(int32, int32_t)
SHMEM_TYPE_P_IMPL(int64, int64_t)
SHMEM_TYPE_P_IMPL(uint8, uint8_t)
SHMEM_TYPE_P_IMPL(uint16, uint16_t)
SHMEM_TYPE_P_IMPL(uint32, uint32_t)
SHMEM_TYPE_P_IMPL(uint64, uint64_t)
SHMEM_TYPE_P_IMPL(size, size_t)
SHMEM_TYPE_P_IMPL(ptrdiff, ptrdiff_t)
SHMEM_TYPE_P_IMPL(float, float)
SHMEM_TYPE_P_IMPL(double, double)

#undef SHMEM_TYPE_P_IMPL

#endif  // __CUDACC__
