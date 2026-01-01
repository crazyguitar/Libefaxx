#pragma once
#include <stdint.h>

#include <algorithm>
#include <tuple>
#include <vector>

enum class DeviceRequestType : uint32_t { kPut = 0, kGet = 1, kFence = 2, kCount = 3 };

struct DeviceRequest {
  uint64_t type;
  uint64_t rank;
  uint64_t size;
  uint64_t addr;
  uint64_t imm;
};

/** @brief Context for GPU kernel to access queue and counters */
template <typename Q>
struct DeviceContext {
  Q* __restrict__ queue;
  uint64_t* __restrict__ posted;
  uint64_t* __restrict__ completed;
  void* const* __restrict__ ipc_ptrs;  ///< Array of IPC pointers indexed by rank (nullptr for remote ranks)
};

/** @brief Merge contiguous device requests by (rank, type, addr) */
inline std::vector<DeviceRequest> Merge(std::vector<DeviceRequest>& reqs) {
  if (reqs.empty()) return {};
  auto cmp = [](const auto& a, const auto& b) { return std::tie(a.rank, a.type, a.addr) < std::tie(b.rank, b.type, b.addr); };
  std::sort(reqs.begin(), reqs.end(), cmp);
  std::vector<DeviceRequest> result;
  result.reserve(reqs.size());
  result.push_back(std::move(reqs[0]));
  for (size_t i = 1; i < reqs.size(); ++i) {
    auto& last = result.back();
    auto& cur = reqs[i];
    if (cur.rank == last.rank && cur.type == last.type && cur.addr == last.addr + last.size) {
      last.size += cur.size;
    } else {
      result.push_back(std::move(cur));
    }
  }
  reqs.clear();
  return result;
}

#ifdef __CUDACC__
/**
 * @brief GPU-side memory fence - ensures visibility across system
 */
__device__ __forceinline__ void Fence() { __threadfence_system(); }

/**
 * @brief GPU-side quiet - wait for all posted operations to complete
 * @param posted Pointer to posted counter
 * @param completed Pointer to completed counter
 *
 * Spins until completed >= posted, ensuring all outstanding operations finish.
 */
__device__ __forceinline__ void Quiet(uint64_t* posted, uint64_t* completed) {
  uint64_t expected = *posted;
  while (*completed < expected) __threadfence_system();
  __threadfence_system();
}
#endif
