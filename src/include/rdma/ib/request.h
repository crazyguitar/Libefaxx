/**
 * @file request.h
 * @brief Device request structures for GPU-CPU communication (ibverbs version)
 */
#pragma once
#include <stdint.h>

#include <algorithm>
#include <tuple>
#include <vector>

namespace ib {

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
  void* const* __restrict__ ipc_ptrs;
};

#ifdef __CUDACC__
__device__ __forceinline__ void Fence() { __threadfence_system(); }

__device__ __forceinline__ void Quiet(uint64_t* posted, uint64_t* completed) {
  uint64_t expected = *posted;
  while (*completed < expected) __threadfence_system();
  __threadfence_system();
}
#endif

}  // namespace ib
