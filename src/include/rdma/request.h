/**
 * @file request.h
 * @brief Shared device request structures for GPU-CPU communication
 */
#pragma once
#include <cstdint>

namespace rdma {

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

}  // namespace rdma

// Backend-specific aliases for backward compatibility
namespace fi {
using DeviceRequestType = rdma::DeviceRequestType;
using DeviceRequest = rdma::DeviceRequest;
template <typename Q>
using DeviceContext = rdma::DeviceContext<Q>;
#ifdef __CUDACC__
using rdma::Fence;
using rdma::Quiet;
#endif
}  // namespace fi

namespace ib {
using DeviceRequestType = rdma::DeviceRequestType;
using DeviceRequest = rdma::DeviceRequest;
template <typename Q>
using DeviceContext = rdma::DeviceContext<Q>;
#ifdef __CUDACC__
using rdma::Fence;
using rdma::Quiet;
#endif
}  // namespace ib
