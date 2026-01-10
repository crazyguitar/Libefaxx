/**
 * @file proxy.cuh
 * @brief GPU-Initiated Networking (GIN) module for proxy-based RDMA
 *
 * This module implements GPU-initiated RDMA writes via a CPU proxy coroutine.
 * The GPU kernel pushes requests to a queue, and the CPU proxy executes
 * the actual RDMA operations. This enables GPU-driven communication patterns.
 *
 * ## Architecture Overview
 * ```
 * ┌────────────────────────────────────────────────────────────────┐
 * │                         GPU                                    │
 * │  ┌─────────────────────────────────────────────────────────┐   │
 * │  │  Kernel                                                 │   │
 * │  │  ┌─────────┐    ┌─────────┐    ┌─────────┐              │   │
 * │  │  │ Write   │───►│ Push to │───►│ Fence + │              │   │
 * │  │  │ Data    │    │ Queue   │    │ Quiet   │              │   │
 * │  │  └─────────┘    └────┬────┘    └─────────┘              │   │
 * │  └──────────────────────┼──────────────────────────────────┘   │
 * └─────────────────────────┼──────────────────────────────────────┘
 *                           │ MPSC Queue (GPU→CPU)
 *                           ▼
 * ┌────────────────────────────────────────────────────────────────┐
 * │                         CPU                                    │
 * │  ┌─────────────────────────────────────────────────────────┐   │
 * │  │  Proxy Coroutine                                        │   │
 * │  │  ┌─────────┐    ┌─────────┐    ┌─────────┐              │   │
 * │  │  │ Pop     │───►│ RDMA    │───►│Complete │              │   │
 * │  │  │ Request │    │ Write   │    │ Counter │              │   │
 * │  │  └─────────┘    └────┬────┘    └─────────┘              │   │
 * │  └──────────────────────┼──────────────────────────────────┘   │
 * └─────────────────────────┼──────────────────────────────────────┘
 *                           │ EFA RDMA
 *                           ▼
 *                      ┌─────────┐
 *                      │ Remote  │
 *                      │ Memory  │
 *                      └─────────┘
 * ```
 *
 * ## Synchronization
 * - `posted`: Atomic counter incremented by GPU after each push
 * - `completed`: Atomic counter incremented by CPU after each RDMA
 * - `Fence()`: __threadfence_system() for GPU-CPU visibility
 * - `Quiet()`: Spin-wait until completed >= posted
 */
#pragma once

#include <io/awaiter.h>
#include <io/progress.h>
#include <io/runner.h>
#include <rdma/request.h>

/**
 * @brief Write data and push RDMA request to queue (with Quiet per write)
 *
 * Blocking mode: waits for completion after each write.
 */
template <typename Ctx, typename Request>
__device__ __forceinline__ void DeviceWrite(Ctx ctx, int target, size_t len, int* __restrict__ data, uint64_t imm) {
  for (size_t idx = threadIdx.x; idx < len; idx += blockDim.x) data[idx] = target + static_cast<int>(idx);
  __threadfence_system();
  if (threadIdx.x == 0) {
    Request req{
        .type = static_cast<uint64_t>(rdma::DeviceRequestType::kPut),
        .rank = static_cast<uint64_t>(target),
        .size = len * sizeof(int),
        .addr = reinterpret_cast<uint64_t>(data),
        .imm = imm
    };
    while (!ctx.queue->Push(req)) __threadfence_system();
    atomicAdd(reinterpret_cast<unsigned long long*>(ctx.posted), 1ULL);
    rdma::Fence();
    rdma::Quiet(ctx.posted, ctx.completed);
  }
  __syncthreads();
}

/** @brief NBI (non-blocking interface) write - push without waiting, busy-wait on queue full */
template <typename Ctx, typename Request>
__device__ __forceinline__ void DeviceWriteNBI(Ctx ctx, int target, size_t len, int* __restrict__ data, uint64_t imm) {
  for (size_t idx = threadIdx.x; idx < len; idx += blockDim.x) data[idx] = target + static_cast<int>(idx);
  __threadfence_system();
  if (threadIdx.x == 0) {
    Request req{
        .type = static_cast<uint64_t>(rdma::DeviceRequestType::kPut),
        .rank = static_cast<uint64_t>(target),
        .size = len * sizeof(int),
        .addr = reinterpret_cast<uint64_t>(data),
        .imm = imm
    };
    while (!ctx.queue->Push(req)) __threadfence_system();
    atomicAdd(reinterpret_cast<unsigned long long*>(ctx.posted), 1ULL);
    rdma::Fence();
  }
  __syncthreads();
}

/** @brief Wait for expected number of completions */
template <typename Ctx>
__device__ __forceinline__ void DeviceWait(Ctx ctx, int expected) {
  if (threadIdx.x == 0) {
    while (*ctx.completed < expected) __threadfence_system();
  }
  __syncthreads();
}

/** @brief Verify received data matches expected pattern */
__device__ __forceinline__ bool DeviceVerify(int expected, size_t len, const int* __restrict__ data) {
  for (size_t idx = threadIdx.x; idx < len; idx += blockDim.x) {
    if (data[idx] != expected + static_cast<int>(idx)) return false;
  }
  return true;
}

/**
 * @brief Blocking kernel - wait for each completion
 *
 *   GPU          CPU Proxy         Network
 *    |              |                 |
 *    |---Push[0]--->|                 |
 *    |              |----RDMA[0]----->|
 *    |<--Complete---|                 |
 *    |   (Quiet)    |                 |
 *    |              |                 |
 *    |---Push[1]--->|                 |   <- bubble: GPU idle
 *    |              |----RDMA[1]----->|
 *    |<--Complete---|                 |
 *    |   (Quiet)    |                 |
 *    v              v                 v
 */
template <typename Ctx, typename Request>
__global__ void ProxyWriteKernel(Ctx ctx, int target, size_t len, int* __restrict__ data, uint64_t imm, int iters) {
  for (int i = 0; i < iters; ++i) DeviceWrite<Ctx, Request>(ctx, target, len, data, imm);
}

/**
 * @brief NBI (Non-Blocking Interface) kernel - pipelined writes with single Quiet at end
 *
 *   GPU          CPU Proxy         Network
 *    |              |                 |
 *    |---Push[0]--->|                 |
 *    |---Push[1]--->|----RDMA[0]----->|
 *    |---Push[2]--->|----RDMA[1]----->|
 *    |      ...     |----RDMA[2]----->|
 *    |   (Quiet)    |      ...        |
 *    |<--Complete---|                 |
 *    v              v                 v
 *
 * NBI eliminates per-operation wait bubbles by overlapping push and RDMA.
 */
template <typename Ctx, typename Request>
__global__ void ProxyWriteNBIKernel(Ctx ctx, int target, size_t len, int* __restrict__ data, uint64_t imm, int iters) {
  for (int i = 0; i < iters; ++i) DeviceWriteNBI<Ctx, Request>(ctx, target, len, data, imm);
  if (threadIdx.x == 0) rdma::Quiet(ctx.posted, ctx.completed);
  __syncthreads();
}

template <typename Ctx>
__global__ void ProxyWaitKernel(Ctx ctx, int rank, size_t len, int* __restrict__ data, int iters, int* __restrict__ result) {
  DeviceWait(ctx, iters);
  if (threadIdx.x == 0) *result = DeviceVerify(rank, len, data) ? 1 : 0;
}

/** @brief Kernel launcher tags */
template <typename Ctx, typename Request>
struct KernelBlocking {
  static void Launch(cudaLaunchConfig_t& cfg, Ctx ctx, int target, size_t len, int* data, uint64_t imm, int iters) {
    LAUNCH_KERNEL(&cfg, (ProxyWriteKernel<Ctx, Request>), ctx, target, len, data, imm, iters);
  }
};
template <typename Ctx, typename Request>
struct KernelNBI {
  static void Launch(cudaLaunchConfig_t& cfg, Ctx ctx, int target, size_t len, int* data, uint64_t imm, int iters) {
    LAUNCH_KERNEL(&cfg, (ProxyWriteNBIKernel<Ctx, Request>), ctx, target, len, data, imm, iters);
  }
};

/**
 * @brief Proxy write: GPU kernel pushes requests, CPU proxy executes RDMA
 * @tparam Launcher KernelBlocking (Quiet per write) or KernelNBI (Quiet at end)
 */
template <typename Peer, typename Selector, bool MultiChannel, template <typename, typename> class Launcher = KernelBlocking>
struct ProxyWrite {
  int iters, target;

  template <typename Buf>
  static Coro<ssize_t> Write(Buf& buf, int rank, uint64_t imm) {
    if constexpr (MultiChannel)
      return buf->Writeall(rank, imm);
    else
      return buf->Writeall(rank, imm, 0);
  }

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& write) {
    if (peer.mpi.GetWorldRank() != 0) return;

    auto ctx = write[target]->GetContext();
    using CtxType = decltype(ctx);
    using Request = rdma::DeviceRequest;
    auto* data = reinterpret_cast<int*>(write[target]->Data());
    size_t size = write[target]->Size(), len = size / sizeof(int);

    cudaLaunchConfig_t cfg{.gridDim = {1, 1, 1}, .blockDim = {256, 1, 1}, .stream = peer.stream};
    Launcher<CtxType, Request>::Launch(cfg, ctx, target, len, data, 1ULL, iters);

    Progress progress(iters, MultiChannel ? peer.GetTotalBandwidth() : peer.GetBandwidth(0));
    for (auto& efa : peer.efas) IO::Get().Join<Selector>(efa);
    ::Run([&]() -> Coro<> {
      for (int done = 0; done < iters;) {
        Request req;
        if (ctx.queue->Pop(req)) {
          co_await Write(write[target], target, req.imm);
          write[target]->Complete();
          if (++done % Progress::kPrintFreq == 0) progress.Print(std::chrono::high_resolution_clock::now(), size, done);
        }
        co_await YieldAwaiter{};
      }
      for (auto& efa : peer.efas) IO::Get().Quit<Selector>(efa);
    }());
    CUDA_CHECK(cudaStreamSynchronize(peer.stream));
  }
};

/**
 * @brief Proxy read: wait for RDMA completions and signal GPU
 */
template <typename Peer, typename Selector, bool MultiChannel>
struct ProxyRead {
  int iters;

  template <typename Buf>
  static Coro<> Wait(Buf& buf, uint64_t imm) {
    if constexpr (MultiChannel)
      return buf->WaitallImmdata(imm);
    else
      return buf->WaitImmdata(imm);
  }

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& read) {
    if (peer.mpi.GetWorldRank() == 0) return;

    auto ctx = read[0]->GetContext();
    using CtxType = decltype(ctx);
    auto* data = reinterpret_cast<int*>(read[0]->Data());
    size_t len = read[0]->Size() / sizeof(int);
    int* result;
    CUDA_CHECK(cudaMallocManaged(&result, sizeof(int)));
    *result = 1;

    cudaLaunchConfig_t cfg{.gridDim = {1, 1, 1}, .blockDim = {256, 1, 1}, .stream = peer.stream};
    LAUNCH_KERNEL(&cfg, ProxyWaitKernel<CtxType>, ctx, peer.mpi.GetWorldRank(), len, data, iters, result);

    for (auto& efa : peer.efas) IO::Get().Join<Selector>(efa);
    ::Run([&]() -> Coro<> {
      for (int i = 0; i < iters; ++i) {
        co_await Wait(read[0], 1);
        read[0]->Complete();
      }
      for (auto& efa : peer.efas) IO::Get().Quit<Selector>(efa);
    }());

    CUDA_CHECK(cudaStreamSynchronize(peer.stream));
    bool ok = (*result == 1);
    CUDA_CHECK(cudaFree(result));
    if (!ok) throw std::runtime_error(fmt::format("Verification failed on rank {}", peer.mpi.GetWorldRank()));
  }
};
