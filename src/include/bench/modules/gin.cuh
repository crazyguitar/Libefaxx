/**
 * @file gin.cuh
 * @brief GPU-Initiated Networking (GIN) module for proxy-based RDMA
 */
#pragma once

#include <io/awaiter.h>
#include <io/runner.h>
#include <rdma/fabric/selector.h>

#include <bench/mpi/fabric.cuh>

/** @brief Write data and push RDMA request to queue (with Quiet per write) */
__device__ __forceinline__ void DeviceWrite(DeviceContext ctx, int target, size_t len, int* __restrict__ data, uint64_t imm) {
  for (size_t idx = threadIdx.x; idx < len; idx += blockDim.x) data[idx] = target + static_cast<int>(idx);
  __threadfence_system();
  if (threadIdx.x == 0) {
    DeviceRequest req{
        .type = static_cast<uint64_t>(DeviceRequestType::kPut),
        .rank = static_cast<uint64_t>(target),
        .size = len * sizeof(int),
        .addr = reinterpret_cast<uint64_t>(data),
        .imm = imm
    };
    while (!ctx.queue->Push(req)) __threadfence_system();
    atomicAdd(reinterpret_cast<unsigned long long*>(ctx.posted), 1ULL);
    Fence();
    Quiet(ctx.posted, ctx.completed);
  }
  __syncthreads();
}

/** @brief NBI (non-blocking interface) write - push without waiting, busy-wait on queue full */
__device__ __forceinline__ void DeviceWriteNBI(DeviceContext ctx, int target, size_t len, int* __restrict__ data, uint64_t imm) {
  for (size_t idx = threadIdx.x; idx < len; idx += blockDim.x) data[idx] = target + static_cast<int>(idx);
  __threadfence_system();
  if (threadIdx.x == 0) {
    DeviceRequest req{
        .type = static_cast<uint64_t>(DeviceRequestType::kPut),
        .rank = static_cast<uint64_t>(target),
        .size = len * sizeof(int),
        .addr = reinterpret_cast<uint64_t>(data),
        .imm = imm
    };
    while (!ctx.queue->Push(req)) __threadfence_system();
    atomicAdd(reinterpret_cast<unsigned long long*>(ctx.posted), 1ULL);
    Fence();
  }
  __syncthreads();
}

/** @brief Wait for expected number of completions */
__device__ __forceinline__ void DeviceWait(DeviceContext ctx, int expected) {
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
__global__ void ProxyWriteKernel(DeviceContext ctx, int target, size_t len, int* __restrict__ data, uint64_t imm, int iters) {
  for (int i = 0; i < iters; ++i) DeviceWrite(ctx, target, len, data, imm);
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
__global__ void ProxyWriteNBIKernel(DeviceContext ctx, int target, size_t len, int* __restrict__ data, uint64_t imm, int iters) {
  for (int i = 0; i < iters; ++i) DeviceWriteNBI(ctx, target, len, data, imm);
  if (threadIdx.x == 0) Quiet(ctx.posted, ctx.completed);
  __syncthreads();
}

__global__ void ProxyWaitKernel(DeviceContext ctx, int rank, size_t len, int* __restrict__ data, int iters, int* __restrict__ result) {
  DeviceWait(ctx, iters);
  if (threadIdx.x == 0) *result = DeviceVerify(rank, len, data) ? 1 : 0;
}

/** @brief Kernel launcher tags */
struct KernelBlocking {
  static void Launch(cudaLaunchConfig_t& cfg, DeviceContext ctx, int target, size_t len, int* data, uint64_t imm, int iters) {
    LAUNCH_KERNEL(&cfg, ProxyWriteKernel, ctx, target, len, data, imm, iters);
  }
};
struct KernelNBI {
  static void Launch(cudaLaunchConfig_t& cfg, DeviceContext ctx, int target, size_t len, int* data, uint64_t imm, int iters) {
    LAUNCH_KERNEL(&cfg, ProxyWriteNBIKernel, ctx, target, len, data, imm, iters);
  }
};

/**
 * @brief Proxy write: GPU kernel pushes requests, CPU proxy executes RDMA
 * @tparam Launcher KernelBlocking (Quiet per write) or KernelNBI (Quiet at end)
 */
template <typename Peer, bool MultiChannel, typename Launcher = KernelBlocking>
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
    auto* data = reinterpret_cast<int*>(write[target]->Data());
    size_t size = write[target]->Size(), len = size / sizeof(int);

    cudaLaunchConfig_t cfg{.gridDim = {1, 1, 1}, .blockDim = {256, 1, 1}, .stream = peer.stream};
    Launcher::Launch(cfg, ctx, target, len, data, 1ULL, iters);

    Progress progress(iters, MultiChannel ? peer.GetTotalBandwidth() : peer.GetBandwidth(0));
    for (auto& efa : peer.efas) IO::Get().Join<FabricSelector>(efa);
    ::Run([&]() -> Coro<> {
      for (int done = 0; done < iters;) {
        DeviceRequest req;
        if (ctx.queue->Pop(req)) {
          co_await Write(write[target], target, req.imm);
          write[target]->Complete();
          if (++done % Progress::kPrintFreq == 0) progress.Print(std::chrono::high_resolution_clock::now(), size, done);
        }
        co_await YieldAwaiter{};
      }
      for (auto& efa : peer.efas) IO::Get().Quit<FabricSelector>(efa);
    }());
    CUDA_CHECK(cudaStreamSynchronize(peer.stream));
  }
};

/**
 * @brief Proxy read: wait for RDMA completions and signal GPU
 */
template <typename Peer, bool MultiChannel>
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
    auto* data = reinterpret_cast<int*>(read[0]->Data());
    size_t len = read[0]->Size() / sizeof(int);
    int* result;
    CUDA_CHECK(cudaMallocManaged(&result, sizeof(int)));
    *result = 1;

    cudaLaunchConfig_t cfg{.gridDim = {1, 1, 1}, .blockDim = {256, 1, 1}, .stream = peer.stream};
    LAUNCH_KERNEL(&cfg, ProxyWaitKernel, ctx, peer.mpi.GetWorldRank(), len, data, iters, result);

    for (auto& efa : peer.efas) IO::Get().Join<FabricSelector>(efa);
    ::Run([&]() -> Coro<> {
      for (int i = 0; i < iters; ++i) {
        co_await Wait(read[0], 1);
        read[0]->Complete();
      }
      for (auto& efa : peer.efas) IO::Get().Quit<FabricSelector>(efa);
    }());

    CUDA_CHECK(cudaStreamSynchronize(peer.stream));
    bool ok = (*result == 1);
    CUDA_CHECK(cudaFree(result));
    if (!ok) throw std::runtime_error(fmt::format("Verification failed on rank {}", peer.mpi.GetWorldRank()));
  }
};
