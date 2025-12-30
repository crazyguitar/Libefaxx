/**
 * @file ipc.cuh
 * @brief GPU-Initiated IPC module for intra-node communication
 */
#pragma once

#include <bench/mpi/fabric.cuh>

/** @brief IPC verify kernel - check data on GPU */
__global__ void IPCVerifyKernel(const int* __restrict__ data, int expected, size_t len, int* __restrict__ result) {
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (size_t idx = tid; idx < len; idx += stride) {
    if (data[idx] != expected + static_cast<int>(idx)) {
      atomicExch(result, 0);
      return;
    }
  }
}

/** @brief IPC write kernel - direct write to remote GPU memory */
__global__ void IPCWriteKernel(void* const* __restrict__ ipc_ptrs, int target, size_t len, int iters) {
  int* remote = static_cast<int*>(ipc_ptrs[target]);
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (int iter = 0; iter < iters; ++iter) {
    for (size_t idx = tid; idx < len; idx += stride) {
      remote[idx] = target + static_cast<int>(idx);
    }
    __threadfence_system();
  }
}

/**
 * @brief IPC Write benchmark functor with configurable parallelism
 */
template <typename Peer, unsigned int NumBlocks = 1, unsigned int NumThreads = 256>
struct IPCWrite {
  int target;

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& a, typename Peer::template Buffers<T>&) {
    int rank = peer.mpi.GetWorldRank();
    if (rank != 0) return;

    auto ctx = a[rank]->GetContext();
    size_t len = a[rank]->Size() / sizeof(int);

    cudaLaunchConfig_t cfg{.gridDim = {NumBlocks, 1, 1}, .blockDim = {NumThreads, 1, 1}, .stream = peer.stream};
    LAUNCH_KERNEL(&cfg, IPCWriteKernel, ctx.ipc_ptrs, target, len, 1);
    CUDA_CHECK(cudaStreamSynchronize(peer.stream));
  }
};

/**
 * @brief IPC Verify functor with configurable parallelism (GPU-side verification)
 */
template <typename Peer, unsigned int NumBlocks = 1, unsigned int NumThreads = 256>
struct IPCVerify {
  int target;

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& bufs) {
    int rank = peer.mpi.GetWorldRank();
    if (rank != target) return;

    auto* data = static_cast<int*>(bufs[rank]->Data());
    size_t len = bufs[rank]->Size() / sizeof(int);

    int* result;
    CUDA_CHECK(cudaMallocManaged(&result, sizeof(int)));
    *result = 1;
    cudaLaunchConfig_t cfg{.gridDim = {NumBlocks, 1, 1}, .blockDim = {NumThreads, 1, 1}, .stream = peer.stream};
    LAUNCH_KERNEL(&cfg, IPCVerifyKernel, data, target, len, result);
    CUDA_CHECK(cudaStreamSynchronize(peer.stream));
    bool ok = (*result == 1);
    CUDA_CHECK(cudaFree(result));
    if (!ok) throw std::runtime_error("IPC verification failed");
  }
};
