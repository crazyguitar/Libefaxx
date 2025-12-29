/**
 * @file ipc.cuh
 * @brief GPU-Initiated IPC module for intra-node communication
 */
#pragma once

#include <bench/mpi/fabric.cuh>

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

/** @brief IPC read kernel - direct read from remote GPU memory */
__global__ void IPCReadKernel(void* const* __restrict__ ipc_ptrs, int source, size_t len, int* __restrict__ local, int iters) {
  int* remote = static_cast<int*>(ipc_ptrs[source]);
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (int iter = 0; iter < iters; ++iter) {
    for (size_t idx = tid; idx < len; idx += stride) {
      local[idx] = remote[idx];
    }
    __threadfence_system();
  }
}

/**
 * @brief IPC Write benchmark functor
 */
template <typename Peer>
struct IPCWrite {
  int iters, target;

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& a, typename Peer::template Buffers<T>& b) {
    int rank = peer.mpi.GetWorldRank();
    if (rank != 0) return;

    auto& mem = *a[rank];
    auto ctx = mem.GetContext();
    size_t len = mem.Size() / sizeof(int);

    cudaLaunchConfig_t cfg{.gridDim = {1, 1, 1}, .blockDim = {256, 1, 1}, .stream = peer.stream};
    LAUNCH_KERNEL(&cfg, IPCWriteKernel, ctx.ipc_ptrs, target, len, iters);
    CUDA_CHECK(cudaStreamSynchronize(peer.stream));
  }
};

/**
 * @brief IPC Read benchmark functor
 */
template <typename Peer>
struct IPCRead {
  int iters, source;

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& a, typename Peer::template Buffers<T>& b) {
    int rank = peer.mpi.GetWorldRank();
    if (rank != 0) return;

    auto& mem = *a[rank];
    auto ctx = mem.GetContext();
    size_t len = mem.Size() / sizeof(int);

    cudaLaunchConfig_t cfg{.gridDim = {1, 1, 1}, .blockDim = {256, 1, 1}, .stream = peer.stream};
    LAUNCH_KERNEL(&cfg, IPCReadKernel, ctx.ipc_ptrs, source, len, static_cast<int*>(mem.Data()), iters);
    CUDA_CHECK(cudaStreamSynchronize(peer.stream));
  }
};

/**
 * @brief IPC Verify functor - verifies data on receiver side
 */
struct IPCVerify {
  int target;

  template <typename P, typename Buffers>
  void operator()(P& peer, Buffers& b) const {
    int rank = peer.mpi.GetWorldRank();
    if (rank != target) return;

    auto& mem = *b[rank];
    size_t num_ints = mem.Size() / sizeof(int);
    std::vector<int> host_buf(num_ints);
    CUDA_CHECK(cudaMemcpy(host_buf.data(), mem.Data(), mem.Size(), cudaMemcpyDeviceToHost));
    if (VerifyBufferData(host_buf, rank, rank, 0) > 0) throw std::runtime_error("IPC verification failed");
  }
};
