/**
 * @file main.cu
 * @brief NVSHMEM-like example using shmem_* API
 *
 * Equivalent to: https://docs.nvidia.com/nvshmem/api/using.html#example-nvshmem-program
 *
 * mpirun -np 2 --npernode 1 ./shmem
 */
#include <io/runner.h>

#include <shmem/shmem.cuh>

void shmem_proxy(void* ptr, int send_count, int recv_count) {
  auto& peer = shmem_peer();
  auto& mem = shmem_mem(ptr);
  auto ctx = shmem_ctx(ptr);
  int npes = shmem_n_pes();
  int mype = shmem_my_pe();
  int next_pe = (mype + 1) % npes;
  int prev_pe = (mype - 1 + npes) % npes;

  // Check if we need RDMA for send or receive
  bool need_rdma_send = !(ctx.ipc_ptrs && ctx.ipc_ptrs[next_pe]);
  bool need_rdma_recv = !(ctx.ipc_ptrs && ctx.ipc_ptrs[prev_pe]);

  // All communication is IPC, no RDMA proxy needed
  if (!need_rdma_send && !need_rdma_recv) return;

  int rdma_send_count = need_rdma_send ? send_count : 0;
  int rdma_recv_count = need_rdma_recv ? recv_count : 0;

  for (auto& efa : peer.efas) IO::Get().Join<FabricSelector>(efa);
  Run([&]() -> Coro<> {
    for (int done = 0; done < rdma_send_count;) {
      DeviceRequest req;
      if (!ctx.queue->Pop(req)) {
        co_await YieldAwaiter{};
        continue;
      }
      co_await mem.Writeall(req.rank, req.imm, 0);
      mem.Complete();
      ++done;
    }
    for (int i = 0; i < rdma_recv_count; ++i) co_await mem.WaitImmdata(1);
    for (auto& efa : peer.efas) IO::Get().Quit<FabricSelector>(efa);
  }());
}

template <typename Ctx>
__global__ void simple_shift(Ctx ctx, int* __restrict__ target, int mype, int npes) {
  int peer = (mype + 1) % npes;
  shmem_int_p(ctx, target, mype, peer);
}

int main(int argc, char* argv[]) {
  shmem_init();

  int mype = shmem_my_pe();
  int npes = shmem_n_pes();

  CUDA_CHECK(cudaSetDevice(MPI::Get().GetLocalRank()));

  int* target = static_cast<int*>(shmem_malloc(sizeof(int)));
  auto ctx = shmem_ctx(target);

  cudaLaunchConfig_t cfg{.gridDim = {1, 1, 1}, .blockDim = {1, 1, 1}};
  LAUNCH_KERNEL(&cfg, simple_shift<decltype(ctx)>, ctx, target, mype, npes);

  shmem_proxy(target, 1, 1);

  CUDA_CHECK(cudaDeviceSynchronize());
  shmem_barrier_all();

  printf("[%d of %d] run complete\n", mype, npes);

  shmem_free(target);
  shmem_finalize();
  return 0;
}
