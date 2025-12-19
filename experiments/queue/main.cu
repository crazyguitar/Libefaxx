#include <cuda.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <queue/queue.cuh>
#include <thread>
#include <vector>

#define ASSERT(exp)                                                           \
  do {                                                                        \
    if (!(exp)) {                                                             \
      SPDLOG_CRITICAL("[{}:{}] " #exp " assertion fail", __FILE__, __LINE__); \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

#define LAUNCH_KERNEL(cfg, kernel, ...) CUDA_CHECK(cudaLaunchKernelEx(cfg, kernel, ##__VA_ARGS__))

static constexpr int kThreads = 512;
static constexpr int kBlocks = 16;
static constexpr int kQueueSize = kThreads * kBlocks;

__global__ void Produce(Queue<int>* __restrict__ queue, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    // Retry push until successful (queue might be temporarily full)
    // Note: In production code, consider adding backoff or limiting retry attempts
    while (!queue->Push(idx)) {
      // Yield briefly to reduce contention
      __threadfence();
    }
  }
}

__host__ void Consume(Queue<int>* queue, std::vector<int>& consumed, int size) {
  int data;
  for (int i = 0; i < size; ++i) {
    // Use direct Pop() attempt in a loop instead of relying on Empty() for control flow
    while (!queue->Pop(data)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    consumed.emplace_back(data);
  }
}

__host__ void Verify(const std::vector<int>& consumed, int size) {
  ASSERT(consumed.size() == static_cast<size_t>(size));
  std::vector<int> sorted = consumed;
  std::sort(sorted.begin(), sorted.end());
  for (int i = 0; i < size; ++i) ASSERT(sorted[i] == i);
}

struct Test {
  Queue<int>* queue;
  cudaStream_t stream;
  cudaEvent_t start;
  cudaEvent_t stop;

  __host__ Test() {
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMallocManaged(&queue, sizeof(Queue<int>)));
    CUDA_CHECK(cudaStreamCreate(&stream));
    // call constructor explicitly
    new (queue) Queue<int>(kQueueSize);
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }

  __host__ ~Test() {
    // call destructor explicitly
    queue->~Queue<int>();
    CUDA_CHECK(cudaFree(queue));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
  }

  __host__ void Launch() {
    std::vector<int> consumed;
    consumed.reserve(kQueueSize);

    // CPU producer
    std::thread consumer(Consume, queue, std::ref(consumed), kQueueSize);
    cudaLaunchConfig_t cfg{0};
    cfg.gridDim = dim3(kBlocks, 1, 1);
    cfg.blockDim = dim3(kThreads, 1, 1);
    cfg.stream = stream;
    LAUNCH_KERNEL(&cfg, Produce, queue, kQueueSize);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    consumer.join();
    Verify(consumed, kQueueSize);
  }

  __host__ void Run(size_t iter) {
    float elapse;
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (size_t i = 0; i < iter; ++i) Launch();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapse, start, stop));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto latency = elapse / iter;
    printf("elapse: %f, latency: %f\n", elapse, latency);
  }
};

int main(int argc, char* argv[]) {
  Test test;
  test.Run(8);
}
