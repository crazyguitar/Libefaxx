/**
 * @file queue_test.cu
 * @brief MPMC Queue Tests and Benchmarks
 *
 * Three queue types tested:
 *   - Queue(Managed):  cudaMallocManaged - unified memory with page migration
 *   - PinnedQueue:     cudaHostAlloc + cudaHostGetDevicePointer - GPU writes over PCIe to host DRAM
 *   - GdrQueue:        GDRCopy - GPU memory with CPU access via PCIe BAR1
 *
 * See flow charts above each benchmark function for detailed operation.
 */
#include <affinity/affinity.h>
#include <affinity/taskset.h>
#include <cuda.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <device/common.cuh>
#include <mutex>
#include <queue/queue.cuh>
#include <thread>
#include <vector>

#define TEST_ASSERT(exp, msg)                                                  \
  do {                                                                         \
    if (!(exp)) {                                                              \
      SPDLOG_CRITICAL("[FAIL] {}:{}: {} - {}", __FILE__, __LINE__, #exp, msg); \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

template <typename T>
struct QueueAlloc {
  static constexpr const char* Name = "Queue(Managed)";
  static Queue<T>* Create(size_t size) { return new Queue<T>(size); }
  static void Destroy(Queue<T>* q) { delete q; }
};

template <typename T>
struct GdrQueueAlloc {
  static constexpr const char* Name = "GdrQueue";
  static GdrQueue<T>* Create(size_t size) { return new GdrQueue<T>(size); }
  static void Destroy(GdrQueue<T>* q) { delete q; }
};

template <typename T>
struct PinnedQueueAlloc {
  static constexpr const char* Name = "PinnedQueue";
  static PinnedQueue<T>* Create(size_t size) { return new PinnedQueue<T>(size); }
  static void Destroy(PinnedQueue<T>* q) { delete q; }
};

// =============================================================================
// Generic Kernels
// =============================================================================
template <typename Q>
__global__ void KernelPushSingle(Q* queue, int value, bool* success) {
  *success = queue->Push(value);
}

template <typename Q>
__global__ void KernelMultiProduce(Q* queue, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    while (!queue->Push(idx)) __threadfence();
  }
}

// =============================================================================
// Templated Tests
// =============================================================================
template <typename Q, typename Alloc>
void TestSingleThreadedBasic() {
  printf("[TEST] %s: Single-threaded basic push/pop...\n", Alloc::Name);

  auto* queue = Alloc::Create(4);
  bool* push_success;
  CUDA_CHECK(cudaMallocManaged(&push_success, sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = cfg.blockDim = dim3(1, 1, 1);
  cfg.stream = stream;

  for (int i = 0; i < 4; ++i) {
    *push_success = false;
    LAUNCH_KERNEL(&cfg, KernelPushSingle<Q>, queue, i, push_success);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TEST_ASSERT(*push_success, "Push should succeed");
  }

  *push_success = true;
  LAUNCH_KERNEL(&cfg, KernelPushSingle<Q>, queue, 99, push_success);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  TEST_ASSERT(!*push_success, "Push should fail when full");

  for (int i = 0; i < 4; ++i) {
    int value;
    TEST_ASSERT(queue->Pop(value), "Pop should succeed");
    TEST_ASSERT(value == i, "Value should match");
  }

  int dummy;
  TEST_ASSERT(!queue->Pop(dummy), "Pop should fail when empty");

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(push_success));
  Alloc::Destroy(queue);
  printf("[PASS] %s: Single-threaded basic push/pop.\n", Alloc::Name);
}

template <typename Q, typename Alloc>
void TestEmptyQueue() {
  printf("[TEST] %s: Empty queue behavior...\n", Alloc::Name);
  auto* queue = Alloc::Create(4);
  int dummy;
  TEST_ASSERT(!queue->Pop(dummy), "Pop should fail for empty queue");
  Alloc::Destroy(queue);
  printf("[PASS] %s: Empty queue behavior.\n", Alloc::Name);
}

template <typename Q, typename Alloc>
void TestFullQueue() {
  printf("[TEST] %s: Full queue behavior...\n", Alloc::Name);

  auto* queue = Alloc::Create(4);
  bool* push_success;
  CUDA_CHECK(cudaMallocManaged(&push_success, sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = cfg.blockDim = dim3(1, 1, 1);
  cfg.stream = stream;

  for (int i = 0; i < 4; ++i) {
    *push_success = false;
    LAUNCH_KERNEL(&cfg, KernelPushSingle<Q>, queue, i, push_success);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TEST_ASSERT(*push_success, "Push should succeed");
  }

  *push_success = true;
  LAUNCH_KERNEL(&cfg, KernelPushSingle<Q>, queue, 99, push_success);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  TEST_ASSERT(!*push_success, "Push should fail when full");

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(push_success));
  Alloc::Destroy(queue);
  printf("[PASS] %s: Full queue behavior.\n", Alloc::Name);
}

template <typename Q, typename Alloc>
void TestWraparound() {
  printf("[TEST] %s: Wraparound behavior...\n", Alloc::Name);

  auto* queue = Alloc::Create(4);
  bool* push_success;
  CUDA_CHECK(cudaMallocManaged(&push_success, sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = cfg.blockDim = dim3(1, 1, 1);
  cfg.stream = stream;

  for (int round = 0; round < 10; ++round) {
    for (int i = 0; i < 4; ++i) {
      *push_success = false;
      LAUNCH_KERNEL(&cfg, KernelPushSingle<Q>, queue, round * 100 + i, push_success);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      TEST_ASSERT(*push_success, "Push should succeed");
    }
    for (int i = 0; i < 4; ++i) {
      int value;
      TEST_ASSERT(queue->Pop(value), "Pop should succeed");
      TEST_ASSERT(value == round * 100 + i, "Value should match");
    }
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(push_success));
  Alloc::Destroy(queue);
  printf("[PASS] %s: Wraparound behavior.\n", Alloc::Name);
}

template <typename Q, typename Alloc>
void TestMPSC() {
  constexpr int kThreads = 256, kBlocks = 4, kTotal = kThreads * kBlocks;
  printf("[TEST] %s: Multi-producer single-consumer...\n", Alloc::Name);

  auto* queue = Alloc::Create(kTotal);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<int> consumed;
  consumed.reserve(kTotal);

  std::thread consumer([&]() {
    int data, count = 0;
    while (count < kTotal) {
      if (queue->Pop(data)) {
        consumed.push_back(data);
        count++;
      } else
        std::this_thread::yield();
    }
  });

  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(kBlocks, 1, 1);
  cfg.blockDim = dim3(kThreads, 1, 1);
  cfg.stream = stream;
  LAUNCH_KERNEL(&cfg, KernelMultiProduce<Q>, queue, kTotal);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  consumer.join();

  TEST_ASSERT(consumed.size() == kTotal, "All items consumed");
  std::sort(consumed.begin(), consumed.end());
  for (int i = 0; i < kTotal; ++i) TEST_ASSERT(consumed[i] == i, "Value present");

  CUDA_CHECK(cudaStreamDestroy(stream));
  Alloc::Destroy(queue);
  printf("[PASS] %s: Multi-producer single-consumer.\n", Alloc::Name);
}

template <typename Q, typename Alloc>
void TestMPMC() {
  constexpr int kThreads = 128, kBlocks = 4, kTotal = kThreads * kBlocks, kConsumers = 4;
  printf("[TEST] %s: Multi-producer multi-consumer...\n", Alloc::Name);

  auto* queue = Alloc::Create(kTotal);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<int> all_consumed;
  std::mutex mtx;
  std::atomic<int> total{0};

  std::vector<std::thread> consumers;
  for (int i = 0; i < kConsumers; ++i) {
    consumers.emplace_back([&]() {
      int data;
      std::vector<int> local;
      while (total.load() < kTotal) {
        if (queue->Pop(data)) {
          local.push_back(data);
          total.fetch_add(1);
        } else
          std::this_thread::yield();
      }
      std::lock_guard<std::mutex> lock(mtx);
      all_consumed.insert(all_consumed.end(), local.begin(), local.end());
    });
  }

  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(kBlocks, 1, 1);
  cfg.blockDim = dim3(kThreads, 1, 1);
  cfg.stream = stream;
  LAUNCH_KERNEL(&cfg, KernelMultiProduce<Q>, queue, kTotal);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (auto& c : consumers) c.join();

  TEST_ASSERT(all_consumed.size() == kTotal, "All items consumed");
  std::sort(all_consumed.begin(), all_consumed.end());
  for (int i = 0; i < kTotal; ++i) TEST_ASSERT(all_consumed[i] == i, "Value present");

  CUDA_CHECK(cudaStreamDestroy(stream));
  Alloc::Destroy(queue);
  printf("[PASS] %s: Multi-producer multi-consumer.\n", Alloc::Name);
}

template <typename Q, typename Alloc>
void TestStress() {
  constexpr int kThreads = 512, kBlocks = 32, kTotal = kThreads * kBlocks, kQueueSize = 256;
  printf("[TEST] %s: Stress test - high contention...\n", Alloc::Name);

  auto* queue = Alloc::Create(kQueueSize);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<int> consumed;
  consumed.reserve(kTotal);

  std::thread consumer([&]() {
    int data, count = 0;
    while (count < kTotal) {
      if (queue->Pop(data)) {
        consumed.push_back(data);
        count++;
      }
    }
  });

  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(kBlocks, 1, 1);
  cfg.blockDim = dim3(kThreads, 1, 1);
  cfg.stream = stream;
  LAUNCH_KERNEL(&cfg, KernelMultiProduce<Q>, queue, kTotal);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  consumer.join();

  TEST_ASSERT(consumed.size() == kTotal, "All items consumed");
  std::sort(consumed.begin(), consumed.end());
  for (int i = 0; i < kTotal; ++i) TEST_ASSERT(consumed[i] == i, "Value present");

  CUDA_CHECK(cudaStreamDestroy(stream));
  Alloc::Destroy(queue);
  printf("[PASS] %s: Stress test - high contention.\n", Alloc::Name);
}

// Run all tests for a queue type
template <typename Q, typename Alloc>
void RunAllTests() {
  TestSingleThreadedBasic<Q, Alloc>();
  TestEmptyQueue<Q, Alloc>();
  TestFullQueue<Q, Alloc>();
  TestWraparound<Q, Alloc>();
  TestMPSC<Q, Alloc>();
  TestMPMC<Q, Alloc>();
  TestStress<Q, Alloc>();
}

/**
 * Throughput Benchmark (Sequential): GPU pushes all, then CPU pops all
 *
 *   ┌───────┐  push N items  ┌───────┐   sync   ┌───────┐  pop N items  ┌───────┐
 *   │ Start │ ─────────────► │  GPU  │ ───────► │ Wait  │ ────────────► │  CPU  │
 *   └───────┘                │Kernel │          │ Sync  │               │ Loop  │
 *                            └───────┘          └───────┘               └───────┘
 */
template <typename Q, typename Alloc>
void BenchThroughput(cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop) {
  constexpr int kThreads = 256, kBlocks = 16, kItems = kThreads * kBlocks, kIters = 10;

  auto* queue = Alloc::Create(kItems);

  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(kBlocks, 1, 1);
  cfg.blockDim = dim3(kThreads, 1, 1);
  cfg.stream = stream;

  // Warmup
  for (int i = 0; i < 2; ++i) {
    LAUNCH_KERNEL(&cfg, KernelMultiProduce<Q>, queue, kItems);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int data;
    while (queue->Pop(data)) {
    }
  }

  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int iter = 0; iter < kIters; ++iter) {
    LAUNCH_KERNEL(&cfg, KernelMultiProduce<Q>, queue, kItems);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int data;
    while (queue->Pop(data)) {
    }
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  printf("  %-20s %.4f ms, %.4f M ops/sec\n", Alloc::Name, ms, (kItems * kIters) / (ms / 1000.0) / 1e6);

  Alloc::Destroy(queue);
}

// Helper for host-side push (GdrQueue/PinnedQueue need PushHost, Queue uses Push)
template <typename Q>
inline bool HostPush(Q* queue, const typename std::remove_pointer<decltype(queue)>::type::value_type& data) {
  return queue->Push(data);
}
template <typename T>
inline bool HostPush(GdrQueue<T>* queue, const T& data) {
  return queue->PushHost(data);
}
template <typename T>
inline bool HostPush(PinnedQueue<T>* queue, const T& data) {
  return queue->PushHost(data);
}

/**
 * CPU Polling Latency: Measures Pop() call overhead (mostly failed pops)
 *
 *   ┌───────┐  pre-fill 1 item  ┌───────┐  pop() x 100K   ┌─────────┐
 *   │ Setup │ ────────────────► │ Queue │ ──────────────► │ Measure │
 *   └───────┘                   └───────┘                 └─────────┘
 *
 *   Note: Only first pop succeeds; measures empty-queue polling cost.
 */
template <typename Q, typename Alloc>
void BenchPolling() {
  constexpr int kIters = 100000;

  auto* queue = Alloc::Create(64);
  HostPush(queue, 42);

  auto t_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kIters; ++i) {
    int data;
    volatile bool found = queue->Pop(data);
    (void)found;
  }
  auto t_end = std::chrono::high_resolution_clock::now();
  double ns = std::chrono::duration<double, std::nano>(t_end - t_start).count();
  printf("  %-20s %.4f ns/op\n", Alloc::Name, ns / kIters);

  Alloc::Destroy(queue);
}

/**
 * Concurrent Benchmark: GPU pushes WHILE CPU pops simultaneously
 *
 *   ┌─────────────────────────────────────────────────────────────┐
 *   │                    Time ──────────────────►                 │
 *   │                                                             │
 *   │  GPU:  ┌─push─┬─push─┬─push─┬─push─┬─ ─ ─►                  │
 *   │        └──────┴──────┴──────┴──────┘                        │
 *   │                  ▼ data visible                             │
 *   │  CPU:      ┌─pop─┬─pop─┬─pop─┬─pop─┬─ ─ ─►                  │
 *   │            └─────┴─────┴─────┴─────┘                        │
 *   └─────────────────────────────────────────────────────────────┘
 */
template <typename Q>
__global__ void ConcurrentPushKernel(Q* queue, int iters) {
  for (int i = 0; i < iters; ++i) {
    while (!queue->Push(i)) {
    }
    __threadfence_system();
  }
}

template <typename Q, typename Alloc>
void BenchConcurrent() {
  constexpr int kIters = 50000;

  auto* queue = Alloc::Create(1024);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cudaLaunchConfig_t cfg{};
  cfg.gridDim = cfg.blockDim = dim3(1, 1, 1);
  cfg.stream = stream;

  auto t_start = std::chrono::high_resolution_clock::now();

  LAUNCH_KERNEL(&cfg, ConcurrentPushKernel<Q>, queue, kIters);

  int consumed = 0;
  while (consumed < kIters) {
    int data;
    if (queue->Pop(data)) consumed++;
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto t_end = std::chrono::high_resolution_clock::now();

  double us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
  printf("  %-20s %.4f us total, %.4f us/op\n", Alloc::Name, us, us / kIters);

  CUDA_CHECK(cudaStreamDestroy(stream));
  Alloc::Destroy(queue);
}

void RunBenchmarks() {
  printf("[BENCH] Throughput (GPU push, CPU pop) - sequential\n\n");

  cudaStream_t stream;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  BenchThroughput<Queue<int>, QueueAlloc<int>>(stream, start, stop);
  BenchThroughput<PinnedQueue<int>, PinnedQueueAlloc<int>>(stream, start, stop);
  BenchThroughput<GdrQueue<int>, GdrQueueAlloc<int>>(stream, start, stop);

  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaStreamDestroy(stream));

  printf("\n[BENCH] CPU polling latency (cached memory)\n\n");

  BenchPolling<Queue<int>, QueueAlloc<int>>();
  BenchPolling<PinnedQueue<int>, PinnedQueueAlloc<int>>();
  BenchPolling<GdrQueue<int>, GdrQueueAlloc<int>>();

  printf("\n[BENCH] Concurrent (GPU push + CPU pop simultaneously)\n\n");

  BenchConcurrent<Queue<int>, QueueAlloc<int>>();
  BenchConcurrent<PinnedQueue<int>, PinnedQueueAlloc<int>>();
  BenchConcurrent<GdrQueue<int>, GdrQueueAlloc<int>>();

  printf("\n");
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char* argv[]) {
  CUDA_CHECK(cudaSetDevice(0));

  auto& loc = GPUloc::Get();
  auto& affinity = loc.GetGPUAffinity()[0];
  Taskset::Set(affinity.cores[0]->logical_index);

  printf("=== Queue (Managed) Tests ===\n\n");
  RunAllTests<Queue<int>, QueueAlloc<int>>();

  printf("\n=== PinnedQueue Tests ===\n\n");
  RunAllTests<PinnedQueue<int>, PinnedQueueAlloc<int>>();

  printf("\n=== GdrQueue Tests ===\n\n");
  RunAllTests<GdrQueue<int>, GdrQueueAlloc<int>>();

  printf("\n=== Performance Benchmarks ===\n\n");
  RunBenchmarks();

  printf("=== All tests passed! ===\n");
  return 0;
}
