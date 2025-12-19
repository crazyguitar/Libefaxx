#include <cuda.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
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

#define LAUNCH_KERNEL(cfg, kernel, ...) CUDA_CHECK(cudaLaunchKernelEx(cfg, kernel, ##__VA_ARGS__))

// =============================================================================
// Test: Power-of-two size validation
// =============================================================================
void TestPowerOfTwoValidation() {
  printf("[TEST] Power-of-two size validation...\n");

  // Valid sizes (power of two)
  Queue<int>* q1;
  CUDA_CHECK(cudaMallocManaged(&q1, sizeof(Queue<int>)));
  new (q1) Queue<int>(4);
  q1->~Queue<int>();
  CUDA_CHECK(cudaFree(q1));

  Queue<int>* q2;
  CUDA_CHECK(cudaMallocManaged(&q2, sizeof(Queue<int>)));
  new (q2) Queue<int>(16);
  q2->~Queue<int>();
  CUDA_CHECK(cudaFree(q2));

  Queue<int>* q3;
  CUDA_CHECK(cudaMallocManaged(&q3, sizeof(Queue<int>)));
  new (q3) Queue<int>(1024);
  q3->~Queue<int>();
  CUDA_CHECK(cudaFree(q3));

  printf("[PASS] Power-of-two validation passed for valid sizes.\n");
}

// =============================================================================
// Test: Basic single-threaded push/pop
// =============================================================================
__global__ void PushSingle(Queue<int>* queue, int value, bool* success) { *success = queue->Push(value); }

void TestSingleThreadedBasic() {
  printf("[TEST] Single-threaded basic push/pop...\n");

  Queue<int>* queue;
  CUDA_CHECK(cudaMallocManaged(&queue, sizeof(Queue<int>)));
  new (queue) Queue<int>(4);

  bool* push_success;
  CUDA_CHECK(cudaMallocManaged(&push_success, sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cudaLaunchConfig_t cfg{0};
  cfg.gridDim = dim3(1, 1, 1);
  cfg.blockDim = dim3(1, 1, 1);
  cfg.stream = stream;

  // Push 4 items (queue size is 4)
  for (int i = 0; i < 4; ++i) {
    *push_success = false;
    LAUNCH_KERNEL(&cfg, PushSingle, queue, i, push_success);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TEST_ASSERT(*push_success, "Push should succeed for non-full queue");
  }

  // Verify queue is full by trying to push (should fail)
  *push_success = true;
  LAUNCH_KERNEL(&cfg, PushSingle, queue, 99, push_success);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  TEST_ASSERT(!*push_success, "Push should fail when queue is full");

  // Pop all items and verify
  for (int i = 0; i < 4; ++i) {
    int value;
    bool pop_success = queue->Pop(value);
    TEST_ASSERT(pop_success, "Pop should succeed for non-empty queue");
    TEST_ASSERT(value == i, "Popped value should match pushed value");
  }

  // Verify queue is empty by trying to pop (should fail)
  int dummy;
  TEST_ASSERT(!queue->Pop(dummy), "Pop should fail for empty queue");

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(push_success));
  queue->~Queue<int>();
  CUDA_CHECK(cudaFree(queue));

  printf("[PASS] Single-threaded basic push/pop.\n");
}

// =============================================================================
// Test: Full queue behavior
// =============================================================================
__global__ void TryPush(Queue<int>* queue, int value, bool* success) { *success = queue->Push(value); }

void TestFullQueueBehavior() {
  printf("[TEST] Full queue behavior...\n");

  Queue<int>* queue;
  CUDA_CHECK(cudaMallocManaged(&queue, sizeof(Queue<int>)));
  new (queue) Queue<int>(4);

  bool* push_success;
  CUDA_CHECK(cudaMallocManaged(&push_success, sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cudaLaunchConfig_t cfg{0};
  cfg.gridDim = dim3(1, 1, 1);
  cfg.blockDim = dim3(1, 1, 1);
  cfg.stream = stream;

  // Fill the queue
  for (int i = 0; i < 4; ++i) {
    *push_success = false;
    LAUNCH_KERNEL(&cfg, TryPush, queue, i, push_success);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TEST_ASSERT(*push_success, "Push should succeed while filling queue");
  }

  // Try to push to full queue (should fail)
  *push_success = true;
  LAUNCH_KERNEL(&cfg, TryPush, queue, 99, push_success);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  TEST_ASSERT(!*push_success, "Push should fail when queue is full");

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(push_success));
  queue->~Queue<int>();
  CUDA_CHECK(cudaFree(queue));

  printf("[PASS] Full queue behavior.\n");
}

// =============================================================================
// Test: Empty queue behavior
// =============================================================================
void TestEmptyQueueBehavior() {
  printf("[TEST] Empty queue behavior...\n");

  Queue<int>* queue;
  CUDA_CHECK(cudaMallocManaged(&queue, sizeof(Queue<int>)));
  new (queue) Queue<int>(4);

  // Verify new queue is empty by trying to pop (should fail)
  int dummy;
  TEST_ASSERT(!queue->Pop(dummy), "Pop should fail for empty queue");

  queue->~Queue<int>();
  CUDA_CHECK(cudaFree(queue));

  printf("[PASS] Empty queue behavior.\n");
}

// =============================================================================
// Test: Multi-producer (GPU threads) single-consumer (CPU thread)
// =============================================================================
static constexpr int kMPSCThreads = 256;
static constexpr int kMPSCBlocks = 4;
static constexpr int kMPSCQueueSize = kMPSCThreads * kMPSCBlocks;

__global__ void MultiProduce(Queue<int>* queue, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    // Retry push until successful
    // Note: In production code, consider adding backoff or limiting retry attempts
    while (!queue->Push(idx)) {
      // Yield briefly to reduce contention (memory fence helps prevent aggressive spinning)
      __threadfence();
    }
  }
}

void TestMultiProducerSingleConsumer() {
  printf("[TEST] Multi-producer (GPU) single-consumer (CPU)...\n");

  Queue<int>* queue;
  CUDA_CHECK(cudaMallocManaged(&queue, sizeof(Queue<int>)));
  new (queue) Queue<int>(kMPSCQueueSize);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<int> consumed;
  consumed.reserve(kMPSCQueueSize);

  // Start consumer thread
  std::thread consumer([&]() {
    int data;
    int count = 0;
    while (count < kMPSCQueueSize) {
      if (queue->Pop(data)) {
        consumed.push_back(data);
        count++;
      } else {
        std::this_thread::yield();
      }
    }
  });

  // Launch producer kernel
  cudaLaunchConfig_t cfg{0};
  cfg.gridDim = dim3(kMPSCBlocks, 1, 1);
  cfg.blockDim = dim3(kMPSCThreads, 1, 1);
  cfg.stream = stream;
  LAUNCH_KERNEL(&cfg, MultiProduce, queue, kMPSCQueueSize);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Wait for consumer
  consumer.join();

  // Verify all items were consumed
  TEST_ASSERT(consumed.size() == kMPSCQueueSize, "All items should be consumed");

  // Verify all values are present (order may vary)
  std::sort(consumed.begin(), consumed.end());
  for (int i = 0; i < kMPSCQueueSize; ++i) {
    TEST_ASSERT(consumed[i] == i, "All values should be present");
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  queue->~Queue<int>();
  CUDA_CHECK(cudaFree(queue));

  printf("[PASS] Multi-producer (GPU) single-consumer (CPU).\n");
}

// =============================================================================
// Test: Multi-producer (GPU threads) multi-consumer (CPU threads)
// =============================================================================
static constexpr int kMPMCThreads = 128;
static constexpr int kMPMCBlocks = 4;
static constexpr int kMPMCQueueSize = kMPMCThreads * kMPMCBlocks;
static constexpr int kNumConsumers = 4;

void TestMultiProducerMultiConsumer() {
  printf("[TEST] Multi-producer (GPU) multi-consumer (CPU threads)...\n");

  Queue<int>* queue;
  CUDA_CHECK(cudaMallocManaged(&queue, sizeof(Queue<int>)));
  new (queue) Queue<int>(kMPMCQueueSize);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<int> all_consumed;
  std::mutex consumed_mutex;
  std::atomic<int> total_consumed{0};

  // Start multiple consumer threads
  std::vector<std::thread> consumers;
  for (int i = 0; i < kNumConsumers; ++i) {
    consumers.emplace_back([&]() {
      int data;
      std::vector<int> local_consumed;
      while (total_consumed.load() < kMPMCQueueSize) {
        if (queue->Pop(data)) {
          local_consumed.push_back(data);
          total_consumed.fetch_add(1);
        } else {
          std::this_thread::yield();
        }
      }
      // Merge local results
      std::lock_guard<std::mutex> lock(consumed_mutex);
      all_consumed.insert(all_consumed.end(), local_consumed.begin(), local_consumed.end());
    });
  }

  // Launch producer kernel
  cudaLaunchConfig_t cfg{0};
  cfg.gridDim = dim3(kMPMCBlocks, 1, 1);
  cfg.blockDim = dim3(kMPMCThreads, 1, 1);
  cfg.stream = stream;
  LAUNCH_KERNEL(&cfg, MultiProduce, queue, kMPMCQueueSize);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Wait for all consumers
  for (auto& consumer : consumers) {
    consumer.join();
  }

  // Verify all items were consumed
  TEST_ASSERT(all_consumed.size() == kMPMCQueueSize, "All items should be consumed");

  // Verify all values are present (order may vary)
  std::sort(all_consumed.begin(), all_consumed.end());
  for (int i = 0; i < kMPMCQueueSize; ++i) {
    TEST_ASSERT(all_consumed[i] == i, "All values should be present");
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  queue->~Queue<int>();
  CUDA_CHECK(cudaFree(queue));

  printf("[PASS] Multi-producer (GPU) multi-consumer (CPU threads).\n");
}

// =============================================================================
// Test: Queue wraparound behavior
// =============================================================================
void TestWraparound() {
  printf("[TEST] Queue wraparound behavior...\n");

  Queue<int>* queue;
  CUDA_CHECK(cudaMallocManaged(&queue, sizeof(Queue<int>)));
  new (queue) Queue<int>(4);

  bool* push_success;
  CUDA_CHECK(cudaMallocManaged(&push_success, sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cudaLaunchConfig_t cfg{0};
  cfg.gridDim = dim3(1, 1, 1);
  cfg.blockDim = dim3(1, 1, 1);
  cfg.stream = stream;

  // Fill and drain multiple times to test wraparound
  for (int round = 0; round < 5; ++round) {
    // Fill queue
    for (int i = 0; i < 4; ++i) {
      *push_success = false;
      LAUNCH_KERNEL(&cfg, PushSingle, queue, round * 100 + i, push_success);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      TEST_ASSERT(*push_success, "Push should succeed during fill");
    }

    // Drain queue
    for (int i = 0; i < 4; ++i) {
      int value;
      bool pop_success = queue->Pop(value);
      TEST_ASSERT(pop_success, "Pop should succeed during drain");
      TEST_ASSERT(value == round * 100 + i, "Value should match in order");
    }

    // Verify queue is empty by trying to pop (should fail)
    int check_empty;
    TEST_ASSERT(!queue->Pop(check_empty), "Queue should be empty after drain");
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(push_success));
  queue->~Queue<int>();
  CUDA_CHECK(cudaFree(queue));

  printf("[PASS] Queue wraparound behavior.\n");
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char* argv[]) {
  CUDA_CHECK(cudaSetDevice(0));

  printf("=== MPMC Queue Tests ===\n\n");

  TestPowerOfTwoValidation();
  TestSingleThreadedBasic();
  TestFullQueueBehavior();
  TestEmptyQueueBehavior();
  TestMultiProducerSingleConsumer();
  TestMultiProducerMultiConsumer();
  TestWraparound();

  printf("\n=== All tests passed! ===\n");
  return 0;
}
