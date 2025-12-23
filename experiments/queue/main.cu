#include <affinity/affinity.h>
#include <affinity/taskset.h>
#include <bench/arguments.h>
#include <cuda.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <bench/modules/gin.cuh>
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
    while (!queue->Push(idx)) __threadfence();
  }
}

__host__ void Consume(Queue<int>* queue, std::vector<int>& consumed, int size) {
  int data;
  for (int i = 0; i < size; ++i) {
    while (!queue->Pop(data)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
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
  cudaEvent_t start, stop;

  __host__ Test() {
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
    std::thread consumer(Consume, queue, std::ref(consumed), kQueueSize);
    cudaLaunchConfig_t cfg{.gridDim = {kBlocks}, .blockDim = {kThreads}, .stream = stream};
    LAUNCH_KERNEL(&cfg, Produce, queue, kQueueSize);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    consumer.join();
    Verify(consumed, kQueueSize);
  }

  __host__ void Run(int iter) {
    float elapse;
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iter; ++i) Launch();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapse, start, stop));
    printf("Test: elapse=%.2fms latency=%.2fms\n", elapse, elapse / iter);
  }
};

struct Bench {
  Queue<DeviceRequest> queue;
  uint64_t* posted;
  uint64_t* completed;
  int* data;
  cudaStream_t stream;

  Bench(size_t size) {
    CUDA_CHECK(cudaMallocManaged(&posted, sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocManaged(&completed, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&data, size));
    CUDA_CHECK(cudaStreamCreate(&stream));
    *posted = 0;
    *completed = 0;
  }

  ~Bench() {
    CUDA_CHECK(cudaFree(posted));
    CUDA_CHECK(cudaFree(completed));
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  DeviceContext GetContext() { return {&queue, posted, completed}; }

  void Complete() { reinterpret_cast<cuda::std::atomic<uint64_t>*>(completed)->fetch_add(1, cuda::std::memory_order_relaxed); }

  BenchResult Run(size_t size, int iters) {
    *posted = 0;
    *completed = 0;
    auto ctx = GetContext();
    size_t len = size / sizeof(int);

    cudaLaunchConfig_t cfg{.gridDim = {1}, .blockDim = {256}, .stream = stream};
    auto start = std::chrono::high_resolution_clock::now();
    LAUNCH_KERNEL(&cfg, ProxyWriteKernel, ctx, 0, len, data, 0ULL, iters);

    for (int done = 0; done < iters;) {
      DeviceRequest req;
      if (queue.Pop(req)) {
        Complete();
        ++done;
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();

    double time_us = std::chrono::duration<double, std::micro>(end - start).count();
    double lat_us = time_us / iters;
    double throughput_mops = static_cast<double>(iters) / time_us;
    return {size, lat_us, throughput_mops, 0.0};
  }
};

void PrintSummary(const char* title, int warmup, int iters, const std::vector<BenchResult>& results) {
  printf("#\n# %s\n#\n", title);
  printf("# warmup iters: %d\n", warmup);
  printf("# bench iters: %d\n#\n", iters);
  printf("# Pattern: GPU -> Queue -> CPU (no RDMA)\n#\n");
  printf("%12s %14s %10s\n", "size", "Mops/s", "Lat(us)");
  for (const auto& r : results) printf("%12zu %14.2f %10.2f\n", r.size, r.bw_gbps, r.time_us);
  printf("#\n# Benchmark complete.\n");
}

int main(int argc, char* argv[]) {
  auto opts = parse_args(argc, argv);

  int device = 0;
  auto& loc = GPUloc::Get();
  auto& aff = loc.GetGPUAffinity()[device];
  std::cout << fmt::format("CUDA Device {}: \"{}\"\n", device, aff.prop.name) << aff << std::flush;
  Taskset::Set(aff.cores[device]->logical_index);
  CUDA_CHECK(cudaSetDevice(device));

  // Original test
  Test test;
  test.Run(opts.warmup);

  // Proxy-style benchmark (no RDMA)
  Bench bench(opts.maxbytes);
  for (int i = 0; i < opts.warmup; ++i) bench.Run(opts.minbytes, opts.repeat);

  std::vector<BenchResult> results;
  for (auto size : generate_sizes(opts)) {
    BenchResult sum{size, 0, 0, 0};
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < opts.repeat; ++i) {
      auto r = bench.Run(size, 1);
      sum.time_us += r.time_us;
      sum.bw_gbps += r.bw_gbps;
      if (i % 64 == 0) {
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - start).count();
        size_t cur_bytes = i * size, total_bytes = (size_t)opts.repeat * size;
        double lat = sum.time_us / (i + 1);
        printf("\r[%.3fs] ops=%d/%d bytes=%zu/%zu lat=%.3fus\033[K", elapsed, i, opts.repeat, cur_bytes, total_bytes, lat);
        fflush(stdout);
      }
    }
    sum.time_us /= opts.repeat;
    sum.bw_gbps /= opts.repeat;
    results.push_back(sum);
    printf("\r\033[K");
  }

  PrintSummary("MPSC Queue Benchmark", opts.warmup, opts.repeat, results);
}
