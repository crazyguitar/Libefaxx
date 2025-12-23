/**
 * @file main.cu
 * @brief MPSC Queue Benchmark - GPU producer, CPU consumer
 */
#include <affinity/affinity.h>
#include <affinity/taskset.h>
#include <cuda.h>
#include <getopt.h>
#include <io/awaiter.h>
#include <io/runner.h>
#include <rdma/fabric/request.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <device/common.cuh>
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
  Queue<int> queue;
  cudaStream_t stream;
  cudaEvent_t start, stop;

  __host__ Test() : queue(kQueueSize) {
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }

  __host__ ~Test() {
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
  }

  __host__ void Launch() {
    std::vector<int> consumed;
    consumed.reserve(kQueueSize);
    std::thread consumer(Consume, &queue, std::ref(consumed), kQueueSize);
    cudaLaunchConfig_t cfg{.gridDim = {kBlocks}, .blockDim = {kThreads}, .stream = stream};
    LAUNCH_KERNEL(&cfg, Produce, &queue, kQueueSize);
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

struct Options {
  int iters = 65536;  // items per run
  int repeat = 1;     // benchmark repetitions
  int warmup = 1;     // warmup runs
};

inline void usage(const char* prog) {
  printf("Usage: %s [OPTIONS]\n", prog);
  printf("  -n, --iters=N    Items per run (default: 65536)\n");
  printf("  -r, --repeat=N   Benchmark repetitions (default: 1)\n");
  printf("  -w, --warmup=N   Warmup runs (default: 1)\n");
}

inline Options parse_args(int argc, char* argv[]) {
  Options opts;
  static struct option long_opts[] = {
      {"help", no_argument, nullptr, 'h'},
      {"iters", required_argument, nullptr, 'n'},
      {"repeat", required_argument, nullptr, 'r'},
      {"warmup", required_argument, nullptr, 'w'},
      {nullptr, 0, nullptr, 0}
  };
  int opt;
  while ((opt = getopt_long(argc, argv, "hn:r:w:", long_opts, nullptr)) != -1) {
    switch (opt) {
      case 'h':
        usage(argv[0]);
        exit(0);
      case 'n':
        opts.iters = std::stoi(optarg);
        break;
      case 'r':
        opts.repeat = std::stoi(optarg);
        break;
      case 'w':
        opts.warmup = std::stoi(optarg);
        break;
      default:
        usage(argv[0]);
        exit(1);
    }
  }
  return opts;
}

struct BenchResult {
  size_t size;
  double time_us;
  double mops;
};

template <size_t N>
struct alignas(8) Payload {
  uint8_t data[N];
};

template <typename T>
struct BenchContext {
  Queue<T>* queue;
  uint64_t* posted;
  uint64_t* completed;
};

/** @brief Push with Quiet per operation (blocking) */
template <typename T>
__device__ __forceinline__ void DevicePush(BenchContext<T> ctx) {
  __threadfence_system();
  if (threadIdx.x == 0) {
    T payload{};
    while (!ctx.queue->Push(payload)) __threadfence_system();
    atomicAdd(reinterpret_cast<unsigned long long*>(ctx.posted), 1ULL);
    Fence();
    Quiet(ctx.posted, ctx.completed);
  }
  __syncthreads();
}

/** @brief Push without waiting (NBI - non-blocking interface) */
template <typename T>
__device__ __forceinline__ void DevicePushNBI(BenchContext<T> ctx) {
  __threadfence_system();
  if (threadIdx.x == 0) {
    T payload{};
    while (!ctx.queue->Push(payload)) __threadfence_system();
    atomicAdd(reinterpret_cast<unsigned long long*>(ctx.posted), 1ULL);
    Fence();
  }
  __syncthreads();
}

/** @brief Blocking kernel - Quiet per push */
template <typename T>
__global__ void PushKernel(BenchContext<T> ctx, int iters) {
  for (int i = 0; i < iters; ++i) DevicePush(ctx);
}

/** @brief NBI kernel - pipelined pushes, single Quiet at end */
template <typename T>
__global__ void PushNBIKernel(BenchContext<T> ctx, int iters) {
  for (int i = 0; i < iters; ++i) DevicePushNBI(ctx);
  if (threadIdx.x == 0) Quiet(ctx.posted, ctx.completed);
  __syncthreads();
}

template <typename T>
struct Bench {
  Queue<T> queue;
  uint64_t* posted;
  uint64_t* completed;
  cudaStream_t stream;

  Bench() {
    CUDA_CHECK(cudaMallocManaged(&posted, sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocManaged(&completed, sizeof(uint64_t)));
    CUDA_CHECK(cudaStreamCreate(&stream));
    *posted = 0;
    *completed = 0;
  }

  ~Bench() {
    CUDA_CHECK(cudaFree(posted));
    CUDA_CHECK(cudaFree(completed));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  BenchContext<T> GetContext() { return {&queue, posted, completed}; }

  void Complete() { reinterpret_cast<cuda::std::atomic<uint64_t>*>(completed)->fetch_add(1, cuda::std::memory_order_relaxed); }

  template <typename Kernel>
  BenchResult Run(Kernel kernel, int iters) {
    *posted = 0;
    *completed = 0;
    auto ctx = GetContext();

    cudaLaunchConfig_t cfg{.gridDim = {1}, .blockDim = {256}, .stream = stream};
    auto start = std::chrono::high_resolution_clock::now();
    LAUNCH_KERNEL(&cfg, kernel, ctx, iters);

    ::Run([&]() -> Coro<> {
      for (int done = 0; done < iters;) {
        T item;
        if (queue.Pop(item)) {
          Complete();
          ++done;
          if (done == 1 || done % 64 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            double lat = std::chrono::duration<double, std::micro>(now - start).count() / done;
            printf("\r[%.3fs] ops=%d/%d size=%zu lat=%.3fus\033[K", elapsed, done, iters, sizeof(T), lat);
            fflush(stdout);
          }
        }
        co_await YieldAwaiter{};
      }
    }());
    printf("\r\033[K");
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();

    double time_us = std::chrono::duration<double, std::micro>(end - start).count();
    double lat_us = time_us / iters;
    double mops = static_cast<double>(iters) / time_us;
    return {sizeof(T), lat_us, mops};
  }
};

struct BenchResultPair {
  size_t size;
  BenchResult blocking;
  BenchResult nbi;
};

template <typename T>
BenchResultPair RunBench(const Options& opts) {
  Bench<T> bench;
  // Warmup
  for (int i = 0; i < opts.warmup; ++i) bench.Run(PushKernel<T>, opts.iters);

  // Blocking benchmark
  BenchResult blocking{sizeof(T), 0, 0};
  for (int i = 0; i < opts.repeat; ++i) {
    auto r = bench.Run(PushKernel<T>, opts.iters);
    blocking.time_us += r.time_us;
    blocking.mops += r.mops;
  }
  blocking.time_us /= opts.repeat;
  blocking.mops /= opts.repeat;

  // NBI benchmark
  BenchResult nbi{sizeof(T), 0, 0};
  for (int i = 0; i < opts.repeat; ++i) {
    auto r = bench.Run(PushNBIKernel<T>, opts.iters);
    nbi.time_us += r.time_us;
    nbi.mops += r.mops;
  }
  nbi.time_us /= opts.repeat;
  nbi.mops /= opts.repeat;

  return {sizeof(T), blocking, nbi};
}

void PrintSummary(const char* title, const Options& opts, const std::vector<BenchResultPair>& results) {
  printf("#\n# %s\n#\n", title);
  printf("# iters per run: %d\n", opts.iters);
  printf("# repeat: %d\n", opts.repeat);
  printf("# warmup: %d\n#\n", opts.warmup);
  printf("# Pattern: GPU -> Queue -> CPU\n#\n");
  printf("%12s %14s %10s %14s %10s\n", "size", "Blocking", "Lat(us)", "NBI", "Lat(us)");
  for (const auto& r : results) {
    printf("%12zu %14.2f %10.2f %14.2f %10.2f\n", r.size, r.blocking.mops, r.blocking.time_us, r.nbi.mops, r.nbi.time_us);
  }
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

  printf("Running Test...\n");
  fflush(stdout);
  Test test;
  test.Run(opts.warmup);

  printf("Running Bench...\n");
  fflush(stdout);

  std::vector<BenchResultPair> results;
  results.push_back(RunBench<Payload<8>>(opts));
  results.push_back(RunBench<Payload<16>>(opts));
  results.push_back(RunBench<Payload<32>>(opts));
  results.push_back(RunBench<Payload<64>>(opts));
  results.push_back(RunBench<Payload<128>>(opts));
  results.push_back(RunBench<Payload<256>>(opts));
  results.push_back(RunBench<Payload<512>>(opts));
  results.push_back(RunBench<Payload<1024>>(opts));

  PrintSummary("MPSC Queue Benchmark", opts, results);
}
