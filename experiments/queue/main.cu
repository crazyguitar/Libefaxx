/**
 * @file main.cu
 * @brief MPMC Queue Benchmark - GPU producer, CPU consumer
 *
 * Benchmarks three queue implementations:
 *   - Queue(Managed):  cudaMallocManaged - unified memory with page migration
 *   - PinnedQueue:     cudaHostAlloc + cudaHostGetDevicePointer - GPU writes over PCIe
 *   - GdrQueue:        GDRCopy - GPU memory with CPU access via PCIe BAR1
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

template <typename Q>
struct BenchContext {
  Q* queue;
  uint64_t* posted;
  uint64_t* completed;
};

/** @brief Push with Quiet per operation (blocking) */
template <typename T>
__device__ __forceinline__ void DevicePush(BenchContext<T> ctx) {
  __threadfence_system();
  if (threadIdx.x == 0) {
    typename T::value_type payload{};
    while (!ctx.queue->Push(payload)) __threadfence_system();
    atomicAdd(reinterpret_cast<unsigned long long*>(ctx.posted), 1ULL);
    fi::Fence();
    fi::Quiet(ctx.posted, ctx.completed);
  }
  __syncthreads();
}

/** @brief Push without waiting (NBI - non-blocking interface) */
template <typename T>
__device__ __forceinline__ void DevicePushNBI(BenchContext<T> ctx) {
  __threadfence_system();
  if (threadIdx.x == 0) {
    typename T::value_type payload{};
    while (!ctx.queue->Push(payload)) __threadfence_system();
    atomicAdd(reinterpret_cast<unsigned long long*>(ctx.posted), 1ULL);
    fi::Fence();
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
  if (threadIdx.x == 0) fi::Quiet(ctx.posted, ctx.completed);
  __syncthreads();
}

// Queue allocation traits
template <typename Q>
struct QueueTraits;

template <typename T>
struct QueueTraits<Queue<T>> {
  static constexpr const char* Name = "Queue(Managed)";
  static Queue<T>* Alloc() { return new Queue<T>(); }
  static void Free(Queue<T>* q) { delete q; }
};

template <typename T>
struct QueueTraits<PinnedQueue<T>> {
  static constexpr const char* Name = "PinnedQueue";
  static PinnedQueue<T>* Alloc() { return new PinnedQueue<T>(); }
  static void Free(PinnedQueue<T>* q) { delete q; }
};

template <typename T>
struct QueueTraits<GdrQueue<T>> {
  static constexpr const char* Name = "GdrQueue";
  static GdrQueue<T>* Alloc() { return new GdrQueue<T>(); }
  static void Free(GdrQueue<T>* q) { delete q; }
};

template <typename Q>
struct Bench {
  using T = typename Q::value_type;
  using Traits = QueueTraits<Q>;
  Q* queue;
  uint64_t* posted;
  uint64_t* completed;
  cudaStream_t stream;

  Bench() {
    queue = Traits::Alloc();
    CUDA_CHECK(cudaMallocManaged(&posted, sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocManaged(&completed, sizeof(uint64_t)));
    CUDA_CHECK(cudaStreamCreate(&stream));
    *posted = 0;
    *completed = 0;
  }

  ~Bench() {
    Traits::Free(queue);
    CUDA_CHECK(cudaFree(posted));
    CUDA_CHECK(cudaFree(completed));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  BenchContext<Q> GetContext() { return {queue, posted, completed}; }

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
        if (queue->Pop(item)) {
          Complete();
          ++done;
          if (done == 1 || done % 64 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            double lat = std::chrono::duration<double, std::micro>(now - start).count() / done;
            printf("\r[%s] [%.3fs] ops=%d/%d size=%zu lat=%.3fus\033[K", Traits::Name, elapsed, done, iters, sizeof(T), lat);
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

template <typename Q>
BenchResultPair RunBench(const Options& opts) {
  using T = typename Q::value_type;
  Bench<Q> bench;
  // Warmup
  for (int i = 0; i < opts.warmup; ++i) bench.Run(PushKernel<Q>, opts.iters);

  // Blocking benchmark
  BenchResult blocking{sizeof(T), 0, 0};
  for (int i = 0; i < opts.repeat; ++i) {
    auto r = bench.Run(PushKernel<Q>, opts.iters);
    blocking.time_us += r.time_us;
    blocking.mops += r.mops;
  }
  blocking.time_us /= opts.repeat;
  blocking.mops /= opts.repeat;

  // NBI benchmark
  BenchResult nbi{sizeof(T), 0, 0};
  for (int i = 0; i < opts.repeat; ++i) {
    auto r = bench.Run(PushNBIKernel<Q>, opts.iters);
    nbi.time_us += r.time_us;
    nbi.mops += r.mops;
  }
  nbi.time_us /= opts.repeat;
  nbi.mops /= opts.repeat;

  return {sizeof(T), blocking, nbi};
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

  // Queue (Managed)
  std::vector<BenchResultPair> managed;
  managed.push_back(RunBench<Queue<Payload<8>>>(opts));
  managed.push_back(RunBench<Queue<Payload<16>>>(opts));
  managed.push_back(RunBench<Queue<Payload<32>>>(opts));
  managed.push_back(RunBench<Queue<Payload<64>>>(opts));
  managed.push_back(RunBench<Queue<Payload<128>>>(opts));
  managed.push_back(RunBench<Queue<Payload<256>>>(opts));
  managed.push_back(RunBench<Queue<Payload<512>>>(opts));
  managed.push_back(RunBench<Queue<Payload<1024>>>(opts));

  // PinnedQueue
  std::vector<BenchResultPair> pinned;
  pinned.push_back(RunBench<PinnedQueue<Payload<8>>>(opts));
  pinned.push_back(RunBench<PinnedQueue<Payload<16>>>(opts));
  pinned.push_back(RunBench<PinnedQueue<Payload<32>>>(opts));
  pinned.push_back(RunBench<PinnedQueue<Payload<64>>>(opts));
  pinned.push_back(RunBench<PinnedQueue<Payload<128>>>(opts));
  pinned.push_back(RunBench<PinnedQueue<Payload<256>>>(opts));
  pinned.push_back(RunBench<PinnedQueue<Payload<512>>>(opts));
  pinned.push_back(RunBench<PinnedQueue<Payload<1024>>>(opts));

  // GdrQueue
  std::vector<BenchResultPair> gdr;
  gdr.push_back(RunBench<GdrQueue<Payload<8>>>(opts));
  gdr.push_back(RunBench<GdrQueue<Payload<16>>>(opts));
  gdr.push_back(RunBench<GdrQueue<Payload<32>>>(opts));
  gdr.push_back(RunBench<GdrQueue<Payload<64>>>(opts));
  gdr.push_back(RunBench<GdrQueue<Payload<128>>>(opts));
  gdr.push_back(RunBench<GdrQueue<Payload<256>>>(opts));
  gdr.push_back(RunBench<GdrQueue<Payload<512>>>(opts));
  gdr.push_back(RunBench<GdrQueue<Payload<1024>>>(opts));

  // Print grouped summary by mode
  printf("\n#\n# Summary: Blocking Mode (sync after each push)\n#\n");
  printf("# iters per run: %d, repeat: %d, warmup: %d\n", opts.iters, opts.repeat, opts.warmup);
  printf("# Pattern: GPU -> Queue -> CPU\n#\n");
  printf("%12s %14s %10s %14s %10s %14s %10s\n", "size", "Managed", "Lat(us)", "Pinned", "Lat(us)", "GdrQueue", "Lat(us)");
  for (size_t i = 0; i < managed.size(); ++i) {
    printf(
        "%12zu %14.2f %10.2f %14.2f %10.2f %14.2f %10.2f\n", managed[i].size, managed[i].blocking.mops, managed[i].blocking.time_us,
        pinned[i].blocking.mops, pinned[i].blocking.time_us, gdr[i].blocking.mops, gdr[i].blocking.time_us
    );
  }

  printf("\n#\n# Summary: NBI Mode (batch push, sync at end)\n#\n");
  printf("# iters per run: %d, repeat: %d, warmup: %d\n", opts.iters, opts.repeat, opts.warmup);
  printf("# Pattern: GPU -> Queue -> CPU\n#\n");
  printf("%12s %14s %10s %14s %10s %14s %10s\n", "size", "Managed", "Lat(us)", "Pinned", "Lat(us)", "GdrQueue", "Lat(us)");
  for (size_t i = 0; i < managed.size(); ++i) {
    printf(
        "%12zu %14.2f %10.2f %14.2f %10.2f %14.2f %10.2f\n", managed[i].size, managed[i].nbi.mops, managed[i].nbi.time_us, pinned[i].nbi.mops,
        pinned[i].nbi.time_us, gdr[i].nbi.mops, gdr[i].nbi.time_us
    );
  }

  printf("#\n# Benchmark complete.\n");
}
