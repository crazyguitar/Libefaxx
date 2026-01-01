#pragma once

#include <gdrapi.h>

#include <cstdint>
#include <cuda/std/atomic>
#include <device/common.cuh>

/// Cache line size for alignment (128 bytes covers both CPU 64B and GPU 128B cache lines)
inline constexpr size_t kCacheLineSize = 128;

#define GDR_CHECK(exp)                                                            \
  do {                                                                            \
    int rc = (exp);                                                               \
    if (rc != 0) {                                                                \
      SPDLOG_CRITICAL("[{}:{}] " #exp " failed with {}", __FILE__, __LINE__, rc); \
      exit(1);                                                                    \
    }                                                                             \
  } while (0)

/**
 * MPMC Queue Implementations for GPU→CPU Communication
 *
 * This header provides three queue implementations optimized for different
 * GPU-to-CPU communication patterns:
 *
 * | Queue Type   | GPU Write        | CPU Read         | Best For                    |
 * |--------------|------------------|------------------|-----------------------------|
 * | Queue        | Local VRAM       | Page migration   | Sequential bulk transfers   |
 * | PinnedQueue  | PCIe to host     | Local DRAM       | CPU-primary workloads       |
 * | GdrQueue     | Local VRAM       | PCIe BAR1        | Concurrent GPU+CPU access   |
 *
 * Performance Characteristics (p5.48xlarge, H100):
 *
 *   Throughput (Sequential GPU push, then CPU pop):
 *     Queue(Managed)    1.1 Mops/sec   - Pages migrate to GPU, fast local writes
 *     GdrQueue          0.4 Mops/sec   - Stable, no migration overhead
 *     PinnedQueue       0.03 Mops/sec  - Every GPU atomic over PCIe (~40x slower)
 *
 *   CPU Polling Latency (pure CPU read speed):
 *     Queue(Managed)    ~1.8 ns/op     - Cached in CPU DRAM after migration
 *     PinnedQueue       ~1.7 ns/op     - Already in host DRAM
 *     GdrQueue          ~1360 ns/op    - PCIe BAR1 round-trip to GPU VRAM
 *
 *   Concurrent (GPU push + CPU pop simultaneously):
 *     GdrQueue          12.5 us/op     - No page migration, predictable latency
 *     PinnedQueue       14.1 us/op     - No migration, but slow GPU writes
 *     Queue(Managed)    15.1 us/op     - Page migration ping-pong overhead
 *
 * Key Insight: GdrQueue wins in concurrent scenarios despite higher CPU polling
 * latency because it avoids page migration thrashing when both GPU and CPU are
 * actively accessing the queue.
 */

/**
 * Lock-free bounded Multi-Producer Multi-Consumer (MPMC) queue.
 *
 * Based on Dmitry Vyukov's bounded MPMC queue algorithm.
 * See: https://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue
 *
 * Algorithm overview:
 * - Each cell has a `sequence` field that tracks the cell's state.
 * - Producers and consumers use compare-and-swap (CAS) on `head`/`tail` to claim positions.
 * - The `sequence` field transitions through states to coordinate access:
 *
 *   For producers (Push):
 *   - Initial state: sequence == pos (cell is ready for writing)
 *   - After write:   sequence == pos + 1 (cell contains valid data)
 *
 *   For consumers (Pop):
 *   - Ready state:   sequence == pos + 1 (cell has data to read)
 *   - After read:    sequence == pos + mask + 1 (cell is recycled for next round)
 *
 * Wraparound handling:
 * - The sequence advances by (mask + 1) each cycle, so after a complete wrap,
 *   sequence values are: 0, size, 2*size, 3*size, etc. for a given cell.
 * - This allows detecting whether a cell is from the current cycle or a previous one.
 *
 * Memory ordering:
 * - Producers use acquire on sequence load (to see consumer's release store)
 * - Producers use release on sequence store (to make data visible to consumers)
 * - Consumers use acquire on sequence load (to see producer's release store)
 * - Consumers use release on sequence store (to make cell available to producers)
 * - CAS operations on head/tail use relaxed memory order since the sequence
 *   field provides the synchronization.
 *
 * Usage constraints:
 * - Queue size MUST be a power of two (validated in constructor, aborts if invalid).
 * - Push is __host__ __device__ (both GPU and CPU can produce).
 * - Pop is __host__ only (CPU consumers).
 *
 * @tparam T The type of elements stored in the queue. Must be trivially copyable.
 */
template <typename T>
struct alignas(kCacheLineSize) Queue {
  using value_type = T;
  /// Default queue capacity (must be power of two)
  static constexpr size_t kDefaultSize = 8192;

  struct Cell {
    cuda::std::atomic<uint64_t> sequence;
    T data;
  };

  alignas(kCacheLineSize) Cell* buffer;                      ///< Cell buffer
  uint64_t mask;                                             ///< Size mask for index wrapping
  alignas(kCacheLineSize) cuda::std::atomic<uint64_t> head;  ///< Producer index (separate cache line to avoid false sharing)
  alignas(kCacheLineSize) cuda::std::atomic<uint64_t> tail;  ///< Consumer index (separate cache line to avoid false sharing)

  /**
   * Construct a new Queue with the given size.
   * @param size Must be a power of two. Aborts if not.
   */
  __host__ Queue(size_t size = kDefaultSize) : mask{size - 1} {
    // Validate that size is a power of two (always checked, even in release builds)
    if (size == 0 || (size & (size - 1)) != 0) {
      fprintf(stderr, "Queue size must be a positive power of two, got: %zu\n", size);
      exit(1);
    }

    CUDA_CHECK(cudaMallocManaged(&buffer, size * sizeof(Cell)));

    // Initialize each cell's sequence to its index (ready for first round of writes)
    for (size_t i = 0; i < size; ++i) {
      buffer[i].sequence.store(i, cuda::std::memory_order_relaxed);
    }
    head.store(0, cuda::std::memory_order_relaxed);
    tail.store(0, cuda::std::memory_order_relaxed);
  }

  __host__ ~Queue() { CUDA_CHECK(cudaFree(buffer)); }

  /**
   * Push an element to the queue (non-blocking).
   *
   * @param data The element to push.
   * @return true if the element was successfully pushed, false if the queue is full.
   */
  __host__ __device__ __forceinline__ bool Push(const T& data) {
    uint64_t pos = head.load(cuda::std::memory_order_relaxed);

    for (;;) {
      Cell* cell = &buffer[pos & mask];
      uint64_t seq = cell->sequence.load(cuda::std::memory_order_acquire);
      int64_t diff = static_cast<int64_t>(seq) - static_cast<int64_t>(pos);

      if (diff == 0) {
        // Cell is ready for writing at this position.
        // Try to claim this position by advancing head.
        if (head.compare_exchange_weak(pos, pos + 1, cuda::std::memory_order_relaxed)) {
          // Successfully claimed the position. Write data and publish.
          cell->data = data;
          cell->sequence.store(pos + 1, cuda::std::memory_order_release);
          return true;
        }
        // CAS failed, another producer claimed this position. Loop will retry with updated pos.
      } else if (diff < 0) {
        // Cell's sequence is behind, meaning the queue is full (this cell hasn't been
        // consumed yet from the previous cycle).
        return false;
      } else {
        // diff > 0: Another producer already claimed and possibly completed this position.
        // Reload head and try the next position.
        pos = head.load(cuda::std::memory_order_relaxed);
      }
    }
  }

  /**
   * Pop an element from the queue (non-blocking).
   *
   * This is a host-only function for CPU consumers.
   *
   * @param data Output parameter to receive the popped element.
   * @return true if an element was successfully popped, false if the queue is empty.
   */
  __host__ inline bool Pop(T& data) {
    uint64_t pos = tail.load(cuda::std::memory_order_relaxed);

    for (;;) {
      Cell* cell = &buffer[pos & mask];
      uint64_t seq = cell->sequence.load(cuda::std::memory_order_acquire);
      int64_t diff = static_cast<int64_t>(seq) - static_cast<int64_t>(pos + 1);

      if (diff == 0) {
        // Cell has data ready to be consumed at this position.
        // Try to claim this position by advancing tail.
        if (tail.compare_exchange_weak(pos, pos + 1, cuda::std::memory_order_relaxed)) {
          // Successfully claimed the position. Read data and recycle cell.
          data = cell->data;
          // Advance sequence by (mask + 1) to mark cell as ready for next cycle
          cell->sequence.store(pos + mask + 1, cuda::std::memory_order_release);
          return true;
        }
        // CAS failed, another consumer claimed this position. Loop will retry with updated pos.
      } else if (diff < 0) {
        // Cell's sequence indicates no data available (queue is empty from this consumer's view).
        return false;
      } else {
        // diff > 0: Another consumer already claimed this position.
        // Reload tail and try the next position.
        pos = tail.load(cuda::std::memory_order_relaxed);
      }
    }
  }
};

/**
 * GDRCopy-based lock-free bounded MPMC queue.
 *
 * Uses GDRCopy for low-latency CPU access to GPU memory via PCIe BAR1 mapping,
 * avoiding the page fault overhead of cudaMallocManaged.
 *
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │                           MEMORY ARCHITECTURE                               │
 * └─────────────────────────────────────────────────────────────────────────────┘
 *
 *   ┌─────────────────────┐                    ┌─────────────────────┐
 *   │      CPU (Host)     │                    │     GPU (Device)    │
 *   │                     │                    │                     │
 *   │  h_buffer ──────────┼───── PCIe BAR1 ───►│  d_buffer           │
 *   │  (mapped pointer)   │    (same physical  │  (device pointer)   │
 *   │                     │       memory)      │                     │
 *   │  Pop() reads here   │                    │  Push() writes here │
 *   └─────────────────────┘                    └─────────────────────┘
 *
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │                     WHY GDRCOPY IS FASTER THAN MANAGED MEMORY               │
 * └─────────────────────────────────────────────────────────────────────────────┘
 *
 *   cudaMallocManaged (Page Faults):           GDRCopy (Direct BAR1 Access):
 *   ────────────────────────────────           ─────────────────────────────
 *   CPU read ──► Page fault!                   CPU read ──► BAR1 ──► GPU VRAM
 *            ──► Trap to driver                         (direct, ~100-200ns)
 *            ──► Migrate page GPU→CPU
 *            ──► Update page tables
 *            ──► Retry access
 *            (~10-100+ µs per fault)
 *
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │                           64KB ALIGNMENT HANDLING                           │
 * └─────────────────────────────────────────────────────────────────────────────┘
 *
 *   GDRCopy maps memory at 64KB-aligned boundaries. If cudaMalloc returns
 *   an address not aligned to 64KB, there's an offset:
 *
 *   info.va (64KB aligned) ──────┬─────────────────────────────────────┐
 *                                │         offset                      │
 *   d_buffer ────────────────────┴──►┌─────────────────────────────────┤
 *                                    │      Actual buffer data         │
 *   map_ptr ─────────────────────────┤                                 │
 *                                    └─────────────────────────────────┘
 *
 *   h_buffer = map_ptr + (d_buffer - info.va)
 *
 * Usage: Allocate with cudaMallocManaged + placement new so GPU can access
 *        the queue struct itself (for d_buffer pointer):
 *
 *   GdrQueue<int>* queue;
 *   cudaMallocManaged(&queue, sizeof(GdrQueue<int>));
 *   new (queue) GdrQueue<int>(size);
 */
template <typename T>
struct alignas(kCacheLineSize) GdrQueue {
  using value_type = T;
  static constexpr size_t kDefaultSize = 8192;

  struct Cell {
    cuda::std::atomic<uint64_t> sequence;
    T data;
  };

  // Device-accessible members (must be at start for GPU kernel access)
  alignas(kCacheLineSize) Cell* d_buffer;                    ///< GPU device pointer (used by Push)
  uint64_t mask;                                             ///< Size mask for index wrapping (size - 1)
  alignas(kCacheLineSize) cuda::std::atomic<uint64_t> head;  ///< Producer index
  alignas(kCacheLineSize) cuda::std::atomic<uint64_t> tail;  ///< Consumer index

  // Host-only members (not accessed by GPU)
  Cell* h_buffer;    ///< CPU-mapped pointer via BAR1 (used by Pop)
  void* h_map_base;  ///< Base mapping address from gdr_map (for cleanup)
  gdr_t gdr;         ///< GDRCopy library handle
  gdr_mh_t mh;       ///< GDRCopy memory handle
  size_t buf_size;   ///< Buffer size in bytes

  __host__ GdrQueue(size_t size = kDefaultSize) : mask{size - 1}, buf_size{size * sizeof(Cell)} {
    if (size == 0 || (size & (size - 1)) != 0) {
      fprintf(stderr, "GdrQueue size must be a power of two, got: %zu\n", size);
      exit(1);
    }

    CUDA_CHECK(cudaMalloc(&d_buffer, buf_size));

    gdr = gdr_open();
    if (!gdr) {
      SPDLOG_CRITICAL("gdr_open() failed");
      exit(1);
    }

    GDR_CHECK(gdr_pin_buffer(gdr, (CUdeviceptr)d_buffer, buf_size, 0, 0, &mh));

    void* map_ptr = nullptr;
    GDR_CHECK(gdr_map(gdr, mh, &map_ptr, buf_size));
    h_map_base = map_ptr;

    // Handle 64KB alignment offset (see diagram above)
    gdr_info_t info;
    GDR_CHECK(gdr_get_info(gdr, mh, &info));
    h_buffer = reinterpret_cast<Cell*>(static_cast<char*>(map_ptr) + ((char*)d_buffer - (char*)info.va));

    for (size_t i = 0; i < size; ++i) {
      h_buffer[i].sequence.store(i, cuda::std::memory_order_relaxed);
    }
    head.store(0, cuda::std::memory_order_relaxed);
    tail.store(0, cuda::std::memory_order_relaxed);

    CUDA_CHECK(cudaDeviceSynchronize());
  }

  __host__ ~GdrQueue() {
    gdr_unmap(gdr, mh, h_map_base, buf_size);
    gdr_unpin_buffer(gdr, mh);
    gdr_close(gdr);
    CUDA_CHECK(cudaFree(d_buffer));
  }

  GdrQueue(const GdrQueue&) = delete;
  GdrQueue& operator=(const GdrQueue&) = delete;

  __device__ __forceinline__ bool Push(const T& data) {
    uint64_t pos = head.load(cuda::std::memory_order_relaxed);
    for (;;) {
      Cell* cell = &d_buffer[pos & mask];
      uint64_t seq = cell->sequence.load(cuda::std::memory_order_acquire);
      int64_t diff = static_cast<int64_t>(seq) - static_cast<int64_t>(pos);
      if (diff == 0) {
        if (head.compare_exchange_weak(pos, pos + 1, cuda::std::memory_order_relaxed)) {
          cell->data = data;
          cell->sequence.store(pos + 1, cuda::std::memory_order_release);
          return true;
        }
      } else if (diff < 0) {
        return false;
      } else {
        pos = head.load(cuda::std::memory_order_relaxed);
      }
    }
  }

  __host__ inline bool PushHost(const T& data) {
    uint64_t pos = head.load(cuda::std::memory_order_relaxed);
    for (;;) {
      Cell* cell = &h_buffer[pos & mask];
      uint64_t seq = cell->sequence.load(cuda::std::memory_order_acquire);
      int64_t diff = static_cast<int64_t>(seq) - static_cast<int64_t>(pos);
      if (diff == 0) {
        if (head.compare_exchange_weak(pos, pos + 1, cuda::std::memory_order_relaxed)) {
          cell->data = data;
          cell->sequence.store(pos + 1, cuda::std::memory_order_release);
          return true;
        }
      } else if (diff < 0) {
        return false;
      } else {
        pos = head.load(cuda::std::memory_order_relaxed);
      }
    }
  }

  __host__ inline bool Pop(T& data) {
    uint64_t pos = tail.load(cuda::std::memory_order_relaxed);
    for (;;) {
      Cell* cell = &h_buffer[pos & mask];
      uint64_t seq = cell->sequence.load(cuda::std::memory_order_acquire);
      int64_t diff = static_cast<int64_t>(seq) - static_cast<int64_t>(pos + 1);
      if (diff == 0) {
        if (tail.compare_exchange_weak(pos, pos + 1, cuda::std::memory_order_relaxed)) {
          data = cell->data;
          cell->sequence.store(pos + mask + 1, cuda::std::memory_order_release);
          return true;
        }
      } else if (diff < 0) {
        return false;
      } else {
        pos = tail.load(cuda::std::memory_order_relaxed);
      }
    }
  }
};

/**
 * PinnedQueue - MPMC queue using cudaHostAlloc mapped memory.
 *
 * Memory model:
 * - Buffer allocated with cudaHostAlloc(cudaHostAllocMapped) in host DRAM
 * - GPU accesses via device pointer from cudaHostGetDevicePointer (PCIe writes)
 * - CPU accesses via host pointer (local DRAM reads)
 *
 * | Allocation Method                        | GPU Write      | CPU Read    |
 * |------------------------------------------|----------------|-------------|
 * | cudaHostAlloc + cudaHostGetDevicePointer | PCIe to host   | Local DRAM  |
 */
template <typename T>
struct alignas(kCacheLineSize) PinnedQueue {
  using value_type = T;
  static constexpr size_t kDefaultSize = 8192;

  struct Cell {
    cuda::std::atomic<uint64_t> sequence;
    T data;
  };

  alignas(kCacheLineSize) Cell* d_buffer;  ///< Device pointer (used by GPU Push)
  uint64_t mask;
  alignas(kCacheLineSize) cuda::std::atomic<uint64_t> head;
  alignas(kCacheLineSize) cuda::std::atomic<uint64_t> tail;

  Cell* h_buffer;  ///< Host pointer (used by CPU Pop)

  __host__ PinnedQueue(size_t size = kDefaultSize) : mask{size - 1} {
    if (size == 0 || (size & (size - 1)) != 0) {
      fprintf(stderr, "PinnedQueue size must be a power of two, got: %zu\n", size);
      exit(1);
    }

    CUDA_CHECK(cudaHostAlloc(&h_buffer, size * sizeof(Cell), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_buffer, h_buffer, 0));

    for (size_t i = 0; i < size; ++i) {
      h_buffer[i].sequence.store(i, cuda::std::memory_order_relaxed);
    }
    head.store(0, cuda::std::memory_order_relaxed);
    tail.store(0, cuda::std::memory_order_relaxed);
  }

  __host__ ~PinnedQueue() { CUDA_CHECK(cudaFreeHost(h_buffer)); }

  PinnedQueue(const PinnedQueue&) = delete;
  PinnedQueue& operator=(const PinnedQueue&) = delete;

  __device__ __forceinline__ bool Push(const T& data) {
    uint64_t pos = head.load(cuda::std::memory_order_relaxed);
    for (;;) {
      Cell* cell = &d_buffer[pos & mask];
      uint64_t seq = cell->sequence.load(cuda::std::memory_order_acquire);
      int64_t diff = static_cast<int64_t>(seq) - static_cast<int64_t>(pos);
      if (diff == 0) {
        if (head.compare_exchange_weak(pos, pos + 1, cuda::std::memory_order_relaxed)) {
          cell->data = data;
          cell->sequence.store(pos + 1, cuda::std::memory_order_release);
          return true;
        }
      } else if (diff < 0) {
        return false;
      } else {
        pos = head.load(cuda::std::memory_order_relaxed);
      }
    }
  }

  __host__ inline bool PushHost(const T& data) {
    uint64_t pos = head.load(cuda::std::memory_order_relaxed);
    for (;;) {
      Cell* cell = &h_buffer[pos & mask];
      uint64_t seq = cell->sequence.load(cuda::std::memory_order_acquire);
      int64_t diff = static_cast<int64_t>(seq) - static_cast<int64_t>(pos);
      if (diff == 0) {
        if (head.compare_exchange_weak(pos, pos + 1, cuda::std::memory_order_relaxed)) {
          cell->data = data;
          cell->sequence.store(pos + 1, cuda::std::memory_order_release);
          return true;
        }
      } else if (diff < 0) {
        return false;
      } else {
        pos = head.load(cuda::std::memory_order_relaxed);
      }
    }
  }

  __host__ inline bool Pop(T& data) {
    uint64_t pos = tail.load(cuda::std::memory_order_relaxed);
    for (;;) {
      Cell* cell = &h_buffer[pos & mask];
      uint64_t seq = cell->sequence.load(cuda::std::memory_order_acquire);
      int64_t diff = static_cast<int64_t>(seq) - static_cast<int64_t>(pos + 1);
      if (diff == 0) {
        if (tail.compare_exchange_weak(pos, pos + 1, cuda::std::memory_order_relaxed)) {
          data = cell->data;
          cell->sequence.store(pos + mask + 1, cuda::std::memory_order_release);
          return true;
        }
      } else if (diff < 0) {
        return false;
      } else {
        pos = tail.load(cuda::std::memory_order_relaxed);
      }
    }
  }
};
