#pragma once

#include <cstdint>
#include <cuda/std/atomic>
#include <device/common.cuh>

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
 * - Push is __device__ only (GPU producers, CPU consumers pattern).
 * - Pop is __host__ only.
 *
 * @tparam T The type of elements stored in the queue. Must be trivially copyable.
 */
template <typename T>
struct Queue {
  struct Cell {
    cuda::std::atomic<uint64_t> sequence;
    T data;
  };

  Cell* buffer;
  uint64_t mask;
  cuda::std::atomic<uint64_t> head;
  cuda::std::atomic<uint64_t> tail;

  /**
   * Construct a new Queue with the given size.
   * @param size Must be a power of two. Aborts if not.
   */
  __host__ Queue(size_t size) : mask{size - 1} {
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
   * This is a device-only function for GPU producers.
   *
   * @param data The element to push.
   * @return true if the element was successfully pushed, false if the queue is full.
   */
  __device__ bool Push(const T& data) {
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
  __host__ bool Pop(T& data) {
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
