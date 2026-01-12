/**
 * @file ib.cuh
 * @brief IB (ibverbs) RDMA benchmark using unified template
 */
#pragma once

#include <bootstrap/mpi/ib.h>
#include <rdma/ib/memory.h>

#include <bench/mpi/bench.cuh>

// Mark host buffer types
template <>
struct BufferTraits<ib::HostBuffer> {
  static constexpr bool is_host = true;
};
template <>
struct BufferTraits<ib::SymmetricHostMemory> {
  static constexpr bool is_host = true;
};

/**
 * @brief IB-specific buffer creation traits
 */
struct IBTraits {
  template <typename T>
  static std::unique_ptr<T>
  MakeBuffer(std::vector<ib::EFA>& efas, std::vector<std::vector<ib::Channel>>& channels, int device, size_t size, int world_size) {
    if constexpr (std::is_same_v<T, ib::HostBuffer> || std::is_same_v<T, ib::DeviceDMABuffer>) {
      return std::make_unique<T>(efas, channels, device, size);
    } else {
      return std::make_unique<T>(efas, channels, device, size, world_size);
    }
  }
};

using IBBench = BenchBase<ib::Peer, IBTraits>;
