/**
 * @file fabric.cuh
 * @brief Fabric RDMA benchmark using unified template
 */
#pragma once

#include <bootstrap/mpi/fabric.h>
#include <rdma/fabric/memory.h>

#include <bench/mpi/bench.cuh>

// Mark host buffer types
template <>
struct BufferTraits<fi::HostBuffer> {
  static constexpr bool is_host = true;
};
template <>
struct BufferTraits<fi::SymmetricHostMemory> {
  static constexpr bool is_host = true;
};

/**
 * @brief Fabric-specific buffer creation traits
 */
struct FabricTraits {
  template <typename T>
  static std::unique_ptr<T>
  MakeBuffer(std::vector<fi::EFA>& efas, std::vector<std::vector<fi::Channel>>& channels, int device, size_t size, int world_size) {
    if constexpr (std::is_same_v<T, fi::HostBuffer> || std::is_same_v<T, fi::DeviceDMABuffer> || std::is_same_v<T, fi::DevicePinBuffer>) {
      return std::make_unique<T>(efas, channels, device, size);
    } else {
      return std::make_unique<T>(efas, channels, device, size, world_size);
    }
  }
};

using FabricBench = BenchBase<fi::Peer, FabricTraits>;
