/**
 * @file ib.h
 * @brief RDMA peer using ibverbs EFA (mirrors fabric.h structure)
 */
#pragma once

#include <affinity/affinity.h>
#include <bootstrap/mpi/mpi.h>
#include <rdma/ib/efa.h>

#include <vector>

namespace ib {

/**
 * @brief RDMA peer using ib::EFA class for communication
 */
class Peer : private NoCopy {
 public:
  MPI& mpi;
  const GPUloc& loc;
  int device = -1;
  std::vector<EFA> efas;
  std::vector<std::vector<char>> remote_addrs;  // [rank][addr bytes]
  std::vector<std::vector<ibv_ah*>> ahs;        // [rank][device]

  Peer() : mpi(MPI::Get()), loc(GPUloc::Get()) {
    const auto world_size = mpi.GetWorldSize();
    device = mpi.GetLocalRank();
    remote_addrs.resize(world_size);
    ahs.resize(world_size);

    auto& affinity = loc.GetGPUAffinity()[device];
    for (auto e : affinity.efas) {
      efas.emplace_back(EFA(e));
    }
  }

  ~Peer() {
    for (auto& rank_ahs : ahs) {
      for (auto* ah : rank_ahs)
        if (ah) ibv_destroy_ah(ah);
    }
  }

  void Exchange() {
    const auto world_size = mpi.GetWorldSize();
    for (size_t d = 0; d < efas.size(); ++d) {
      std::vector<char> all_addrs(world_size * kAddrSize);
      MPI_Allgather(efas[d].GetAddr(), kAddrSize, MPI_BYTE, all_addrs.data(), kAddrSize, MPI_BYTE, MPI_COMM_WORLD);

      for (int r = 0; r < world_size; ++r) {
        remote_addrs[r].insert(remote_addrs[r].end(), all_addrs.begin() + r * kAddrSize, all_addrs.begin() + (r + 1) * kAddrSize);
      }
    }
  }

  void Connect() {
    const auto rank = mpi.GetWorldRank();
    const auto world_size = mpi.GetWorldSize();

    for (int r = 0; r < world_size; ++r) {
      if (r == rank) continue;
      for (size_t d = 0; d < efas.size(); ++d) {
        const char* addr = remote_addrs[r].data() + d * kAddrSize;
        auto* ah = ib_av_insert(efas[d].GetAV(), addr);
        IB_CHECK(ah);
        ahs[r].push_back(ah);
      }
    }
  }
};

}  // namespace ib
