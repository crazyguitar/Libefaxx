/**
 * @file ib.h
 * @brief Base RDMA peer class for EFA communication using ibverbs
 */
#pragma once

#include <affinity/affinity.h>
#include <affinity/taskset.h>
#include <bootstrap/mpi/mpi.h>
#include <io/io.h>
#include <rdma/ib/buffer.h>
#include <rdma/ib/channel.h>
#include <rdma/ib/efa.h>
#include <rdma/ib/selector.h>

#include <array>
#include <cstring>
#include <memory>
#include <vector>

namespace ib {

/**
 * @brief Base RDMA peer for EFA communication using ibverbs
 */
class Peer : private NoCopy {
 public:
  using AddrBuffer = std::array<char, kMaxAddrSize>;
  template <typename T>
  using Buffers = std::vector<std::unique_ptr<T>>;

  MPI& mpi;
  const GPUloc& loc;
  int device = -1;
  std::vector<EFA> efas;
  std::vector<std::vector<Channel>> channels;
  std::vector<std::vector<AddrBuffer>> addrs;

  Peer() : mpi(MPI::Get()), loc(GPUloc::Get()) {
    const auto world_size = mpi.GetWorldSize();
    const auto rank = mpi.GetWorldRank();
    device = mpi.GetLocalRank();
    if (rank == 0 && !g_device_info_printed) {
      auto& aff = loc.GetGPUAffinity()[device];
      std::cout << fmt::format("CUDA Device {}: \"{}\"\n", device, aff.prop.name) << aff << std::flush;
      g_device_info_printed = true;
    }
    addrs.resize(world_size);
    auto& affinity = loc.GetGPUAffinity()[device];
    Taskset::Set(affinity.cores[device]->logical_index);
    IO::Get().Set(std::make_unique<IBSelector>());
    efas.reserve(affinity.efas.size());
    for (auto e : affinity.efas) efas.emplace_back(EFA(e));
  }

  void Exchange() {
    const auto my_rank = mpi.GetWorldRank();
    const auto world_size = mpi.GetWorldSize();
    const size_t total_size = static_cast<size_t>(world_size) * kMaxAddrSize;
    std::vector<char> recvbuf(total_size, 0);
    for (const auto& e : efas) {
      MPI_Allgather(e.GetAddr(), kMaxAddrSize, MPI_BYTE, recvbuf.data(), kMaxAddrSize, MPI_BYTE, MPI_COMM_WORLD);
      for (int r = 0; r < world_size; ++r) {
        AddrBuffer addr_buf{};
        std::memcpy(addr_buf.data(), recvbuf.data() + r * kMaxAddrSize, kMaxAddrSize);
        addrs[r].push_back(std::move(addr_buf));
      }
    }
    if (my_rank == 0) {
      for (int r = 0; r < world_size; ++r) {
        SPDLOG_DEBUG("rank: {}", r);
        for ([[maybe_unused]] const auto& addr : addrs[r]) SPDLOG_DEBUG("  {}", EFA::Addr2Str(addr.data()));
      }
    }
  }

  void Connect() {
    const auto world_size = mpi.GetWorldSize();
    const auto rank = mpi.GetWorldRank();
    channels.resize(world_size);
    for (int i = 0; i < world_size; ++i) {
      if (i == rank) continue;
      auto& remotes = addrs[i];
      const auto n = std::min(remotes.size(), efas.size());
      for (size_t j = 0; j < n; ++j) channels[i].emplace_back(Channel{std::addressof(efas[j]), remotes[j].data()});
    }
  }

  template <typename T>
  void Handshake(Buffers<T>& write_bufs, Buffers<T>& read_bufs) {
    const auto world_size = mpi.GetWorldSize();
    const auto rank = mpi.GetWorldRank();
    for (int peer = 0; peer < world_size; ++peer) {
      if (peer == rank) continue;
      const auto local_rma = read_bufs[peer]->GetLocalRmaIovs();
      const size_t sz = local_rma.size() * sizeof(ib_rma_iov);
      std::vector<ib_rma_iov> peer_iovs(local_rma.size());
      MPI_Sendrecv(local_rma.data(), sz, MPI_BYTE, peer, 0, peer_iovs.data(), sz, MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      write_bufs[peer]->SetRemoteRmaIovs(peer, peer_iovs);
    }
  }

  size_t GetBandwidth(int ch = 0) const noexcept {
    // EFA link speed: 100 Gbps per device
    return 100ULL * 1000 * 1000 * 1000;
  }

  size_t GetTotalBandwidth() const noexcept { return efas.size() * GetBandwidth(0); }
};

}  // namespace ib
