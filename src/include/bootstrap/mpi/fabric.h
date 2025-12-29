/**
 * @file fabric.h
 * @brief Base RDMA peer class for EFA communication (CPU-only)
 */
#pragma once

#include <affinity/affinity.h>
#include <affinity/taskset.h>
#include <bootstrap/mpi/mpi.h>
#include <io/io.h>
#include <rdma/fabric/buffer.h>
#include <rdma/fabric/channel.h>
#include <rdma/fabric/efa.h>
#include <rdma/fabric/selector.h>

#include <array>
#include <cstring>
#include <memory>
#include <vector>

/**
 * @brief Base RDMA peer for EFA communication
 *
 * Manages MPI, EFA endpoints, and RDMA channels. Provides connection setup
 * and RMA handshake for multi-rank communication.
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
    static bool printed = false;
    if (rank == 0 && !printed) {
      auto& aff = loc.GetGPUAffinity()[device];
      std::cout << fmt::format("CUDA Device {}: \"{}\"\n", device, aff.prop.name) << aff << std::flush;
      printed = true;
    }
    addrs.resize(world_size);
    auto& affinity = loc.GetGPUAffinity()[device];
    Taskset::Set(affinity.cores[device]->logical_index);
    IO::Get().Set(std::make_unique<FabricSelector>());
    efas.reserve(affinity.efas.size());
    for (auto e : affinity.efas) efas.emplace_back(EFA(e));
  }

  /** @brief Exchange EFA addresses across all ranks via MPI_Allgather */
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

  /** @brief Create RDMA channels to all remote peers */
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

  /**
   * @brief Exchange IPC handles for intra-node communication
   *
   * Exchanges CUDA IPC handles among ranks on the same node and opens
   * remote memory handles for direct GPU-to-GPU access.
   *
   * @tparam T SymmetricMemory type (must be DeviceDMABuffer-based)
   * @param bufs Buffer set to exchange IPC handles for
   */
  template <typename T>
  void HandshakeIPC(Buffers<T>& bufs) {
    const int local_size = mpi.GetLocalSize();
    const int local_rank = mpi.GetLocalRank();
    const int world_rank = mpi.GetWorldRank();

    if (local_size <= 1) return;

    std::vector<cudaIpcMemHandle_t> all_handles(local_size);
    std::vector<int> local_world_ranks(local_size);

    cudaIpcMemHandle_t local_handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&local_handle, bufs[world_rank]->Data()));

    MPI_Comm local = mpi.GetLocalComm();
    size_t hsz = sizeof(cudaIpcMemHandle_t);
    MPI_Allgather(&world_rank, 1, MPI_INT, local_world_ranks.data(), 1, MPI_INT, local);
    MPI_Allgather(&local_handle, hsz, MPI_BYTE, all_handles.data(), hsz, MPI_BYTE, local);

    bufs[world_rank]->OpenIPCHandles(all_handles, local_world_ranks, local_rank);
  }

  /**
   * @brief Exchange RMA keys for a single buffer set (symmetric pattern)
   *
   * Each buffer exchanges its own RMA info with the corresponding peer buffer.
   *
   * @tparam T Buffer type (must have GetLocalRmaIovs and SetRemoteRmaIovs)
   * @param bufs Buffer set to exchange keys for
   */
  template <typename T>
  void Handshake(Buffers<T>& bufs) {
    const auto world_size = mpi.GetWorldSize();
    const auto rank = mpi.GetWorldRank();
    for (int peer = 0; peer < world_size; ++peer) {
      if (peer == rank) continue;
      const auto local_rma = bufs[peer]->GetLocalRmaIovs();
      const size_t sz = local_rma.size() * sizeof(struct fi_rma_iov);
      std::vector<struct fi_rma_iov> peer_iovs(local_rma.size());
      MPI_Sendrecv(local_rma.data(), sz, MPI_BYTE, peer, 0, peer_iovs.data(), sz, MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      bufs[peer]->SetRemoteRmaIovs(peer, peer_iovs);
    }
  }

  /**
   * @brief Exchange RMA keys for write/read buffer pair
   *
   * For RDMA write: write_bufs[peer] writes to peer's read_bufs[rank].
   * Exchanges read buffer RMA info and sets it on write buffers.
   *
   * @tparam T Buffer type
   * @param write_bufs Write buffers (receives remote read buffer RMA info)
   * @param read_bufs Read buffers (sends local RMA info)
   */
  template <typename T>
  void Handshake(Buffers<T>& write_bufs, Buffers<T>& read_bufs) {
    const auto world_size = mpi.GetWorldSize();
    const auto rank = mpi.GetWorldRank();
    for (int peer = 0; peer < world_size; ++peer) {
      if (peer == rank) continue;
      const auto local_rma = read_bufs[peer]->GetLocalRmaIovs();
      const size_t sz = local_rma.size() * sizeof(struct fi_rma_iov);
      std::vector<struct fi_rma_iov> peer_iovs(local_rma.size());
      MPI_Sendrecv(local_rma.data(), sz, MPI_BYTE, peer, 0, peer_iovs.data(), sz, MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      write_bufs[peer]->SetRemoteRmaIovs(peer, peer_iovs);
    }
  }

  /**
   * @brief Get link speed for a specific EFA channel
   * @param ch Channel index
   * @return Link speed in bits/sec
   */
  size_t GetBandwidth(int ch = 0) const noexcept { return efas[ch].GetInfo()->nic->link_attr->speed; }

  /** @brief Get total link speed across all EFA channels in bits/sec */
  size_t GetTotalBandwidth() const noexcept {
    size_t total = 0;
    for (const auto& efa : efas) total += efa.GetInfo()->nic->link_attr->speed;
    return total;
  }
};
