#include <io/io.h>
#include <io/common.h>
#include <affinity/taskset.h>
#include <rdma/fabric/efa.h>
#include <rdma/fabric/buffer.h>
#include <bootstrap/mpi/fabric.h>
#include <affinity/affinity.h>
#include <device/common.cuh>
#include <spdlog/spdlog.h>
#include <iostream>
#include <array>
#include <limits>

struct FabricPeer : public Peer {
  FabricPeer() : Peer() {}

  void Bootstrap() {
    const auto rank = mpi.GetWorldRank();
    const auto world_size = mpi.GetWorldSize();
    SPDLOG_INFO("Rank {}: Exchange", rank);
    Exchange();
    SPDLOG_INFO("Rank {}: Connect", rank);
    Connect();
    SPDLOG_INFO("Rank {}: Handshake", rank);
    std::vector<std::unique_ptr<HostBuffer>> write_buffers(world_size);
    std::vector<std::unique_ptr<HostBuffer>> read_buffers(world_size);
    for (int i = 0; i < world_size; ++i) {
      if (i == rank) continue;
      write_buffers[i] = std::make_unique<HostBuffer>(channels[i], 1024);
      read_buffers[i] = std::make_unique<HostBuffer>(channels[i], 1024);
    }
    Handshake(write_buffers, read_buffers);

    SPDLOG_INFO("Rank {}: Bootstrap complete", rank);
  }

  friend std::ostream& operator<<(std::ostream &os, const FabricPeer &peer) {
    for (auto& e : peer.efas) os << EFA::Addr2Str(e.GetAddr()) << std::endl;
    return os;
  }
};

int main(int argc, char *argv[]) {
  auto peer = FabricPeer();
  peer.Bootstrap();
}
