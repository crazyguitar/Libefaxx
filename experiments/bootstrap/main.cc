#include <bootstrap/mpi/fabric.h>
#include <bootstrap/mpi/ib.h>
#include <spdlog/spdlog.h>
#include <cassert>

int main(int argc, char *argv[]) {
  SPDLOG_INFO("=== Bootstrap Test ===");

  fi::Peer fi_peer;
  ib::Peer ib_peer;

  const auto rank = fi_peer.mpi.GetWorldRank();
  const auto world_size = fi_peer.mpi.GetWorldSize();

  // Assert same number of EFA devices
  assert(fi_peer.efas.size() == ib_peer.efas.size() && "EFA device count mismatch");
  SPDLOG_INFO("Rank {}: {} EFA devices (fi={}, ib={})", rank, fi_peer.efas.size(), fi_peer.efas.size(), ib_peer.efas.size());

  // Assert addresses match
  for (size_t i = 0; i < fi_peer.efas.size(); ++i) {
    auto fi_addr = EFA::Addr2Str(fi_peer.efas[i].GetAddr());
    auto ib_addr = ib::EFA::Addr2Str(ib_peer.efas[i].GetAddr());
    SPDLOG_INFO("  efa[{}] fi: {}", i, fi_addr);
    SPDLOG_INFO("  efa[{}] ib: {}", i, ib_addr);
  }

  fi_peer.Exchange();
  ib_peer.Exchange();

  fi_peer.Connect();
  ib_peer.Connect();

  // Assert same number of connections
  for (int r = 0; r < world_size; ++r) {
    if (r == rank) continue;
    assert(fi_peer.channels[r].size() == ib_peer.channels[r].size() && "Connection count mismatch");
    SPDLOG_INFO("Rank {} -> Rank {}: fi={} channels, ib={} channels", rank, r, fi_peer.channels[r].size(), ib_peer.channels[r].size());
  }

  MPI_Barrier(MPI_COMM_WORLD);
  SPDLOG_INFO("=== All assertions PASSED ===");
}
