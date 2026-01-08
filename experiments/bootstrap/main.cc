#include <bootstrap/mpi/fabric.h>
#include <bootstrap/mpi/ib.h>
#include <spdlog/spdlog.h>

void BootstrapFabric() {
  SPDLOG_INFO("=== libfabric bootstrap ===");
  fi::Peer peer;
  const auto rank = peer.mpi.GetWorldRank();
  const auto world_size = peer.mpi.GetWorldSize();
  SPDLOG_INFO("Rank {}: {} EFA devices", rank, peer.efas.size());
  for (size_t i = 0; i < peer.efas.size(); ++i) {
    SPDLOG_INFO("  efa[{}]: {}", i, EFA::Addr2Str(peer.efas[i].GetAddr()));
  }
  SPDLOG_INFO("Rank {}: Exchange", rank);
  peer.Exchange();
  SPDLOG_INFO("Rank {}: Connect", rank);
  peer.Connect();
  SPDLOG_INFO("Rank {}: Bootstrap complete", rank);
  for (int r = 0; r < world_size; ++r) {
    if (r == rank) continue;
    SPDLOG_INFO("Rank {} -> Rank {}: {} channels", rank, r, peer.channels[r].size());
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

void BootstrapIB() {
  SPDLOG_INFO("=== ibverbs bootstrap ===");
  ib::Peer peer;
  const auto rank = peer.mpi.GetWorldRank();
  const auto world_size = peer.mpi.GetWorldSize();
  SPDLOG_INFO("Rank {}: {} EFA devices", rank, peer.devices.size());
  for (size_t i = 0; i < peer.devices.size(); ++i) {
    SPDLOG_INFO("  device[{}]: {}", i, peer.devices[i].Name());
  }
  SPDLOG_INFO("Rank {}: Exchange", rank);
  peer.Exchange();
  SPDLOG_INFO("Rank {}: Connect", rank);
  peer.Connect();
  SPDLOG_INFO("Rank {}: Bootstrap complete", rank);
  for (int r = 0; r < world_size; ++r) {
    if (r == rank) continue;
    SPDLOG_INFO("Rank {} -> Rank {}: {} AHs", rank, r, peer.ahs[r].size());
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
  BootstrapFabric();
  BootstrapIB();
}
