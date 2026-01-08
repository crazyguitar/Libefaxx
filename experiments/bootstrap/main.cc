#include <bootstrap/mpi/fabric.h>
#include <bootstrap/mpi/ib.h>
#include <spdlog/spdlog.h>

template <typename P>
void Bootstrap(const char* name) {
  SPDLOG_INFO("=== {} bootstrap ===", name);
  P peer;
  const auto rank = peer.mpi.GetWorldRank();
  SPDLOG_INFO("Rank {}: Exchange", rank);
  peer.Exchange();
  SPDLOG_INFO("Rank {}: Connect", rank);
  peer.Connect();
  SPDLOG_INFO("Rank {}: Bootstrap complete", rank);
  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
  (Bootstrap<fi::Peer>("libfabric"), Bootstrap<ib::Peer>("ibverbs"));
}
