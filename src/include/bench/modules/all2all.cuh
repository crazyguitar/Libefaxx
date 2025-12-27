/**
 * @file all2all.cuh
 * @brief All-to-all RDMA write communication patterns
 */
#pragma once

#include <io/runner.h>
#include <rdma/fabric/memory.h>

#include <vector>

/**
 * @brief All-to-all RDMA write (multi-channel, chunked)
 */
template <typename T>
Coro<>
RunAll2allWriteMultiChannel(std::vector<std::unique_ptr<T>>& write_bufs, std::vector<std::unique_ptr<T>>& read_bufs, int world_size, int rank) {
  std::vector<Future<Coro<ssize_t>>> wfuts;
  std::vector<Future<Coro<>>> rfuts;
  for (int peer = 0; peer < world_size; ++peer) {
    if (peer == rank) continue;
    rfuts.emplace_back(read_bufs[peer]->WaitallImmdata(peer + 1));
    wfuts.emplace_back(write_bufs[peer]->Writeall(peer, rank + 1));
  }
  for (auto& fut : wfuts) co_await fut;
  for (auto& fut : rfuts) co_await fut;
}

/**
 * @brief All-to-all RDMA write (round-robin channel per peer)
 *
 * Channel selection uses (peer + rank) % num_channels to distribute load.
 *
 * Why not use `peer % num_channels`?
 * With 4 ranks and 4 channels, `ch = peer % num_channels` causes contention:
 *   - Channel 0: Ranks 1,2,3 all send to peer 0
 *   - Channel 1: Ranks 0,2,3 all send to peer 1
 *   - Each channel has 3 senders competing!
 *
 * With `ch = (peer + rank) % num_channels`:
 *   Rank 0 → peer 1: ch=1, peer 2: ch=2, peer 3: ch=3
 *   Rank 1 → peer 0: ch=1, peer 2: ch=3, peer 3: ch=0
 *   Rank 2 → peer 0: ch=2, peer 1: ch=3, peer 3: ch=1
 *   Rank 3 → peer 0: ch=3, peer 1: ch=0, peer 2: ch=1
 * Channels are distributed across ranks, reducing contention.
 */
template <typename T>
Coro<> RunAll2allWriteRoundRobin(
    std::vector<std::unique_ptr<T>>& write_bufs,
    std::vector<std::unique_ptr<T>>& read_bufs,
    size_t num_channels,
    int world_size,
    int rank
) {
  std::vector<Future<Coro<ssize_t>>> wfuts;
  std::vector<Future<Coro<>>> rfuts;
  for (int peer = 0; peer < world_size; ++peer) {
    if (peer == rank) continue;
    size_t send_ch = (peer + rank) % num_channels;
    size_t recv_ch = (rank + peer) % num_channels;
    rfuts.emplace_back(read_bufs[peer]->WaitImmdata(T::EncodeImmdata(peer + 1, recv_ch)));
    wfuts.emplace_back(write_bufs[peer]->Writeall(peer, T::EncodeImmdata(rank + 1, send_ch), send_ch));
  }
  for (auto& fut : wfuts) co_await fut;
  for (auto& fut : rfuts) co_await fut;
}

/**
 * @brief All-to-all RDMA write (single channel)
 */
template <typename T>
Coro<>
RunAll2allWrite(std::vector<std::unique_ptr<T>>& write_bufs, std::vector<std::unique_ptr<T>>& read_bufs, int channel, int world_size, int rank) {
  std::vector<Future<Coro<ssize_t>>> wfuts;
  std::vector<Future<Coro<>>> rfuts;
  for (int peer = 0; peer < world_size; ++peer) {
    if (peer == rank) continue;
    rfuts.emplace_back(read_bufs[peer]->WaitImmdata(peer + 1));
    wfuts.emplace_back(write_bufs[peer]->Write(peer, rank + 1, channel));
  }
  for (auto& fut : wfuts) co_await fut;
  for (auto& fut : rfuts) co_await fut;
}
