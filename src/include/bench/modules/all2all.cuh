/**
 * @file all2all.cuh
 * @brief All-to-all RDMA write communication patterns
 *
 * This module implements all-to-all communication where every rank sends
 * data to every other rank simultaneously using RDMA write.
 *
 * ## All-to-All Pattern (4 ranks)
 * ```
 *        Rank 0    Rank 1    Rank 2    Rank 3
 *          │         │         │         │
 *          ├────────►│         │         │  (0→1)
 *          ├─────────┼────────►│         │  (0→2)
 *          ├─────────┼─────────┼────────►│  (0→3)
 *          │◄────────┤         │         │  (1→0)
 *          │         ├────────►│         │  (1→2)
 *          │         ├─────────┼────────►│  (1→3)
 *          │◄────────┼─────────┤         │  (2→0)
 *          │         │◄────────┤         │  (2→1)
 *          │         │         ├────────►│  (2→3)
 *          │◄────────┼─────────┼─────────┤  (3→0)
 *          │         │◄────────┼─────────┤  (3→1)
 *          │         │         │◄────────┤  (3→2)
 *          │         │         │         │
 *       [All writes happen in parallel]
 * ```
 *
 * ## Channel Distribution (Round-Robin)
 * ```
 * ch = (peer + rank) % num_channels
 *
 * With 4 ranks, 4 channels:
 * ┌────────┬─────────────────────────────────────┐
 * │ Sender │  Target → Channel                   │
 * ├────────┼─────────────────────────────────────┤
 * │ Rank 0 │  1→ch1, 2→ch2, 3→ch3                │
 * │ Rank 1 │  0→ch1, 2→ch3, 3→ch0                │
 * │ Rank 2 │  0→ch2, 1→ch3, 3→ch1                │
 * │ Rank 3 │  0→ch3, 1→ch0, 2→ch1                │
 * └────────┴─────────────────────────────────────┘
 *
 * This distributes load evenly across channels.
 * Simple `peer % num_channels` causes contention!
 * ```
 *
 * ## Modes
 * - Single:     All writes on one channel (~97 Gbps)
 * - Multi:      Each write uses all channels (striped)
 * - RoundRobin: Each peer uses different channel (~400 Gbps aggregate)
 */
#pragma once

#include <io/runner.h>

#include <vector>

/**
 * @brief All-to-all RDMA write (multi-channel, chunked)
 *
 * Each rank writes to all other ranks using all channels (data striped).
 * Waits for all incoming writes via WaitallImmdata.
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
