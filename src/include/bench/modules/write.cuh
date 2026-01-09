/**
 * @file write.cuh
 * @brief Write benchmark functors for point-to-point RDMA write
 *
 * This module provides functors for benchmarking RDMA write operations.
 * Three modes are supported: single-channel, multi-channel, and round-robin.
 *
 * ## Single Channel (PairWrite)
 * ```
 * Rank 0                              Rank T (target)
 * ──────                              ───────────────
 *    │                                     │
 *    │  Write(target, imm=1, ch=0)         │
 *    ├────────────────────────────────────►│
 *    │         [EFA channel 0]             │
 *    │                                     │
 *    │                              WaitImmdata(1)
 *    │                                     │
 * ```
 *
 * ## Multi Channel (PairWriteMulti)
 * ```
 * Rank 0                              Rank T (target)
 * ──────                              ───────────────
 *    │                                     │
 *    │  Writeall(target, imm=1)            │
 *    ├──┬──┬──┬───────────────────────────►│
 *    │  │  │  │  [All EFA channels]        │
 *    │  │  │  │  (data striped)            │
 *    │                                     │
 *    │                           WaitallImmdata(1)
 *    │                                     │
 * ```
 *
 * ## Round-Robin (PairWriteRoundRobinAll)
 * ```
 * Rank 0                    Rank 1      Rank 2      Rank 3      Rank 4
 * ──────                    ──────      ──────      ──────      ──────
 *    │                         │           │           │           │
 *    ├─── ch=1%4=1 ───────────►│           │           │           │
 *    ├─── ch=2%4=2 ────────────┼──────────►│           │           │
 *    ├─── ch=3%4=3 ────────────┼───────────┼──────────►│           │
 *    ├─── ch=4%4=0 ────────────┼───────────┼───────────┼──────────►│
 *    │   [Parallel writes]     │           │           │           │
 *    │                         │           │           │           │
 *    │                   WaitImmdata  WaitImmdata  WaitImmdata  WaitImmdata
 *    │                   (enc(1,1))   (enc(1,2))   (enc(1,3))   (enc(1,0))
 * ```
 *
 * ## Bandwidth Characteristics
 * - Single:     ~97 Gbps  (1 EFA)
 * - Multi:      ~97 Gbps  (data striped, same total)
 * - RoundRobin: ~400 Gbps (4 EFAs parallel, different targets)
 */
#pragma once

#include <io/coro.h>
#include <io/runner.h>

/**
 * @brief Rank 0 write functor (single channel)
 */
template <typename Peer>
struct Write {
  int target;
  int channel;

  template <typename T>
  Coro<> operator()(Peer& peer, typename Peer::template Buffers<T>& write) {
    if (peer.mpi.GetWorldRank() != 0) co_return;
    co_await write[target]->Write(target, 1, channel);
  }
};

/**
 * @brief Rank 0 write functor (multi-channel)
 */
template <typename Peer>
struct WriteMulti {
  int target;

  template <typename T>
  Coro<> operator()(Peer& peer, typename Peer::template Buffers<T>& write) {
    if (peer.mpi.GetWorldRank() != 0) co_return;
    co_await write[target]->Writeall(target, 1);
  }
};

/**
 * @brief Target rank wait for immediate data
 */
template <typename Peer>
struct Read {
  int target;

  template <typename T>
  Coro<> operator()(Peer& peer, typename Peer::template Buffers<T>& read) {
    if (peer.mpi.GetWorldRank() != target) co_return;
    co_await read[0]->WaitImmdata(1);
  }
};

/**
 * @brief Target rank wait for immediate data (multi-channel)
 */
template <typename Peer>
struct ReadMulti {
  int target;

  template <typename T>
  Coro<> operator()(Peer& peer, typename Peer::template Buffers<T>& read) {
    if (peer.mpi.GetWorldRank() != target) co_return;
    co_await read[0]->WaitallImmdata(1);
  }
};

/**
 * @brief Combined pair benchmark functor (single channel)
 */
template <typename Peer, typename Selector>
struct PairWrite {
  int target;
  int channel;

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& write, typename Peer::template Buffers<T>& read) {
    for (auto& efa : peer.efas) IO::Get().Join<Selector>(efa);
    Run([&]() -> Coro<> {
      co_await Write<Peer>{target, channel}.template operator()<T>(peer, write);
      co_await Read<Peer>{target}.template operator()<T>(peer, read);
      for (auto& efa : peer.efas) IO::Get().Quit<Selector>(efa);
    }());
  }
};

/**
 * @brief Combined pair benchmark functor (multi-channel)
 */
template <typename Peer, typename Selector>
struct PairWriteMulti {
  int target;

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& write, typename Peer::template Buffers<T>& read) {
    for (auto& efa : peer.efas) IO::Get().Join<Selector>(efa);
    Run([&]() -> Coro<> {
      co_await WriteMulti<Peer>{target}.template operator()<T>(peer, write);
      co_await ReadMulti<Peer>{target}.template operator()<T>(peer, read);
      for (auto& efa : peer.efas) IO::Get().Quit<Selector>(efa);
    }());
  }
};

/**
 * @brief Rank 0 write using round-robin channel selection
 */
template <typename Peer>
struct WriteRoundRobin {
  int target;

  template <typename T>
  Coro<> operator()(Peer& peer, typename Peer::template Buffers<T>& write) {
    if (peer.mpi.GetWorldRank() != 0) co_return;
    size_t ch = target % peer.efas.size();
    co_await write[target]->Write(target, 1, ch);
  }
};

/**
 * @brief Target rank wait using round-robin channel
 */
template <typename Peer>
struct ReadRoundRobin {
  int target;

  template <typename T>
  Coro<> operator()(Peer& peer, typename Peer::template Buffers<T>& read) {
    if (peer.mpi.GetWorldRank() != target) co_return;
    co_await read[0]->WaitImmdata(1);
  }
};

/**
 * @brief Combined pair benchmark functor (round-robin channel selection)
 *
 * Each target uses channel = target % num_channels, distributing load across EFAs.
 */
template <typename Peer, typename Selector>
struct PairWriteRoundRobin {
  int target;

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& write, typename Peer::template Buffers<T>& read) {
    for (auto& efa : peer.efas) IO::Get().Join<Selector>(efa);
    Run([&]() -> Coro<> {
      co_await WriteRoundRobin<Peer>{target}.template operator()<T>(peer, write);
      co_await ReadRoundRobin<Peer>{target}.template operator()<T>(peer, read);
      for (auto& efa : peer.efas) IO::Get().Quit<Selector>(efa);
    }());
  }
};

/**
 * @brief Parallel write to all targets using round-robin channels
 */
template <typename Peer, typename Selector>
struct PairWriteRoundRobinAll {
  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& write, typename Peer::template Buffers<T>& read) {
    int rank = peer.mpi.GetWorldRank();
    int world = peer.mpi.GetWorldSize();
    size_t nch = peer.efas.size();
    for (auto& efa : peer.efas) IO::Get().Join<Selector>(efa);
    Run([&]() -> Coro<> {
      if (rank == 0) {
        std::vector<Future<Coro<ssize_t>>> futs;
        for (int t = 1; t < world; ++t) {
          size_t ch = t % nch;
          futs.emplace_back(write[t]->Writeall(t, T::EncodeImmdata(1, ch), ch));
        }
        for (auto& f : futs) co_await f;
      } else {
        size_t ch = rank % nch;
        co_await read[0]->WaitImmdata(T::EncodeImmdata(1, ch));
      }
      for (auto& efa : peer.efas) IO::Get().Quit<Selector>(efa);
    }());
  }
};

/**
 * @brief Rank 0 writes to all targets in parallel using round-robin channels
 *
 * Each target uses a different channel: ch = target % num_channels
 * This enables parallel writes across channels when multiple targets exist.
 */
template <typename T>
Coro<> RunWriteRoundRobin(
    std::vector<std::unique_ptr<T>>& write_bufs,
    std::vector<std::unique_ptr<T>>& read_bufs,
    size_t num_channels,
    int world_size,
    int rank
) {
  if (rank != 0) {
    // Non-rank-0: wait for incoming write from rank 0
    size_t ch = rank % num_channels;
    co_await read_bufs[0]->WaitImmdata(T::EncodeImmdata(1, ch));
    co_return;
  }
  // Rank 0: write to all targets in parallel using different channels
  std::vector<Future<Coro<ssize_t>>> wfuts;
  for (int target = 1; target < world_size; ++target) {
    size_t ch = target % num_channels;
    wfuts.emplace_back(write_bufs[target]->Writeall(target, T::EncodeImmdata(1, ch), ch));
  }
  for (auto& fut : wfuts) co_await fut;
}
