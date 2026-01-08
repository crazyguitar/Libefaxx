/**
 * @file write.cuh
 * @brief Write benchmark functors for point-to-point RDMA write
 */
#pragma once

#include <io/coro.h>
#include <io/runner.h>
#include <rdma/fabric/selector.h>

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
template <typename Peer>
struct PairWrite {
  int target;
  int channel;

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& write, typename Peer::template Buffers<T>& read) {
    for (auto& efa : peer.efas) IO::Get().Join<fi::FabricSelector>(efa);
    Run([&]() -> Coro<> {
      co_await Write<Peer>{target, channel}.template operator()<T>(peer, write);
      co_await Read<Peer>{target}.template operator()<T>(peer, read);
      for (auto& efa : peer.efas) IO::Get().Quit<fi::FabricSelector>(efa);
    }());
  }
};

/**
 * @brief Combined pair benchmark functor (multi-channel)
 */
template <typename Peer>
struct PairWriteMulti {
  int target;

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& write, typename Peer::template Buffers<T>& read) {
    for (auto& efa : peer.efas) IO::Get().Join<fi::FabricSelector>(efa);
    Run([&]() -> Coro<> {
      co_await WriteMulti<Peer>{target}.template operator()<T>(peer, write);
      co_await ReadMulti<Peer>{target}.template operator()<T>(peer, read);
      for (auto& efa : peer.efas) IO::Get().Quit<fi::FabricSelector>(efa);
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
