/**
 * @file write.cuh
 * @brief Write benchmark functors for point-to-point RDMA write
 */
#pragma once

#include <io/coro.h>
#include <io/runner.h>
#include <rdma/proxy.h>

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
    co_await write[target]->Write(1, channel);
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
    co_await write[target]->Writeall(1);
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
    for (auto& efa : peer.efas) IO::Get().Join<FabricProxy>(efa);
    Run([&]() -> Coro<> {
      co_await Write<Peer>{target, channel}.template operator()<T>(peer, write);
      co_await Read<Peer>{target}.template operator()<T>(peer, read);
      for (auto& efa : peer.efas) IO::Get().Quit<FabricProxy>(efa);
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
    for (auto& efa : peer.efas) IO::Get().Join<FabricProxy>(efa);
    Run([&]() -> Coro<> {
      co_await WriteMulti<Peer>{target}.template operator()<T>(peer, write);
      co_await ReadMulti<Peer>{target}.template operator()<T>(peer, read);
      for (auto& efa : peer.efas) IO::Get().Quit<FabricProxy>(efa);
    }());
  }
};

/**
 * @brief No-op verification functor
 */
struct NoVerify {
  template <typename P, typename Buffers>
  void operator()(P&, Buffers&) const {}
};
