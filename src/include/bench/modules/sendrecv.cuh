/**
 * @file sendrecv.cuh
 * @brief SendRecv benchmark functors for point-to-point communication
 */
#pragma once

#include <io/coro.h>
#include <io/runner.h>

/**
 * @brief Rank 0 send/recv functor
 */
template <typename Peer, typename Selector>
struct SendRecv {
  int target;
  int channel;

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& send, typename Peer::template Buffers<T>& recv) {
    if (peer.mpi.GetWorldRank() != 0) return;
    for (auto& efa : peer.efas) IO::Get().Join<Selector>(efa);
    Run([&]() -> Coro<> {
      co_await send[target]->Sendall(target, channel);
      co_await recv[target]->Recvall(target, channel);
      for (auto& efa : peer.efas) IO::Get().Quit<Selector>(efa);
    }());
  }
};

/**
 * @brief Target rank recv/send functor
 */
template <typename Peer, typename Selector>
struct RecvSend {
  int target;
  int channel;

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& send, typename Peer::template Buffers<T>& recv) {
    if (peer.mpi.GetWorldRank() != target) return;
    for (auto& efa : peer.efas) IO::Get().Join<Selector>(efa);
    Run([&]() -> Coro<> {
      co_await recv[0]->Recvall(0, channel);
      co_await send[0]->Sendall(0, channel);
      for (auto& efa : peer.efas) IO::Get().Quit<Selector>(efa);
    }());
  }
};

/**
 * @brief Combined pair benchmark functor
 */
template <typename Peer, typename Selector>
struct PairBench {
  int target;
  int channel = 0;

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& send, typename Peer::template Buffers<T>& recv) {
    SendRecv<Peer, Selector>{target, channel}.template operator()<T>(peer, send, recv);
    RecvSend<Peer, Selector>{target, channel}.template operator()<T>(peer, send, recv);
  }
};
