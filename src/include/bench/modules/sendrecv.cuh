/**
 * @file sendrecv.cuh
 * @brief SendRecv benchmark functors for point-to-point communication
 */
#pragma once

#include <io/coro.h>
#include <io/runner.h>
#include <rdma/proxy.h>

/**
 * @brief Rank 0 send/recv functor
 */
template <typename Peer>
struct SendRecv {
  int target;   ///< Target rank to communicate with
  int channel;  ///< EFA channel to use

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& send, typename Peer::template Buffers<T>& recv) {
    if (peer.mpi.GetWorldRank() != 0) return;
    for (auto& efa : peer.efas) IO::Get().Join<FabricProxy>(efa);
    Run([&]() -> Coro<> {
      co_await send[target]->Sendall(channel);
      co_await recv[target]->Recvall(channel);
      for (auto& efa : peer.efas) IO::Get().Quit<FabricProxy>(efa);
    }());
  }
};

/**
 * @brief Target rank recv/send functor
 */
template <typename Peer>
struct RecvSend {
  int target;   ///< This rank's ID (only this rank executes)
  int channel;  ///< EFA channel to use

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& send, typename Peer::template Buffers<T>& recv) {
    if (peer.mpi.GetWorldRank() != target) return;
    for (auto& efa : peer.efas) IO::Get().Join<FabricProxy>(efa);
    Run([&]() -> Coro<> {
      co_await recv[0]->Recvall(channel);
      co_await send[0]->Sendall(channel);
      for (auto& efa : peer.efas) IO::Get().Quit<FabricProxy>(efa);
    }());
  }
};

/**
 * @brief Combined pair benchmark functor
 */
template <typename Peer>
struct PairBench {
  int target;   ///< Target rank to communicate with
  int channel;  ///< EFA channel to use

  template <typename T>
  void operator()(Peer& peer, typename Peer::template Buffers<T>& send, typename Peer::template Buffers<T>& recv) {
    SendRecv<Peer>{target, channel}.template operator()<T>(peer, send, recv);
    RecvSend<Peer>{target, channel}.template operator()<T>(peer, send, recv);
  }
};
