/**
 * @file sendrecv.cuh
 * @brief SendRecv benchmark functors for point-to-point communication
 *
 * This module benchmarks bidirectional send/recv operations using RDMA WRITE
 * with immediate data (like UCCL pattern). Unlike pure RDMA write, this
 * requires pre-exchanged remote memory info.
 *
 * ## Send/Recv Flow (Ping-Pong)
 * ```
 * Rank 0                              Rank T (target)
 * ──────                              ───────────────
 *    │                                     │
 *    │  Sendall(target, ch)                │
 *    │  [RDMA WRITE + imm to target]       │
 *    ├────────────────────────────────────►│
 *    │                                     │ Recvall(0, ch)
 *    │                                     │ [Wait for imm]
 *    │                                     │
 *    │                                     │ Sendall(0, ch)
 *    │◄────────────────────────────────────┤ [RDMA WRITE + imm back]
 *    │  Recvall(target, ch)                │
 *    │  [Wait for imm]                     │
 *    │                                     │
 * ```
 *
 * ## Implementation Details
 * - Uses RDMA WRITE with immediate data for send semantics
 * - Receiver waits for immediate data completion (no buffer posting)
 * - Requires symmetric memory with pre-exchanged RMA IOVs
 * - Measures round-trip latency and bandwidth
 *
 * ## Bandwidth
 * - Single channel: ~97 Gbps per direction
 * - Bidirectional: ~194 Gbps total (full duplex)
 */
#pragma once

#include <io/coro.h>
#include <io/runner.h>

/**
 * @brief Rank 0 send/recv functor
 *
 * Sends data to target, then receives response.
 * Only rank 0 executes; other ranks return immediately.
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
