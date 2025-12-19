/**
 * @file sendrecv.cuh
 * @brief SendRecv benchmark functors for point-to-point communication
 *
 * Implements rank0 <-> rank_k communication pattern where rank 0 sends
 * then receives, and target rank receives then sends.
 */
#pragma once

#include <io/runner.h>
#include <rdma/proxy.h>

#include <bench/mpi/fabric.cuh>

/**
 * @brief Rank 0 send/recv functor
 *
 * Rank 0 sends data to target, then receives response.
 * Other ranks skip this operation.
 */
struct SendRecv {
  int target;  /**< Target rank to communicate with */
  int channel; /**< EFA channel to use */

  /**
   * @brief Execute send then recv on rank 0
   * @tparam T Buffer type
   * @param peer FabricBench peer instance
   * @param send Send buffer array
   * @param recv Receive buffer array
   */
  template <typename T>
  void operator()(FabricBench& peer, FabricBench::Buffers<T>& send, FabricBench::Buffers<T>& recv) {
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
 *
 * Target rank receives data from rank 0, then sends response.
 * Non-target ranks skip this operation.
 */
struct RecvSend {
  int target;  /**< This rank's ID (only this rank executes) */
  int channel; /**< EFA channel to use */

  /**
   * @brief Execute recv then send on target rank
   * @tparam T Buffer type
   * @param peer FabricBench peer instance
   * @param send Send buffer array
   * @param recv Receive buffer array
   */
  template <typename T>
  void operator()(FabricBench& peer, FabricBench::Buffers<T>& send, FabricBench::Buffers<T>& recv) {
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
 *
 * Executes both SendRecv (rank 0) and RecvSend (target rank)
 * to complete one round-trip communication.
 */
struct PairBench {
  int target;  /**< Target rank to communicate with */
  int channel; /**< EFA channel to use */

  /**
   * @brief Execute paired send/recv operation
   * @tparam T Buffer type
   * @param peer FabricBench peer instance
   * @param send Send buffer array
   * @param recv Receive buffer array
   */
  template <typename T>
  void operator()(FabricBench& peer, FabricBench::Buffers<T>& send, FabricBench::Buffers<T>& recv) {
    SendRecv{target, channel}(peer, send, recv);
    RecvSend{target, channel}(peer, send, recv);
  }
};

/**
 * @brief No-op verification functor
 *
 * Skips verification during benchmark iterations.
 * Used when random initialization makes verification impractical.
 */
struct NoVerify {
  template <typename P, typename Buffers>
  void operator()(P&, Buffers&) const {}
};
