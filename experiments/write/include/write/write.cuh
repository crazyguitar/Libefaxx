/**
 * @file write.cuh
 * @brief Write benchmark functors for point-to-point RDMA write
 *
 * Implements rank0 -> rank_k write pattern where rank 0 writes
 * to each target rank sequentially, target waits for immediate data.
 */
#pragma once

#include <io/runner.h>
#include <rdma/fabric/memory.h>
#include <rdma/proxy.h>

#include <bench/mpi/fabric.cuh>

/**
 * @brief Rank 0 write functor with immediate data (single channel)
 */
struct Write {
  int target;
  int channel;

  template <typename T>
  Coro<> operator()(FabricBench& peer, FabricBench::Buffers<T>& write) {
    if (peer.mpi.GetWorldRank() != 0) co_return;
    uint64_t imm_data = 1;
    co_await write[target]->Write(imm_data, channel);
  }
};

/**
 * @brief Rank 0 write functor with immediate data (multi-channel)
 */
struct WriteMulti {
  int target;

  template <typename T>
  Coro<> operator()(FabricBench& peer, FabricBench::Buffers<T>& write) {
    if (peer.mpi.GetWorldRank() != 0) co_return;
    uint64_t imm_data = 1;
    co_await write[target]->Writeall(imm_data);
  }
};

/**
 * @brief Target rank wait for immediate data
 */
struct Read {
  int target;

  template <typename T>
  Coro<> operator()(FabricBench& peer, FabricBench::Buffers<T>& read) {
    if (peer.mpi.GetWorldRank() != target) co_return;
    co_await read[0]->WaitImmdata(1);
  }
};

/**
 * @brief Target rank wait for immediate data from all channels
 */
struct ReadMulti {
  int target;

  template <typename T>
  Coro<> operator()(FabricBench& peer, FabricBench::Buffers<T>& read) {
    if (peer.mpi.GetWorldRank() != target) co_return;
    co_await read[0]->WaitallImmdata(1);
  }
};

/**
 * @brief Combined pair benchmark functor (single channel)
 */
struct PairWrite {
  int target;
  int channel;

  template <typename T>
  void operator()(FabricBench& peer, FabricBench::Buffers<T>& write, FabricBench::Buffers<T>& read) {
    for (auto& efa : peer.efas) IO::Get().Join<FabricProxy>(efa);
    Run([&]() -> Coro<> {
      co_await Write{target, channel}(peer, write);
      co_await Read{target}(peer, read);
      for (auto& efa : peer.efas) IO::Get().Quit<FabricProxy>(efa);
    }());
  }
};

/**
 * @brief Combined pair benchmark functor (multi-channel)
 */
struct PairWriteMulti {
  int target;

  template <typename T>
  void operator()(FabricBench& peer, FabricBench::Buffers<T>& write, FabricBench::Buffers<T>& read) {
    for (auto& efa : peer.efas) IO::Get().Join<FabricProxy>(efa);
    Run([&]() -> Coro<> {
      co_await WriteMulti{target}(peer, write);
      co_await ReadMulti{target}(peer, read);
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
