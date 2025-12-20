/**
 * @file all2all.cuh
 * @brief All-to-all RDMA write communication patterns
 */
#pragma once

#include <io/runner.h>

#include <vector>

/**
 * @brief All-to-all RDMA write (multi-channel)
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
