#pragma once

#include <io/runner.h>

#include <vector>

/**
 * @brief All-to-all RDMA write communication pattern using multi-channel Writeall
 * Each rank writes to all other ranks using the new Writeall without channel parameter
 * Note: Assumes all peers have the same number of channels
 */
template <typename T>
Coro<>
RunAll2allWriteMultiChannel(std::vector<std::unique_ptr<T>>& write_bufs, std::vector<std::unique_ptr<T>>& read_bufs, int world_size, int rank) {
  std::vector<Future<Coro<ssize_t>>> wfuts;
  std::vector<Future<Coro<>>> rfuts;

  for (int peer = 0; peer < world_size; ++peer) {
    if (peer == rank) continue;
    uint64_t imm_data = rank + 1;                                   // We send rank+1
    rfuts.emplace_back(read_bufs[peer]->WaitallImmdata(peer + 1));  // We expect peer+1 from peer
    wfuts.emplace_back(write_bufs[peer]->Writeall(imm_data));       // We send rank+1 to peer
  }

  for (auto& fut : wfuts) co_await fut;
  for (auto& fut : rfuts) co_await fut;
}

/**
 * @brief All-to-all RDMA write communication pattern
 * Each rank writes to all other ranks and waits for immediate data
 */
template <typename T>
Coro<>
RunAll2allWrite(std::vector<std::unique_ptr<T>>& write_bufs, std::vector<std::unique_ptr<T>>& read_bufs, int channel, int world_size, int rank) {
  std::vector<Future<Coro<ssize_t>>> wfuts;
  std::vector<Future<Coro<>>> rfuts;

  for (int peer = 0; peer < world_size; ++peer) {
    if (peer == rank) continue;
    uint64_t imm_data = rank + 1;
    rfuts.emplace_back(read_bufs[peer]->WaitImmdata(peer + 1));
    wfuts.emplace_back(write_bufs[peer]->Write(imm_data, channel));
  }

  for (auto& fut : wfuts) co_await fut;
  for (auto& fut : rfuts) co_await fut;
}
