/**
 * @file selector.h
 * @brief Shared selector types and utilities for RDMA backends
 */
#pragma once
#include <io/event.h>

#include <unordered_map>
#include <vector>

namespace rdma {

static constexpr size_t kMaxCQEntries = 128;

/** @brief Entry for tracking immediate data completions and waiters */
template <typename ImmCtx>
struct ImmEntry {
  ImmCtx* ctx{nullptr};
  int pending{0};
};

template <typename ImmCtx>
using ImmContextMap = std::unordered_map<uint64_t, ImmEntry<ImmCtx>>;

/** @brief Register an immediate data context for monitoring */
template <typename ImmCtx>
bool JoinImm(ImmContextMap<ImmCtx>& imm, ImmCtx& ctx) {
  auto& entry = imm[ctx.imm_data];
  if (entry.pending > 0) {
    entry.pending--;
    if (entry.pending == 0 && entry.ctx == nullptr) imm.erase(ctx.imm_data);
    return true;
  }
  entry.ctx = &ctx;
  return false;
}

/** @brief Unregister an immediate data context */
template <typename ImmCtx>
void QuitImm(ImmContextMap<ImmCtx>& imm, ImmCtx& ctx) {
  auto it = imm.find(ctx.imm_data);
  if (it == imm.end()) return;
  it->second.ctx = nullptr;
  if (it->second.pending == 0) imm.erase(it);
}

/** @brief Handle immediate data from remote RDMA write */
template <typename CQEntry, typename ImmCtx>
void HandleImmdata(CQEntry& entry, std::vector<Event>& ret, ImmContextMap<ImmCtx>& imm) {
  uint64_t imm_data = entry.data;
  if (!imm_data) return;
  auto& e = imm[imm_data];
  if (e.ctx && e.ctx->handle) {
    e.ctx->entry = entry;
    ret.emplace_back(Event{static_cast<int>(entry.flags), e.ctx->handle});
    e.ctx = nullptr;
  } else {
    e.pending++;
  }
}

}  // namespace rdma
