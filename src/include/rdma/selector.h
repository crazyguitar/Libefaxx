/**
 * @file selector.h
 * @brief Shared selector types and utilities for RDMA backends
 */
#pragma once
#include <io/event.h>

#include <unordered_map>
#include <vector>

namespace rdma {

/** @brief Maximum completion queue entries to read per poll */
static constexpr size_t kMaxCQEntries = 128;

/**
 * @brief Entry for tracking immediate data completions and waiters
 * @tparam ImmCtx Immediate context type (fi::ImmContext or ib::ImmContext)
 */
template <typename ImmCtx>
struct ImmEntry {
  ImmCtx* ctx{nullptr};  ///< Waiting context or nullptr
  int pending{0};        ///< Count of unmatched completions

  ImmEntry() noexcept = default;
  ~ImmEntry() noexcept = default;
  ImmEntry(const ImmEntry&) noexcept = default;
  ImmEntry(ImmEntry&&) noexcept = default;
  ImmEntry& operator=(const ImmEntry&) noexcept = default;
  ImmEntry& operator=(ImmEntry&&) noexcept = default;
};

/** @brief Map from immediate data value to entry tracking waiters */
template <typename ImmCtx>
using ImmContextMap = std::unordered_map<uint64_t, ImmEntry<ImmCtx>>;

/**
 * @brief Register an immediate data context for monitoring
 * @param imm Map of immediate data contexts
 * @param ctx Context to register
 * @return true if completion already pending (immediate wakeup), false if registered for later
 */
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

/**
 * @brief Unregister an immediate data context
 * @param imm Map of immediate data contexts
 * @param ctx Context to unregister
 */
template <typename ImmCtx>
void QuitImm(ImmContextMap<ImmCtx>& imm, ImmCtx& ctx) {
  auto it = imm.find(ctx.imm_data);
  if (it == imm.end()) return;
  it->second.ctx = nullptr;
  if (it->second.pending == 0) imm.erase(it);
}

/**
 * @brief Handle immediate data from remote RDMA write
 * @param entry Completion queue entry with immediate data
 * @param ret Output vector of events to wake
 * @param imm Map of immediate data contexts
 */
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
