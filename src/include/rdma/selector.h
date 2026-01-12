/**
 * @file selector.h
 * @brief Shared selector types and utilities for RDMA backends
 */
#pragma once
#include <io/event.h>
#include <io/selector.h>

#include <unordered_map>
#include <unordered_set>
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
    ret.emplace_back(Event{-1, entry.flags, e.ctx->handle});
    e.ctx = nullptr;
  } else {
    e.pending++;
  }
}

/**
 * @brief Generic RDMA selector template for completion queue event multiplexing
 * @tparam Backend Backend traits (FabricBackend or IBBackend)
 */
template <typename Backend>
class RdmaSelector : public detail::Selector {
 public:
  using CQ = typename Backend::CQ;
  using CQEntry = typename Backend::CQEntry;
  using Ctx = typename Backend::Context;
  using ImmCtx = typename Backend::ImmContext;
  using ImmMap = ImmContextMap<ImmCtx>;
  using ms = std::chrono::milliseconds;

  [[nodiscard]] std::vector<Event> Select(ms duration = ms{500}) override final {
    if (Stopped()) return {};
    std::vector<Event> res;
    CQEntry entries[kMaxCQEntries];
    for (auto* cq : cqs_) {
      auto rc = Backend::CQRead(cq, entries, kMaxCQEntries);
      if (rc > 0)
        HandleCompletion(entries, static_cast<size_t>(rc), res);
      else if (rc < 0)
        Backend::HandleError(cq, rc);
    }
    return res;
  }

  template <typename E>
  void Join(E& efa) noexcept {
    cqs_.emplace(efa.GetCQ());
  }

  template <typename E>
  void Quit(E& efa) noexcept {
    cqs_.erase(efa.GetCQ());
  }

  bool Join(ImmCtx& ctx) { return JoinImm(imm_, ctx); }
  void Quit(ImmCtx& ctx) { QuitImm(imm_, ctx); }

  [[nodiscard]] bool Stopped() const noexcept override final { return cqs_.empty(); }

 private:
  void HandleCompletion(CQEntry* entries, size_t n, std::vector<Event>& ret) {
    for (size_t i = 0; i < n; ++i) {
      auto& e = entries[i];
      if (Backend::IsRemoteWrite(e)) {
        HandleImmdata(e, ret, imm_);
      } else if (auto* ctx = reinterpret_cast<Ctx*>(e.op_context)) {
        ctx->entry = e;
        ret.emplace_back(Event{-1, e.flags, ctx->handle});
      }
    }
  }

  std::unordered_set<CQ*> cqs_;
  ImmMap imm_;
};

}  // namespace rdma
