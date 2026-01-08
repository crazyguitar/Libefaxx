/**
 * @file selector.h
 * @brief ibverbs completion queue selector for async I/O (equivalent to FabricSelector)
 */
#pragma once
#include <arpa/inet.h>
#include <io/event.h>
#include <io/selector.h>
#include <rdma/ib/context.h>
#include <rdma/ib/ib.h>
#include <spdlog/spdlog.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ib {

/** @brief Entry for tracking immediate data completions and waiters */
struct ImmEntry {
  ImmContext* ctx{nullptr};
  int pending{0};
};

using ImmContextMap = std::unordered_map<uint64_t, ImmEntry>;

static constexpr size_t kMaxCQEntries = 128;

/**
 * @brief Read completion queue entries (equivalent to fi_cq_read)
 *
 * Reference: libfabric/prov/efa/src/efa_cq.c:44-56 (efa_cq_read_entry_common)
 * Reference: libfabric/prov/efa/src/efa_cq.c:63-75 (efa_cq_read_data_entry)
 *
 * @param cq Completion queue handle
 * @param entries Output array for completion entries
 * @param max_entries Maximum entries to read
 * @return Number of entries read, or negative error code
 */
inline ssize_t ib_cq_read(ib_cq* cq, ib_cq_data_entry* entries, size_t max_entries) {
  ibv_poll_cq_attr attr{};
  int ret = ibv_start_poll(cq->cq, &attr);
  if (ret == ENOENT) return 0;
  if (ret != 0) return -ret;

  auto* cq_ex = cq->cq;
  size_t count = 0;
  do {
    if (cq_ex->status) {
      SPDLOG_ERROR("CQ error: status={} vendor_err={}", static_cast<int>(cq_ex->status), ibv_wc_read_vendor_err(cq_ex));
      ibv_end_poll(cq->cq);
      return -EIO;
    }

    auto opcode = ibv_wc_read_opcode(cq_ex);
    auto wr_id = cq_ex->wr_id;
    auto flags = ibv_wc_read_wc_flags(cq_ex);
    bool is_recv_rdma_imm = (opcode == IBV_WC_RECV_RDMA_WITH_IMM);

    if (wr_id || is_recv_rdma_imm) {
      entries[count].op_context = reinterpret_cast<void*>(wr_id);
      entries[count].flags = flags;
      entries[count].len = ibv_wc_read_byte_len(cq_ex);
      entries[count].buf = nullptr;
      entries[count].data = 0;
      if (flags & IBV_WC_WITH_IMM) {
        entries[count].data = ntohl(ibv_wc_read_imm_data(cq_ex));
      }
      ++count;
      if (count >= max_entries) break;
    }
  } while (ibv_next_poll(cq_ex) == 0);

  ibv_end_poll(cq->cq);
  return static_cast<ssize_t>(count);
}

/**
 * @brief ibverbs-based selector for completion queue event multiplexing
 */
class IBSelector : public detail::Selector {
 public:
  using ms = std::chrono::milliseconds;

  [[nodiscard]] std::vector<Event> Select(ms duration = ms{500}) override final {
    if (Stopped()) return {};
    std::vector<Event> res;
    ib_cq_data_entry cq_entries[kMaxCQEntries];

    for (auto* cq : cqs_) {
      auto rc = ib_cq_read(cq, cq_entries, kMaxCQEntries);
      if (rc > 0) HandleCompletion(cq_entries, static_cast<size_t>(rc), res, imm_);
      if (rc < 0) SPDLOG_ERROR("ib_cq_read error: {}", rc);
    }
    return res;
  }

  /** @brief Register an EFA endpoint's completion queue */
  template <typename E>
  void Join(E& efa) noexcept {
    cqs_.emplace(efa.GetCQ());
  }

  /** @brief Unregister an EFA endpoint's completion queue */
  template <typename E>
  void Quit(E& efa) noexcept {
    cqs_.erase(efa.GetCQ());
  }

  /** @brief Register an immediate data context for monitoring */
  bool Join(ImmContext& ctx) {
    auto& entry = imm_[ctx.imm_data];
    if (entry.pending > 0) {
      entry.pending--;
      if (entry.pending == 0 && entry.ctx == nullptr) imm_.erase(ctx.imm_data);
      return true;
    }
    entry.ctx = &ctx;
    return false;
  }

  /** @brief Unregister an immediate data context */
  void Quit(ImmContext& ctx) {
    auto it = imm_.find(ctx.imm_data);
    if (it == imm_.end()) return;
    it->second.ctx = nullptr;
    if (it->second.pending == 0) imm_.erase(it);
  }

  [[nodiscard]] bool Stopped() const noexcept override final { return cqs_.empty(); }

 private:
  static void HandleImmdata(ib_cq_data_entry& entry, std::vector<Event>& ret, ImmContextMap& imm) {
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

  static void HandleCompletion(ib_cq_data_entry* cq_entries, size_t n, std::vector<Event>& ret, ImmContextMap& imm) {
    for (size_t i = 0; i < n; ++i) {
      auto& entry = cq_entries[i];
      // Check for remote write with immediate data
      if (entry.data != 0) {
        HandleImmdata(entry, ret, imm);
      } else {
        Context* context = reinterpret_cast<Context*>(entry.op_context);
        if (!context) continue;
        context->entry = entry;
        Handle* handle = context->handle;
        ret.emplace_back(Event{-1, entry.flags, handle});
      }
    }
  }

 private:
  std::unordered_set<ib_cq*> cqs_;
  ImmContextMap imm_;
};

}  // namespace ib
