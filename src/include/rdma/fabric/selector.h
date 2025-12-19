/**
 * @file selector.h
 * @brief Libfabric completion queue selector for async I/O
 */
#pragma once
#include <io/event.h>
#include <io/selector.h>
#include <rdma/fabric.h>
#include <rdma/fabric/context.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * @brief Entry for tracking immediate data completions and waiters
 */
struct ImmEntry {
  ImmContext* ctx{nullptr};  ///< Waiting context (nullptr if none)
  int pending{0};            ///< Pending completions count
};

/// Type alias for mapping immediate data values to entries
using ImmContextMap = std::unordered_map<uint64_t, ImmEntry>;

static constexpr size_t kMaxCQEntries = 128;  ///< Maximum completion queue entries per read

/**
 * @brief Libfabric-based selector for completion queue event multiplexing
 *
 * Implements the Selector interface using libfabric completion queues.
 * Supports both regular operation completions and immediate data events
 * from remote RDMA writes.
 */
class FabricSelector : public detail::Selector {
 public:
  using ms = std::chrono::milliseconds;

  /**
   * @brief Poll completion queues for events
   * @param duration Timeout duration (unused, non-blocking poll)
   * @return Vector of ready events
   */
  [[nodiscard]] std::vector<Event> Select(ms duration = ms{500}) override final {
    if (Stopped()) return {};
    std::vector<Event> res;
    struct fi_cq_data_entry cq_entries[kMaxCQEntries];
    for (auto cq : cqs_) {
      auto rc = fi_cq_read(cq, cq_entries, kMaxCQEntries);
      if (rc > 0) {
        HandleCompletion(cq_entries, rc, res, imm_);
      } else if (rc == -FI_EAVAIL) {
        HandleError(cq);
      } else if (rc == -FI_EAGAIN) {
        continue;
      } else {
        FatalError(rc);
      }
    }
    return res;
  }

  /**
   * @brief Register an EFA endpoint's completion queue
   * @tparam E EFA type (must have GetCQ() method)
   * @param efa EFA object to monitor
   */
  template <typename E>
  void Join(E& efa) noexcept {
    cqs_.emplace(efa.GetCQ());
  }

  /**
   * @brief Unregister an EFA endpoint's completion queue
   * @tparam E EFA type (must have GetCQ() method)
   * @param efa EFA object to stop monitoring
   */
  template <typename E>
  void Quit(E& efa) noexcept {
    cqs_.erase(efa.GetCQ());
  }

  /**
   * @brief Register an immediate data context for monitoring
   * @param ctx Context to monitor for immediate data events
   * @return true if a completion was already pending (no need to wait)
   */
  bool Join(ImmContext& ctx) {
    auto& entry = imm_[ctx.imm_data];
    if (entry.pending > 0) {
      // Completion already arrived, consume it
      entry.pending--;
      if (entry.pending == 0 && entry.ctx == nullptr) imm_.erase(ctx.imm_data);
      return true;  // Don't need to wait
    }
    entry.ctx = &ctx;
    return false;  // Need to wait
  }

  /**
   * @brief Unregister an immediate data context
   * @param ctx Context to stop monitoring
   */
  void Quit(ImmContext& ctx) {
    auto it = imm_.find(ctx.imm_data);
    if (it == imm_.end()) return;
    it->second.ctx = nullptr;
    if (it->second.pending == 0) {
      imm_.erase(it);
    }
  }

  /**
   * @brief Check if selector has no registered completion queues
   * @return true if no completion queues are registered
   */
  [[nodiscard]] bool Stopped() const noexcept override final { return cqs_.empty(); }

 private:
  /**
   * @brief Handle immediate data from remote RDMA write
   * @param entry Completion queue entry with immediate data
   * @param ret Output event vector
   * @param imm Immediate data context map
   */
  inline static void HandleImmdata(struct fi_cq_data_entry& entry, std::vector<Event>& ret, ImmContextMap& imm) {
    uint64_t imm_data = entry.data;
    if (!imm_data) [[unlikely]]
      return;
    auto& e = imm[imm_data];
    if (e.ctx && e.ctx->handle) {
      // Waiter is ready, wake it up
      e.ctx->entry = entry;
      ret.emplace_back(Event{static_cast<int>(entry.flags), e.ctx->handle});
      e.ctx = nullptr;  // Consumed
    } else {
      // No waiter yet, increment pending count
      e.pending++;
    }
  }

  /**
   * @brief Process successful completion queue entries
   * @param cq_entries Array of completion entries
   * @param n Number of entries
   * @param ret Output event vector
   * @param imm Immediate data context map
   */
  inline static void HandleCompletion(struct fi_cq_data_entry* cq_entries, size_t n, std::vector<Event>& ret, ImmContextMap& imm) {
    for (size_t i = 0; i < n; ++i) {
      auto& entry = cq_entries[i];
      auto flags = entry.flags;
      if (flags & FI_REMOTE_WRITE) {
        HandleImmdata(entry, ret, imm);
      } else {
        Context* context = reinterpret_cast<Context*>(entry.op_context);
        if (!context) continue;
        context->entry = entry;
        Handle* handle = context->handle;
        ret.emplace_back(Event{-1, flags, handle});
      }
    }
  }

  /**
   * @brief Handle completion queue errors
   * @param cq Completion queue with error
   * @throws std::runtime_error with error details
   */
  inline static void HandleError(struct fid_cq* cq) {
    struct fi_cq_err_entry err_entry;
    auto rc = fi_cq_readerr(cq, &err_entry, 0);
    if (rc < 0) {
      auto msg = fmt::format("fatal error. error({}): {}", rc, fi_strerror(-rc));
      throw std::runtime_error(msg);
    }
    if (rc > 0) {
      auto err = fi_cq_strerror(cq, err_entry.prov_errno, err_entry.err_data, nullptr, 0);
      auto msg = fmt::format("libfabric operation fail. error: {}", err);
      throw std::runtime_error(msg);
    } else {
      auto msg = fmt::format("fi_cq_readerr returned 0 (no error available) but -FI_EAVAIL was indicated");
      throw std::runtime_error(msg);
    }
  }

  /**
   * @brief Handle fatal libfabric errors
   * @param rc Error code from libfabric
   * @throws std::runtime_error with error details
   */
  inline static void FatalError(int rc) {
    auto msg = fmt::format("fatal error. error({}): {}", rc, fi_strerror(-rc));
    throw std::runtime_error(msg);
  }

 private:
  std::unordered_set<struct fid_cq*> cqs_;  ///< Registered completion queues
  ImmContextMap imm_;                       ///< Immediate data tracking map
};
