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
#include <rdma/selector.h>

#include <unordered_set>
#include <vector>

namespace fi {

using ImmEntry = rdma::ImmEntry<ImmContext>;
using ImmContextMap = rdma::ImmContextMap<ImmContext>;
using rdma::kMaxCQEntries;

/**
 * @brief Libfabric-based selector for completion queue event multiplexing
 */
class FabricSelector : public detail::Selector {
 public:
  using ms = std::chrono::milliseconds;

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

  template <typename E>
  void Join(E& efa) noexcept {
    cqs_.emplace(efa.GetCQ());
  }

  template <typename E>
  void Quit(E& efa) noexcept {
    cqs_.erase(efa.GetCQ());
  }

  bool Join(ImmContext& ctx) { return rdma::JoinImm(imm_, ctx); }
  void Quit(ImmContext& ctx) { rdma::QuitImm(imm_, ctx); }

  [[nodiscard]] bool Stopped() const noexcept override final { return cqs_.empty(); }

 private:
  static void HandleCompletion(struct fi_cq_data_entry* cq_entries, size_t n, std::vector<Event>& ret, ImmContextMap& imm) {
    for (size_t i = 0; i < n; ++i) {
      auto& entry = cq_entries[i];
      if (entry.flags & FI_REMOTE_WRITE) {
        rdma::HandleImmdata(entry, ret, imm);
      } else {
        Context* context = reinterpret_cast<Context*>(entry.op_context);
        if (!context) continue;
        context->entry = entry;
        ret.emplace_back(Event{-1, entry.flags, context->handle});
      }
    }
  }

  static void HandleError(struct fid_cq* cq) {
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
      throw std::runtime_error("fi_cq_readerr returned 0 but -FI_EAVAIL was indicated");
    }
  }

  static void FatalError(int rc) { throw std::runtime_error(fmt::format("fatal error. error({}): {}", rc, fi_strerror(-rc))); }

 private:
  std::unordered_set<struct fid_cq*> cqs_;
  ImmContextMap imm_;
};

}  // namespace fi
