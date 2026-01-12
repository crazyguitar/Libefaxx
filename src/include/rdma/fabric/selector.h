/**
 * @file selector.h
 * @brief Libfabric completion queue selector for async I/O
 */
#pragma once
#include <rdma/fabric.h>
#include <rdma/fabric/context.h>
#include <rdma/fi_domain.h>
#include <rdma/selector.h>

namespace fi {

/**
 * @brief Backend traits for libfabric
 */
struct FabricBackend {
  using CQ = fid_cq;
  using CQEntry = fi_cq_data_entry;
  using Context = fi::Context;
  using ImmContext = fi::ImmContext;

  static ssize_t CQRead(CQ* cq, CQEntry* entries, size_t n) {
    auto rc = fi_cq_read(cq, entries, n);
    if (rc == -FI_EAGAIN) return 0;
    if (rc == -FI_EAVAIL) {
      HandleCQError(cq);
      return 0;
    }
    return rc;
  }

  static bool IsRemoteWrite(const CQEntry& e) { return e.flags & FI_REMOTE_WRITE; }

  static void HandleError(CQ*, ssize_t rc) { throw std::runtime_error(fmt::format("fatal error. error({}): {}", rc, fi_strerror(-rc))); }

 private:
  static void HandleCQError(CQ* cq) {
    fi_cq_err_entry err_entry;
    auto rc = fi_cq_readerr(cq, &err_entry, 0);
    if (rc < 0) throw std::runtime_error(fmt::format("fatal error. error({}): {}", rc, fi_strerror(-rc)));
    if (rc > 0) {
      auto err = fi_cq_strerror(cq, err_entry.prov_errno, err_entry.err_data, nullptr, 0);
      throw std::runtime_error(fmt::format("libfabric operation fail. error: {}", err));
    }
    throw std::runtime_error("fi_cq_readerr returned 0 but -FI_EAVAIL was indicated");
  }
};

using FabricSelector = rdma::RdmaSelector<FabricBackend>;

}  // namespace fi
