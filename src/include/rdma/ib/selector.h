/**
 * @file selector.h
 * @brief ibverbs completion queue selector for async I/O
 */
#pragma once
#include <rdma/ib/context.h>
#include <rdma/ib/ib.h>
#include <rdma/selector.h>
#include <spdlog/spdlog.h>

namespace ib {

/**
 * @brief Backend traits for ibverbs
 */
struct IBBackend {
  using CQ = ib_cq;
  using CQEntry = ib_cq_data_entry;
  using Context = ib::Context;
  using ImmContext = ib::ImmContext;

  static ssize_t CQRead(CQ* cq, CQEntry* entries, size_t n) { return ib_cq_read(cq, entries, n); }
  static bool IsRemoteWrite(const CQEntry& e) { return e.data != 0; }
  static void HandleError(CQ*, ssize_t rc) { SPDLOG_ERROR("ib_cq_read error: {}", rc); }
};

using IBSelector = rdma::RdmaSelector<IBBackend>;

}  // namespace ib
