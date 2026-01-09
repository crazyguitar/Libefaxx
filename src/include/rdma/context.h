/**
 * @file context.h
 * @brief Unified CQ context for RDMA operations
 *
 * Template-based context that works with both IB and Fabric backends
 */
#pragma once

#include <io/handle.h>

#include <cstdint>

namespace rdma {

/**
 * @brief Context for completion queue operations
 * @tparam CQEntry The backend-specific CQ entry type (fi_cq_data_entry or ib_cq_data_entry)
 */
template <typename CQEntry>
struct Context {
  CQEntry entry{};   ///< Completion queue entry filled by selector
  Handle* handle{};  ///< Coroutine handle to resume on completion

  Context() noexcept = default;
  ~Context() noexcept = default;
  Context(const Context&) noexcept = default;
  Context(Context&&) noexcept = default;
  Context& operator=(const Context&) noexcept = default;
  Context& operator=(Context&&) noexcept = default;
};

/**
 * @brief Context for immediate data operations
 * @tparam CQEntry The backend-specific CQ entry type
 *
 * Extends Context with immediate data field for RDMA write with immediate.
 */
template <typename CQEntry>
struct ImmContext : public Context<CQEntry> {
  uint64_t imm_data{0};  ///< Expected immediate data value to match

  ImmContext() noexcept = default;
  ~ImmContext() noexcept = default;
  /** @brief Construct with specific immediate data value */
  explicit ImmContext(uint64_t data) noexcept : Context<CQEntry>{}, imm_data(data) {}
  ImmContext(const ImmContext&) noexcept = default;
  ImmContext(ImmContext&&) noexcept = default;
  ImmContext& operator=(const ImmContext&) noexcept = default;
  ImmContext& operator=(ImmContext&&) noexcept = default;
};

}  // namespace rdma
