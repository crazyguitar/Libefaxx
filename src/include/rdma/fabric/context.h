/**
 * @file context.h
 * @brief Libfabric-specific context for completion queue operations
 */
#pragma once
#include <io/handle.h>
#include <rdma/fi_domain.h>

namespace fi {

/**
 * @brief Context for completion queue operations
 */
struct Context {
  struct fi_cq_data_entry entry{};  ///< Completion entry filled by selector
  Handle* handle{};                 ///< Coroutine handle to resume
};

/**
 * @brief Context for immediate data operations
 *
 * Extends Context with immediate data field for RDMA write with immediate.
 */
struct ImmContext : public Context {
  uint64_t imm_data{0};  ///< Expected immediate data value

  ImmContext() noexcept = default;
  /** @brief Construct with specific immediate data value */
  explicit ImmContext(uint64_t data) noexcept : Context{}, imm_data(data) {}
};

}  // namespace fi
