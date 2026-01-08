#pragma once
#include <io/handle.h>
#include <rdma/fi_domain.h>

namespace fi {

/**
 * @brief Context for completion queue operations
 */
struct Context {
  struct fi_cq_data_entry entry;  ///< Completion queue entry data
  Handle* handle;                 ///< Associated handle for the operation
};

/**
 * @brief Context for immediate data operations
 *
 * Extends base Context with immediate data support for RDMA operations.
 */
struct ImmContext : public Context {
  uint64_t imm_data;  ///< Immediate data value

  ImmContext() noexcept : Context{}, imm_data(0) {}
  ImmContext(uint64_t data) noexcept : Context{}, imm_data(data) {}
};

}  // namespace fi
