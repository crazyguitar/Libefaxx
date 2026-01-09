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
  struct fi_cq_data_entry entry{};
  Handle* handle{};
};

/**
 * @brief Context for immediate data operations
 */
struct ImmContext : public Context {
  uint64_t imm_data{0};

  ImmContext() noexcept = default;
  explicit ImmContext(uint64_t data) noexcept : Context{}, imm_data(data) {}
};

}  // namespace fi
