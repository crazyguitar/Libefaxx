/**
 * @file context.h
 * @brief IBVerbs-specific context
 */
#pragma once
#include <io/handle.h>

#include <cstdint>

namespace ib {

/**
 * @brief Completion queue data entry (equivalent to fi_cq_data_entry)
 */
struct ib_cq_data_entry {
  void* op_context{};
  uint64_t flags{};
  size_t len{};
  void* buf{};
  uint64_t data{};
};

/**
 * @brief Context for completion queue operations
 */
struct Context {
  ib_cq_data_entry entry{};
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

}  // namespace ib
