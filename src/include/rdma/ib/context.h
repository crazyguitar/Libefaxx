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

  ib_cq_data_entry() noexcept = default;
  ~ib_cq_data_entry() noexcept = default;
  ib_cq_data_entry(const ib_cq_data_entry&) noexcept = default;
  ib_cq_data_entry(ib_cq_data_entry&&) noexcept = default;
  ib_cq_data_entry& operator=(const ib_cq_data_entry&) noexcept = default;
  ib_cq_data_entry& operator=(ib_cq_data_entry&&) noexcept = default;
};

/**
 * @brief Context for completion queue operations
 */
struct Context {
  ib_cq_data_entry entry{};
  Handle* handle{};

  Context() noexcept = default;
  ~Context() noexcept = default;
  Context(const Context&) noexcept = default;
  Context(Context&&) noexcept = default;
  Context& operator=(const Context&) noexcept = default;
  Context& operator=(Context&&) noexcept = default;
};

/**
 * @brief Context for immediate data operations
 *
 * Flattened structure (no inheritance) for nvcc compatibility with std::unordered_map.
 * Contains entry, handle, and imm_data directly.
 */
struct ImmContext {
  ib_cq_data_entry entry{};
  Handle* handle{};
  uint64_t imm_data{0};

  ImmContext() noexcept = default;
  ~ImmContext() noexcept = default;
  explicit ImmContext(uint64_t data) noexcept : entry{}, handle{}, imm_data(data) {}
  ImmContext(const ImmContext&) noexcept = default;
  ImmContext(ImmContext&&) noexcept = default;
  ImmContext& operator=(const ImmContext&) noexcept = default;
  ImmContext& operator=(ImmContext&&) noexcept = default;
};

}  // namespace ib
