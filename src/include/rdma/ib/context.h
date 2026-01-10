/**
 * @file context.h
 * @brief IBVerbs-specific context for completion queue operations
 */
#pragma once
#include <io/handle.h>

#include <cstdint>

namespace ib {

/**
 * @brief Completion queue data entry (equivalent to fi_cq_data_entry)
 *
 * Unified structure for ibverbs completion data, matching libfabric's interface.
 */
struct ib_cq_data_entry {
  void* op_context{};  ///< User context from work request
  uint64_t flags{};    ///< Completion flags
  size_t len{};        ///< Bytes transferred
  void* buf{};         ///< Buffer pointer (unused for EFA)
  uint64_t data{};     ///< Immediate data value

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
  ib_cq_data_entry entry{};  ///< Completion entry filled by selector
  Handle* handle{};          ///< Coroutine handle to resume

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
  ib_cq_data_entry entry{};  ///< Completion entry filled by selector
  Handle* handle{};          ///< Coroutine handle to resume
  uint64_t imm_data{0};      ///< Expected immediate data value

  ImmContext() noexcept = default;
  ~ImmContext() noexcept = default;
  /** @brief Construct with specific immediate data value */
  explicit ImmContext(uint64_t data) noexcept : entry{}, handle{}, imm_data(data) {}
  ImmContext(const ImmContext&) noexcept = default;
  ImmContext(ImmContext&&) noexcept = default;
  ImmContext& operator=(const ImmContext&) noexcept = default;
  ImmContext& operator=(ImmContext&&) noexcept = default;
};

}  // namespace ib
