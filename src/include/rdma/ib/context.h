/**
 * @file context.h
 * @brief Context structures for ibverbs completion queue operations
 */
#pragma once
#include <infiniband/verbs.h>
#include <io/handle.h>

namespace ib {

/**
 * @brief Completion queue data entry (equivalent to fi_cq_data_entry)
 *
 * Reference: libfabric/prov/efa/src/efa_cq.c:63-75 (efa_cq_read_data_entry)
 */
struct ib_cq_data_entry {
  void* op_context;  ///< Operation context (wr_id)
  uint64_t flags;    ///< Completion flags
  size_t len;        ///< Bytes transferred
  void* buf;         ///< Buffer (unused, set to NULL)
  uint64_t data;     ///< Immediate data
};

/**
 * @brief Context for completion queue operations (equivalent to fabric Context)
 */
struct Context {
  struct ib_cq_data_entry entry;  ///< Completion queue entry data
  Handle* handle;                 ///< Associated handle for the operation
};

/**
 * @brief Context for immediate data operations
 */
struct ImmContext : public Context {
  uint64_t imm_data;  ///< Expected immediate data value

  ImmContext() noexcept : Context{}, imm_data(0) {}
  ImmContext(uint64_t data) noexcept : Context{}, imm_data(data) {}
};

}  // namespace ib
