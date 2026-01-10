/**
 * @file buffer.h
 * @brief Shared RDMA buffer utilities
 *
 * Common awaiter for immediate data operations used by both IB and Fabric buffers
 */
#pragma once

#include <io/handle.h>
#include <io/io.h>

namespace rdma {

/**
 * @brief Awaiter for immediate data operations
 */
template <typename Selector, typename ImmCtx>
struct ImmdataAwaiter {
  uint64_t imm_data{0};
  ImmCtx context{0};

  constexpr bool await_ready() const noexcept { return false; }

  void await_resume() noexcept { IO::Get().Quit<Selector>(context); }

  template <typename Promise>
  bool await_suspend(std::coroutine_handle<Promise> coroutine) {
    if (imm_data == 0) [[unlikely]]
      return false;
    context.handle = &coroutine.promise();
    context.imm_data = imm_data;
    if (IO::Get().Join<Selector>(context)) return false;
    coroutine.promise().SetState(Handle::kSuspend);
    return true;
  }
};

}  // namespace rdma
