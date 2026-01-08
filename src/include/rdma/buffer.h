/**
 * @file buffer.h
 * @brief Shared buffer utilities for RDMA backends
 */
#pragma once
#include <io/coro.h>
#include <io/handle.h>
#include <io/io.h>

namespace rdma {

/**
 * @brief Awaiter for immediate data operations
 * @tparam Selector The selector type (FabricSelector or IBSelector)
 * @tparam ImmCtx The immediate context type
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
