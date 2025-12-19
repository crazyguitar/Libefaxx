/**
 * @file runner.h
 * @brief Utility for running coroutines to completion using the I/O event loop
 */
#pragma once
#include <io/future.h>
#include <io/io.h>

#include <type_traits>

/**
 * @brief Run a coroutine to completion using the I/O event loop
 * @param coro Coroutine to execute
 * @return The coroutine's result value
 */
template <typename C>
decltype(auto) Run(C&& coro) {
  auto fut = Future(std::forward<C>(coro));
  IO::Get().Run();
  if constexpr (std::is_lvalue_reference_v<C&&>) return fut.result();
  return std::move(fut).result();
}
