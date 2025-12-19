/**
 * @file event.h
 * @brief Event descriptor for epoll-based I/O multiplexing
 */
#pragma once
#include <io/handle.h>

#include <cstdint>

/**
 * @brief Event descriptor for epoll operations
 */
struct Event {
  int fd = -1;               ///< File descriptor
  uint32_t flags = 0;        ///< Epoll event flags
  Handle* handle = nullptr;  ///< Associated coroutine handle

  Event() = default;
  Event(int f, Handle* h = nullptr) : fd(f), flags(0), handle(h) {}
  Event(int f, uint64_t fl, Handle* h) : fd(f), flags(fl), handle(h) {}
  Event(const Event&) = default;
  Event& operator=(const Event&) = default;
  Event(Event&& other) noexcept
      : fd{std::exchange(other.fd, -1)}, flags{std::exchange(other.flags, 0)}, handle{std::exchange(other.handle, nullptr)} {}
  Event& operator=(Event&& other) noexcept {
    if (this != &other) {
      fd = std::exchange(other.fd, -1);
      flags = std::exchange(other.flags, 0);
      handle = std::exchange(other.handle, nullptr);
    }
    return *this;
  }
};
