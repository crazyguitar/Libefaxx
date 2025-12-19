/**
 * @file selector.h
 * @brief I/O event multiplexing abstractions and epoll implementation
 */
#pragma once

#include <errno.h>
#include <io/common.h>
#include <io/event.h>
#include <spdlog/spdlog.h>
#include <sys/epoll.h>

#include <chrono>
#include <cstring>
#include <unordered_set>
#include <vector>

using namespace std::chrono_literals;

namespace detail {

/**
 * @brief Abstract interface for I/O event selection mechanisms
 */
class Selector {
 public:
  using ms = std::chrono::milliseconds;

  virtual ~Selector() = default;

  /**
   * @brief Wait for I/O events
   * @param duration Maximum time to wait for events
   * @return Vector of events that occurred
   */
  [[nodiscard]] virtual std::vector<Event> Select(ms duration = ms{500}) = 0;

  /**
   * @brief Check if selector has any registered event sources
   * @return true if no event sources are registered, false otherwise
   */
  [[nodiscard]] virtual bool Stopped() const noexcept = 0;
};

}  // namespace detail

/**
 * @brief Epoll-based selector implementation
 *
 * Concrete implementation using Linux epoll for I/O event multiplexing.
 */
class Selector : public detail::Selector, private NoCopy {
 public:
  using ms = std::chrono::milliseconds;

  Selector() : epoll_fd_{epoll_create1(0)}, events_(kMaxEvents) { ASSERT(epoll_fd_ >= 0); }

  ~Selector() override { close(epoll_fd_); }

  [[nodiscard]] std::vector<Event> Select(ms duration = ms{500}) override final;

  [[nodiscard]] bool Stopped() const noexcept override final { return fds_.empty(); }

  /**
   * @brief Register an event with epoll
   * @tparam E Event type (must have fd, flags, handle members)
   * @param e Event to monitor
   */
  template <typename E>
  void Join(E& e) noexcept {
    struct epoll_event event{0};
    event.events = e.flags;
    event.data.ptr = reinterpret_cast<void*>(std::addressof(e));
    int rc = epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, e.fd, &event);
    if (rc < 0) {
      int err = errno;
      SPDLOG_ERROR("epoll_ctl({}, EPOLL_CTL_ADD, fd={}, flags={}) fail. error: {}", epoll_fd_, e.fd, static_cast<unsigned>(e.flags), strerror(err));
      ASSERT(false);
    }
    fds_.emplace(e.fd);
  }

  /**
   * @brief Unregister an event from epoll
   * @tparam E Event type (must have fd, flags members)
   * @param e Event to stop monitoring
   */
  template <typename E>
  void Quit(E& e) noexcept {
    struct epoll_event event{0};
    event.events = e.flags;
    int rc = epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, e.fd, &event);
    if (rc < 0) {
      int err = errno;
      if (err != ENOENT && err != EBADF) {
        SPDLOG_WARN("epoll_ctl({}, EPOLL_CTL_DEL) fail. error: {}", e.fd, strerror(err));
      }
    }
    fds_.erase(e.fd);
  }

  static constexpr int kMaxEvents = 64;     ///< Maximum events per epoll_wait call
  int epoll_fd_ = -1;                       ///< Epoll file descriptor
  std::unordered_set<int> fds_;             ///< Set of registered file descriptors
  std::vector<struct epoll_event> events_;  ///< Event buffer for epoll_wait
};
