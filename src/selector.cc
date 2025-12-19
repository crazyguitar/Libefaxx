#include <io/selector.h>
#include <io/stream.h>

std::vector<Event> Selector::Select(ms duration) {
  if (Stopped()) return {};
  int timeout = duration.count();
  int n = epoll_wait(epoll_fd_, events_.data(), events_.size(), timeout);
  std::vector<Event> res;
  if (n < 0) {
    int err = errno;
    if (err != EINTR) SPDLOG_WARN("epoll_wait fail. error: {}", strerror(err));
    return res;
  }
  for (int i = 0; i < n; ++i) {
    auto& ev = events_[i];
    auto event = reinterpret_cast<Event*>(ev.data.ptr);
    if (!event) continue;
    Handle* handle = event->handle;
    if (!handle) continue;
    res.emplace_back(Event{event->fd, ev.events, handle});
  }
  return res;
}
