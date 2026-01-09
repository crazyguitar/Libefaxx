/**
 * @file channel.h
 * @brief Unified RDMA channel for GPU memory communication
 *
 * Template-based channel that works with both IB and Fabric backends
 */
#pragma once

#include <io/awaiter.h>
#include <io/coro.h>
#include <io/future.h>
#include <io/io.h>
#include <rdma/buffer.h>

#include <vector>

namespace rdma {

static constexpr size_t kChunkSize = 1 << 20;  // 1MB

/**
 * @brief RDMA channel managing remote peer connection
 *
 * @tparam Backend Backend traits (fi::Backend or ib::Backend)
 */
template <typename Backend>
class Channel : private NoCopy {
 public:
  using EFA = typename Backend::efa_type;
  using MR = typename Backend::mr_type;
  using Context = typename Backend::context_type;
  using ImmContext = typename Backend::imm_context_type;
  using Selector = typename Backend::selector_type;

 private:
  /**
   * @brief Coroutine awaiter for RDMA write operations
   */
  struct WriteAwaiter {
    void* __restrict__ data = nullptr;
    size_t size = 0;
    MR mr = nullptr;
    uint64_t addr = 0;
    typename Backend::key_type key{};
    uint64_t imm_data = 0;
    typename Backend::remote_addr_type remote{};
    EFA* efa = nullptr;
    Context context{};
    ssize_t rc = 0;

    constexpr bool await_ready() const noexcept { return false; }

    ssize_t await_resume() noexcept {
      if (rc != 0) [[unlikely]]
        return rc;
      return Backend::GetWriteLen(context, size);
    }

    template <typename Promise>
    bool await_suspend(std::coroutine_handle<Promise> coroutine) {
      coroutine.promise().SetState(Handle::kSuspend);
      context.handle = &coroutine.promise();
      rc = Backend::PostWrite(efa, data, size, mr, addr, key, imm_data, remote, &context);
      return rc == 0;
    }
  };

  /**
   * @brief Coroutine awaiter for send operations
   */
  struct SendAwaiter {
    void* __restrict__ data = nullptr;
    size_t size = 0;
    MR mr = nullptr;
    uint64_t addr = 0;
    typename Backend::key_type key{};
    uint64_t imm_data = 0;
    typename Backend::remote_addr_type remote{};
    EFA* efa = nullptr;
    Context context{};
    ssize_t rc = 0;

    constexpr bool await_ready() const noexcept { return false; }

    ssize_t await_resume() noexcept {
      if (rc != 0) [[unlikely]]
        return rc;
      return Backend::GetSendLen(context, size);
    }

    template <typename Promise>
    bool await_suspend(std::coroutine_handle<Promise> coroutine) {
      coroutine.promise().SetState(Handle::kSuspend);
      context.handle = &coroutine.promise();
      rc = Backend::PostSend(efa, data, size, mr, addr, key, imm_data, remote, &context);
      return rc == 0;
    }
  };

  /**
   * @brief Coroutine awaiter for receive operations (Fabric only)
   */
  struct RecvAwaiter {
    void* __restrict__ data = nullptr;
    size_t size = 0;
    MR mr = nullptr;
    EFA* efa = nullptr;
    Context context{};
    ssize_t rc = 0;

    constexpr bool await_ready() const noexcept { return false; }

    ssize_t await_resume() noexcept {
      if (rc != 0) [[unlikely]]
        return rc;
      return Backend::GetRecvLen(context);
    }

    template <typename Promise>
    bool await_suspend(std::coroutine_handle<Promise> coroutine) {
      coroutine.promise().SetState(Handle::kSuspend);
      context.handle = &coroutine.promise();
      rc = Backend::PostRecv(efa, data, size, mr, &context);
      return rc == 0;
    }
  };

 public:
  Channel() = delete;

  Channel(EFA* efa, const char* addr) : efa_{efa} {
    ASSERT(efa_);
    remote_ = Backend::Connect(efa_, addr);
  }

  Channel(Channel&& other) noexcept : efa_{std::exchange(other.efa_, nullptr)}, remote_{std::exchange(other.remote_, Backend::kInvalidAddr)} {}

  Channel& operator=(Channel&& other) noexcept {
    if (this != &other) {
      Backend::Disconnect(efa_, remote_);
      efa_ = std::exchange(other.efa_, nullptr);
      remote_ = std::exchange(other.remote_, Backend::kInvalidAddr);
    }
    return *this;
  }

  ~Channel() {
    Backend::Disconnect(efa_, remote_);
    efa_ = nullptr;
  }

  EFA* GetEFA() const noexcept { return efa_; }

  [[nodiscard]] Coro<ssize_t>
  Write(void* __restrict__ data, size_t len, MR mr, uint64_t addr, typename Backend::key_type key, uint64_t imm_data = 0) {
    co_return co_await Await<false, WriteAwaiter>(data, len, mr, addr, key, imm_data, remote_, efa_);
  }

  [[nodiscard]] Coro<ssize_t>
  Writeall(void* __restrict__ data, size_t len, MR mr, uint64_t addr, typename Backend::key_type key, uint64_t imm_data = 0) {
    co_return co_await GatherWrite<true>(data, len, mr, addr, key, imm_data);
  }

  [[nodiscard]] Coro<ssize_t> Send(void* __restrict__ data, size_t len, MR mr) {
    co_return co_await Await<false, SendAwaiter>(data, len, mr, 0, typename Backend::key_type{}, 0, remote_, efa_);
  }

  [[nodiscard]] Coro<ssize_t> Recv(void* __restrict__ data, size_t len, MR mr) { co_return co_await Await<false, RecvAwaiter>(data, len, mr, efa_); }

  [[nodiscard]] Coro<ssize_t> Sendall(void* __restrict__ data, size_t len, MR mr) {
    co_return co_await Gather<true, SendAwaiter>(data, len, mr, 0, typename Backend::key_type{}, 0, remote_, efa_);
  }

  [[nodiscard]] Coro<ssize_t> Recvall(void* __restrict__ data, size_t len, MR mr) {
    co_return co_await Gather<true, RecvAwaiter>(data, len, mr, efa_);
  }

  // IB-specific: Sendall/Recvall with remote RMA info
  [[nodiscard]] Coro<ssize_t> Sendall(void* __restrict__ data, size_t len, MR mr, uint64_t addr, typename Backend::key_type key, uint64_t imm_data) {
    co_return co_await GatherSend<true>(data, len, mr, addr, key, imm_data);
  }

  [[nodiscard]] Coro<ssize_t> Recvall(void* __restrict__ data, size_t len, MR mr, uint64_t imm_data) {
    co_await ImmdataAwaiter<Selector, ImmContext>{imm_data};
    co_return static_cast<ssize_t>(len);
  }

 private:
  template <bool all, typename Awaiter, typename... Args>
  Coro<ssize_t> Await(void* data, size_t len, Args... args) {
    if constexpr (all) {
      size_t n = 0;
      while (n < len) {
        void* ptr = static_cast<char*>(data) + n;
        size_t remaining = len - n;
        auto rc = co_await Awaiter{ptr, remaining, args...};
        if (rc >= 0) [[likely]] {
          n += static_cast<size_t>(rc);
        } else if (rc == -EAGAIN) {
          co_await YieldAwaiter{};
        } else {
          co_return rc;
        }
      }
      co_return static_cast<ssize_t>(n);
    }

    ssize_t rc = 0;
    while (true) {
      rc = co_await Awaiter{data, len, args...};
      if (rc >= 0) [[likely]] {
        break;
      } else if (rc == -EAGAIN) {
        co_await YieldAwaiter{};
      } else {
        co_return rc;
      }
    }
    co_return rc;
  }

  template <bool all, typename Awaiter, typename... Args>
  Coro<size_t> Gather(void* data, size_t len, Args... args) {
    const size_t num_chunks = (len + kChunkSize - 1) / kChunkSize;
    std::vector<Future<Coro<ssize_t>>> coros;
    coros.reserve(num_chunks);

    for (size_t offset = 0; offset < len; offset += kChunkSize) {
      void* ptr = static_cast<char*>(data) + offset;
      size_t size = std::min(kChunkSize, len - offset);
      coros.emplace_back(Await<all, Awaiter>(ptr, size, args...));
    }

    size_t n = 0;
    for (auto& c : coros) {
      auto rc = co_await c;
      if (rc < 0) [[unlikely]]
        throw std::runtime_error(fmt::format("RDMA error: {}", strerror(-rc)));
      n += static_cast<size_t>(rc);
    }
    co_return n;
  }

  template <bool all>
  Coro<size_t> GatherWrite(void* data, size_t len, MR mr, uint64_t addr, typename Backend::key_type key, uint64_t imm_data) {
    const size_t num_chunks = (len + kChunkSize - 1) / kChunkSize;
    std::vector<Future<Coro<ssize_t>>> coros;
    coros.reserve(num_chunks);

    for (size_t offset = 0; offset < len; offset += kChunkSize) {
      void* ptr = static_cast<char*>(data) + offset;
      size_t size = std::min(kChunkSize, len - offset);
      uint64_t chunk_addr = addr + offset;
      uint64_t chunk_imm = (offset + size >= len) ? imm_data : 0;
      coros.emplace_back(Await<all, WriteAwaiter>(ptr, size, mr, chunk_addr, key, chunk_imm, remote_, efa_));
    }

    size_t n = 0;
    for (auto& c : coros) {
      auto rc = co_await c;
      if (rc < 0) [[unlikely]]
        throw std::runtime_error(fmt::format("RDMA write error: {}", strerror(-rc)));
      n += static_cast<size_t>(rc);
    }
    co_return n;
  }

  template <bool all>
  Coro<size_t> GatherSend(void* data, size_t len, MR mr, uint64_t addr, typename Backend::key_type key, uint64_t imm_data) {
    const size_t num_chunks = (len + kChunkSize - 1) / kChunkSize;
    std::vector<Future<Coro<ssize_t>>> coros;
    coros.reserve(num_chunks);

    for (size_t offset = 0; offset < len; offset += kChunkSize) {
      void* ptr = static_cast<char*>(data) + offset;
      size_t size = std::min(kChunkSize, len - offset);
      uint64_t chunk_addr = addr + offset;
      uint64_t chunk_imm = (offset + size >= len) ? imm_data : 0;
      coros.emplace_back(Await<all, SendAwaiter>(ptr, size, mr, chunk_addr, key, chunk_imm, remote_, efa_));
    }

    size_t n = 0;
    for (auto& c : coros) {
      auto rc = co_await c;
      if (rc < 0) [[unlikely]]
        throw std::runtime_error(fmt::format("RDMA send error: {}", strerror(-rc)));
      n += static_cast<size_t>(rc);
    }
    co_return n;
  }

 private:
  EFA* efa_ = nullptr;
  typename Backend::remote_addr_type remote_ = Backend::kInvalidAddr;
};

}  // namespace rdma
