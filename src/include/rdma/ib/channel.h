/**
 * @file channel.h
 * @brief RDMA channel for GPU memory communication over EFA using ibverbs
 */
#pragma once
#include <io/awaiter.h>
#include <io/coro.h>
#include <io/future.h>
#include <io/io.h>
#include <rdma/ib/context.h>
#include <rdma/ib/efa.h>
#include <rdma/ib/ib.h>
#include <rdma/ib/selector.h>

namespace ib {

/**
 * @brief RDMA channel managing GPU memory registration and remote peer connection
 */
class Channel : private NoCopy {
  /**
   * @brief Coroutine awaiter for asynchronous RDMA write operations
   */
  struct WriteAwaiter {
    void* __restrict__ data = nullptr;
    size_t size = 0;
    uint32_t lkey = 0;
    uint64_t addr = 0;
    uint32_t rkey = 0;
    uint64_t imm_data = 0;
    ibv_ah* ah = nullptr;
    uint32_t qpn = 0;
    uint32_t qkey = 0;
    ib_ep* ep = nullptr;
    Context context{};
    ssize_t rc = 0;

    constexpr bool await_ready() const noexcept { return false; }

    ssize_t await_resume() noexcept {
      if (rc != 0) [[unlikely]]
        return rc;
      return context.entry.len;
    }

    template <typename Promise>
    bool await_suspend(std::coroutine_handle<Promise> coroutine) {
      coroutine.promise().SetState(Handle::kSuspend);
      context.handle = &coroutine.promise();

      iovec iov{data, size};
      ib_rma_iov rma_iov{addr, size, rkey};
      ib_msg_rma msg{&iov, &lkey, 1, &rma_iov, 1, ah, qpn, qkey, &context, imm_data};
      uint64_t flags = imm_data ? IB_REMOTE_CQ_DATA : 0;
      rc = ib_writemsg(ep, &msg, flags);
      return rc == 0;
    }
  };

 public:
  Channel() = delete;

  Channel(EFA* efa, char* addr) : efa_{efa} {
    ASSERT(efa_);
    Connect(addr);
  }

  Channel(Channel&& other) noexcept
      : efa_{std::exchange(other.efa_, nullptr)}, ah_{std::exchange(other.ah_, nullptr)}, qpn_{other.qpn_}, qkey_{other.qkey_} {}

  Channel& operator=(Channel&& other) noexcept {
    if (this != &other) {
      if (ah_) ibv_destroy_ah(ah_);
      efa_ = std::exchange(other.efa_, nullptr);
      ah_ = std::exchange(other.ah_, nullptr);
      qpn_ = other.qpn_;
      qkey_ = other.qkey_;
    }
    return *this;
  }

  ~Channel() {
    if (ah_) {
      ibv_destroy_ah(ah_);
      ah_ = nullptr;
    }
    efa_ = nullptr;
  }

  EFA* GetEFA() const noexcept { return efa_; }

  /**
   * @brief Write data to remote peer's memory via RDMA
   */
  [[nodiscard]] Coro<ssize_t> Write(void* __restrict__ data, size_t len, ib_mr* mr, uint64_t addr, uint32_t key, uint64_t imm_data = 0) {
    co_return co_await Await<false, WriteAwaiter>(data, len, ib_mr_lkey(mr), addr, key, imm_data, ah_, qpn_, qkey_, efa_->GetEP());
  }

  /**
   * @brief Write all data to remote peer's memory via RDMA in chunks
   */
  [[nodiscard]] Coro<ssize_t> Writeall(void* __restrict__ data, size_t len, ib_mr* mr, uint64_t addr, uint32_t key, uint64_t imm_data = 0) {
    co_return co_await GatherWrite<true>(data, len, mr, addr, key, imm_data);
  }

  void Connect(char* addr) {
    auto* av = efa_->GetAV();
    ah_ = ib_av_insert(av, addr);
    ASSERT(ah_);
    qpn_ = ib_addr_qpn(addr);
    qkey_ = ib_addr_qkey(addr);
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

  template <bool all>
  Coro<size_t> GatherWrite(void* data, size_t len, ib_mr* mr, uint64_t addr, uint32_t key, uint64_t imm_data) {
    const size_t num_chunks = (len + kChunkSize - 1) / kChunkSize;
    std::vector<Future<Coro<ssize_t>>> coros;
    coros.reserve(num_chunks);

    for (size_t offset = 0; offset < len; offset += kChunkSize) {
      void* ptr = static_cast<char*>(data) + offset;
      size_t size = std::min(kChunkSize, len - offset);
      uint64_t chunk_addr = addr + offset;
      uint64_t chunk_imm = (offset + size >= len) ? imm_data : 0;
      coros.emplace_back(Await<all, WriteAwaiter>(ptr, size, ib_mr_lkey(mr), chunk_addr, key, chunk_imm, ah_, qpn_, qkey_, efa_->GetEP()));
    }

    size_t n = 0;
    for (auto& c : coros) {
      auto rc = co_await c;
      if (rc < 0) [[unlikely]] {
        throw std::runtime_error(fmt::format("ib_writemsg error: {} total: {}", strerror(-rc), n));
      }
      n += static_cast<size_t>(rc);
    }
    co_return n;
  }

 private:
  static constexpr size_t kChunkSize = 1 << 20;  // 1MB
  EFA* efa_ = nullptr;
  ibv_ah* ah_ = nullptr;
  uint32_t qpn_ = 0;
  uint32_t qkey_ = 0;
};

}  // namespace ib
