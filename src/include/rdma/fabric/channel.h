/**
 * @file channel.h
 * @brief RDMA channel for GPU memory communication over EFA
 */
#pragma once
#include <io/awaiter.h>
#include <io/common.h>
#include <io/coro.h>
#include <io/future.h>
#include <io/io.h>
#include <rdma/fabric/context.h>
#include <rdma/fabric/efa.h>
#include <rdma/fabric/selector.h>
#include <rdma/fi_rma.h>

namespace fi {

/**
 * @brief RDMA channel managing GPU memory registration and remote peer connection
 *
 * Handles libfabric memory registration with CUDA dmabuf support and maintains
 * connection to remote peer via address vector. Supports move semantics for
 * efficient resource transfer.
 */
class Channel : private NoCopy {
  /**
   * @brief Coroutine awaiter for asynchronous send operations
   *
   * Suspends coroutine while fi_sendmsg executes, resumes on completion.
   */
  struct SendAwaiter {
    void* __restrict__ data = nullptr;
    size_t size = 0;
    struct fid_mr* __restrict__ mr = nullptr;
    fi_addr_t remote;
    EFA* __restrict__ efa = nullptr;
    Context context{0};
    ssize_t rc = 0;

    /** @brief Always suspend to initiate async send */
    constexpr bool await_ready() const noexcept { return false; }

    /** @brief Return bytes sent or error code on resume */
    ssize_t await_resume() noexcept {
      if (rc != 0) [[unlikely]]
        return rc;
      return context.entry.len;
    }

    /**
     * @brief Initiate fi_sendmsg and suspend coroutine
     * @return true if suspended, false if immediate error
     */
    template <typename Promise>
    bool await_suspend(std::coroutine_handle<Promise> coroutine) noexcept {
      coroutine.promise().SetState(Handle::kSuspend);
      context.handle = &coroutine.promise();
      struct iovec iov{data, size};
      struct fi_msg msg{&iov, &mr->mem_desc, 1, remote, &context, 0};
      rc = fi_sendmsg(efa->GetEP(), &msg, 0);
      return rc == 0;
    }
  };

  /**
   * @brief Coroutine awaiter for asynchronous receive operations
   *
   * Suspends coroutine while fi_recvmsg executes, resumes on completion.
   */
  struct RecvAwaiter {
    void* __restrict__ data = nullptr;
    size_t size = 0;
    struct fid_mr* __restrict__ mr = nullptr;
    EFA* __restrict__ efa = nullptr;
    Context context{0};
    ssize_t rc = 0;

    /** @brief Always suspend to initiate async receive */
    constexpr bool await_ready() const noexcept { return false; }

    /** @brief Return bytes received or error code on resume */
    ssize_t await_resume() noexcept {
      if (rc != 0) [[unlikely]]
        return rc;
      return context.entry.len;
    }

    /**
     * @brief Initiate fi_recvmsg and suspend coroutine
     * @return true if suspended, false if immediate error
     */
    template <typename Promise>
    bool await_suspend(std::coroutine_handle<Promise> coroutine) noexcept {
      coroutine.promise().SetState(Handle::kSuspend);
      context.handle = &coroutine.promise();
      struct iovec iov{data, size};
      struct fi_msg msg{&iov, &mr->mem_desc, 1, FI_ADDR_UNSPEC, &context, 0};
      rc = fi_recvmsg(efa->GetEP(), &msg, 0);
      return rc == 0;
    }
  };

  /**
   * @brief Coroutine awaiter for asynchronous RDMA write operations
   *
   * Suspends coroutine while fi_writemsg executes, resumes on completion.
   * Supports remote memory access with optional immediate data.
   */
  struct WriteAwaiter {
    void* __restrict__ data = nullptr;
    size_t size = 0;
    struct fid_mr* __restrict__ mr = nullptr;
    uint64_t addr{0};
    uint64_t key{0};
    uint64_t imm_data{0};
    fi_addr_t remote;
    EFA* __restrict__ efa = nullptr;
    Context context{0};
    ssize_t rc = 0;

    /** @brief Always suspend to initiate async write */
    constexpr bool await_ready() const noexcept { return false; }

    /** @brief Return bytes written or error code on resume */
    ssize_t await_resume() noexcept {
      if (rc != 0) [[unlikely]]
        return rc;
      return context.entry.len;
    }

    /**
     * @brief Initiate fi_writemsg and suspend coroutine
     * @return true if suspended, false if immediate error
     */
    template <typename Promise>
    bool await_suspend(std::coroutine_handle<Promise> coroutine) {
      coroutine.promise().SetState(Handle::kSuspend);
      context.handle = &coroutine.promise();
      struct iovec iov{data, size};
      struct fi_rma_iov rma_iov{addr, size, key};
      struct fi_msg_rma msg{&iov, &mr->mem_desc, 1, remote, &rma_iov, 1, &context, imm_data};
      uint64_t flags = 0;
      if (imm_data) flags |= FI_REMOTE_CQ_DATA;
      rc = fi_writemsg(efa->GetEP(), &msg, flags);
      return rc == 0;
    }
  };

 public:
  Channel() = delete;

  /**
   * @brief Construct Channel with remote peer connection
   * @param efa Pointer to EFA instance for libfabric operations
   * @param addr Remote peer address to connect to
   *
   * Inserts remote peer address into address vector for communication.
   * Memory registration is handled separately by Buffer class.
   */
  Channel(EFA* efa, char* addr) : efa_{efa} {
    ASSERT(efa_);
    Connect(addr);
  }

  /**
   * @brief Move constructor - transfers ownership of channel resources
   * @param other Source Channel to move from
   */
  Channel(Channel&& other) noexcept : efa_{std::exchange(other.efa_, nullptr)}, remote_{std::exchange(other.remote_, FI_ADDR_UNSPEC)} {}

  /**
   * @brief Move assignment operator - transfers ownership of channel resources
   * @param other Source Channel to move from
   * @return Reference to this object
   */
  Channel& operator=(Channel&& other) noexcept {
    if (this != &other) {
      efa_ = std::exchange(other.efa_, nullptr);
      remote_ = std::exchange(other.remote_, FI_ADDR_UNSPEC);
    }
    return *this;
  }

  /**
   * @brief Destructor - cleans up remote address
   *
   * Removes remote peer from address vector.
   * Safe for moved-from objects.
   */
  ~Channel() {
    if (remote_ != FI_ADDR_UNSPEC && efa_) {
      auto av = efa_->GetAV();
      if (av) fi_av_remove(av, &remote_, 1, 0);
      remote_ = FI_ADDR_UNSPEC;
    }
    efa_ = nullptr;
  }

  EFA* GetEFA() const noexcept { return efa_; }

  /**
   * @brief Send data to remote peer
   * @param data Buffer to send
   * @param len Number of bytes to send
   * @param mr Memory region handle for the buffer
   * @return Coroutine returning bytes sent or negative error code
   */
  [[nodiscard]] Coro<ssize_t> Send(void* __restrict__ data, size_t len, struct fid_mr* mr) {
    co_return co_await Await<false, SendAwaiter>(data, len, mr, remote_, efa_);
  }

  /**
   * @brief Receive data from remote peer
   * @param data Buffer to receive into
   * @param len Maximum bytes to receive
   * @param mr Memory region handle for the buffer
   * @return Coroutine returning bytes received or negative error code
   */
  [[nodiscard]] Coro<ssize_t> Recv(void* __restrict__ data, size_t len, struct fid_mr* mr) {
    co_return co_await Await<false, RecvAwaiter>(data, len, mr, efa_);
  }

  /**
   * @brief Write data to remote peer's memory via RDMA
   * @param data Local buffer to write from
   * @param len Number of bytes to write
   * @param mr Memory region handle for local buffer
   * @param addr Remote memory address to write to
   * @param key Remote memory key for access
   * @param imm_data Optional immediate data to send with write
   * @return Coroutine returning bytes written or negative error code
   */
  [[nodiscard]] Coro<ssize_t> Write(void* __restrict__ data, size_t len, struct fid_mr* mr, uint64_t addr, uint64_t key, uint64_t imm_data = 0) {
    co_return co_await Await<false, WriteAwaiter>(data, len, mr, addr, key, imm_data, remote_, efa_);
  }

  /**
   * @brief Send all data to remote peer in chunks
   * @param data Buffer to send
   * @param len Total bytes to send
   * @param mr Memory region handle for the buffer
   * @return Coroutine returning total bytes sent or negative error code
   */
  [[nodiscard]] Coro<ssize_t> Sendall(void* __restrict__ data, size_t len, struct fid_mr* mr) {
    co_return co_await Gather<true, SendAwaiter>(data, len, mr, remote_, efa_);
  }

  /**
   * @brief Receive all data from remote peer in chunks
   * @param data Buffer to receive into
   * @param len Total bytes to receive
   * @param mr Memory region handle for the buffer
   * @return Coroutine returning total bytes received or negative error code
   */
  [[nodiscard]] Coro<ssize_t> Recvall(void* __restrict__ data, size_t len, struct fid_mr* mr) {
    co_return co_await Gather<true, RecvAwaiter>(data, len, mr, efa_);
  }

  /**
   * @brief Write all data to remote peer's memory via RDMA in chunks
   * @param data Local buffer to write from
   * @param len Total bytes to write
   * @param mr Memory region handle for local buffer
   * @param addr Remote memory address to write to
   * @param key Remote memory key for access
   * @param imm_data Optional immediate data to send with write (sent only with last chunk)
   * @return Coroutine returning total bytes written or negative error code
   */
  [[nodiscard]] Coro<ssize_t> Writeall(void* __restrict__ data, size_t len, struct fid_mr* mr, uint64_t addr, uint64_t key, uint64_t imm_data = 0) {
    co_return co_await GatherWrite<true>(data, len, mr, addr, key, imm_data);
  }

  /**
   * @brief Insert remote peer address into address vector
   * @param addr Remote peer address buffer
   */
  void Connect(char* addr) {
    auto av = efa_->GetAV();
    FI_EXPECT(fi_av_insert(av, addr, 1, &remote_, 0, nullptr), 1);
  }

 private:
  /**
   * @brief Await single or complete transfer operation with retry on EAGAIN
   * @tparam all If true, loops until all data transferred; if false, single operation
   * @tparam Awaiter SendAwaiter or RecvAwaiter type
   * @tparam Args Variadic arguments forwarded to Awaiter constructor
   * @param data Buffer pointer
   * @param len Buffer length
   * @param args Additional arguments (mr, remote, efa)
   * @return Coroutine returning bytes transferred or negative error code
   *
   * When all=true, retries partial transfers until len bytes complete.
   * When all=false, performs single operation with EAGAIN retry.
   */
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
        } else if (rc == -FI_EAGAIN) {
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
      } else if (rc == -FI_EAGAIN) {
        co_await YieldAwaiter{};
      } else {
        co_return rc;
      }
    }
    co_return rc;
  }

  /**
   * @brief Parallel chunked transfer using multiple coroutines
   * @tparam Awaiter SendAwaiter or RecvAwaiter type
   * @tparam Args Variadic arguments forwarded to Awaiter constructor
   * @param data Buffer pointer
   * @param len Total buffer length
   * @param args Additional arguments (mr, remote, efa)
   * @return Coroutine returning total bytes transferred
   * @throws std::runtime_error on transfer failure
   *
   * Splits data into kChunkSize chunks, creates parallel coroutines for each,
   * and awaits all completions. Each chunk uses Await<true> for reliability.
   */
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

    // Must wait for ALL futures even on error to prevent use-after-free.
    // Early return would destroy coros while RDMA ops are in-flight, causing
    // completion handler to resume destroyed coroutines.
    size_t n = 0;
    ssize_t first_error = 0;
    for (auto& c : coros) {
      auto rc = co_await c;
      if (rc < 0 && first_error == 0) [[unlikely]]
        first_error = rc;
      else if (rc >= 0)
        n += static_cast<size_t>(rc);
    }
    if (first_error < 0) throw std::runtime_error(fmt::format("error: {} total: {}", fi_strerror(-first_error), n));
    co_return n;
  }

  /**
   * @brief Parallel chunked RDMA write with address offset handling
   */
  template <bool all>
  Coro<size_t> GatherWrite(void* data, size_t len, struct fid_mr* mr, uint64_t addr, uint64_t key, uint64_t imm_data) {
    const size_t num_chunks = (len + kChunkSize - 1) / kChunkSize;
    std::vector<Future<Coro<ssize_t>>> coros;
    coros.reserve(num_chunks);

    for (size_t offset = 0; offset < len; offset += kChunkSize) {
      void* ptr = static_cast<char*>(data) + offset;
      size_t size = std::min(kChunkSize, len - offset);
      uint64_t chunk_addr = addr + offset;
      uint64_t chunk_imm = (offset + size >= len) ? imm_data : 0;  // Only last chunk gets imm_data
      coros.emplace_back(Await<all, WriteAwaiter>(ptr, size, mr, chunk_addr, key, chunk_imm, remote_, efa_));
    }

    size_t n = 0;
    ssize_t first_error = 0;
    for (auto& c : coros) {
      auto rc = co_await c;
      if (rc < 0 && first_error == 0) [[unlikely]]
        first_error = rc;
      else if (rc >= 0)
        n += static_cast<size_t>(rc);
    }
    if (first_error < 0) throw std::runtime_error(fmt::format("RDMA write error: {}", fi_strerror(-first_error)));
    co_return n;
  }

 private:
  static constexpr size_t kChunkSize = 1 << 20;  // 1MB (was 256k)
  EFA* efa_ = nullptr;
  fi_addr_t remote_ = FI_ADDR_UNSPEC;
};

}  // namespace fi
