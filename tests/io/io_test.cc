/**
 * @file io_test.cc
 * @brief Comprehensive unit tests for src/include/io components
 */

#include <arpa/inet.h>
#include <fcntl.h>
#include <io/client.h>
#include <io/common.h>
#include <io/coro.h>
#include <io/defer.h>
#include <io/event.h>
#include <io/future.h>
#include <io/handle.h>
#include <io/io.h>
#include <io/result.h>
#include <io/runner.h>
#include <io/selector.h>
#include <io/server.h>
#include <io/sleep.h>
#include <io/stream.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <spdlog/spdlog.h>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// =============================================================================
// Test Framework
// =============================================================================
static int g_test_pass = 0;
static int g_test_fail = 0;

#define REQUIRE(exp)                                                               \
  do {                                                                             \
    if (!(exp)) {                                                                  \
      SPDLOG_ERROR("  [FAIL] {}:{}: REQUIRE({})", __FILE__, __LINE__, #exp);  \
      g_test_fail++;                                                               \
      return;                                                                      \
    }                                                                              \
  } while (0)

#define REQUIRE_THROWS(expr)                                                       \
  do {                                                                             \
    bool threw = false;                                                            \
    try { (void)(expr); } catch (...) { threw = true; }                            \
    if (!threw) {                                                                  \
      SPDLOG_ERROR("  [FAIL] {}:{}: expected exception", __FILE__, __LINE__); \
      g_test_fail++;                                                               \
      return;                                                                      \
    }                                                                              \
  } while (0)

#define REQUIRE_NOTHROW(expr)                                                      \
  do {                                                                             \
    try { (void)(expr); }                                                          \
    catch (...) {                                                                  \
      SPDLOG_ERROR("  [FAIL] {}:{}: unexpected exception", __FILE__, __LINE__); \
      g_test_fail++;                                                               \
      return;                                                                      \
    }                                                                              \
  } while (0)

// =============================================================================
// Helper: Counted class for tracking object lifetimes
// =============================================================================
struct Counted {
  static void Reset() { move_ctor = copy_ctor = default_ctor = dtor = 0; }
  Counted() { id = default_ctor++; }
  ~Counted() { ++dtor; }
  Counted(const Counted&) { ++copy_ctor; }
  Counted(Counted&& o) noexcept { ++move_ctor; o.id = -1; }
  Counted& operator=(const Counted&) = default;
  Counted& operator=(Counted&&) = default;

  static int Constructs() { return move_ctor + copy_ctor + default_ctor; }
  static int Alive() { return Constructs() - dtor; }

  int id = 0;
  static inline int move_ctor = 0;
  static inline int copy_ctor = 0;
  static inline int default_ctor = 0;
  static inline int dtor = 0;
};

// =============================================================================
// Helper: Get ephemeral port
// =============================================================================
int GetPort() {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return 0;

  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(0);

  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    close(fd);
    return 0;
  }

  socklen_t len = sizeof(addr);
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) < 0) {
    close(fd);
    return 0;
  }

  int port = ntohs(addr.sin_port);
  close(fd);
  return port;
}

// =============================================================================
// Test Handle implementation
// =============================================================================
class TestHandle : public Handle {
 public:
  bool ran = false;
  void run() override { ran = true; }
  void stop() override {}
};

// =============================================================================
// Enhanced test handle for cancellation testing
// =============================================================================
class TrackedHandle : public Handle {
 public:
  int run_count = 0;
  int stop_count = 0;
  bool self_cancel = false;  // If true, will cancel itself during run()

  void run() override {
    ++run_count;
    if (self_cancel) {
      IO::Get().Cancel(*this);
    }
  }

  void stop() override {
    ++stop_count;
  }

  void reset() {
    run_count = 0;
    stop_count = 0;
    self_cancel = false;
  }
};

// =============================================================================
// defer.h tests
// =============================================================================
void TestDeferBasic() {
  printf("[TEST] DeferBasic\n");

  bool executed = false;
  {
    defer { executed = true; };
    REQUIRE(!executed);
  }
  REQUIRE(executed);
  g_test_pass++;
}

void TestDeferException() {
  printf("[TEST] DeferException\n");

  bool cleanup = false;
  try {
    defer { cleanup = true; };
    throw std::runtime_error("test");
  } catch (...) {
  }
  REQUIRE(cleanup);
  g_test_pass++;
}

// =============================================================================
// result.h tests
// =============================================================================
void TestResultValue() {
  printf("[TEST] ResultValue\n");

  Result<int> result;
  REQUIRE(!result.has_value());
  result.set_value(42);
  REQUIRE(result.has_value());
  REQUIRE(result.result() == 42);
  g_test_pass++;
}

void TestResultException() {
  printf("[TEST] ResultException\n");

  Result<int> result;
  result.set_exception(std::make_exception_ptr(std::runtime_error("error")));
  REQUIRE_THROWS(result.result());
  g_test_pass++;
}

void TestResultVoid() {
  printf("[TEST] ResultVoid\n");

  Result<void> result;
  REQUIRE(!result.has_value());
  result.return_void();
  REQUIRE(result.has_value());
  REQUIRE_NOTHROW(result.result());
  g_test_pass++;
}

void TestResultMoveSemantics() {
  printf("[TEST] ResultMoveSemantics\n");

  Counted::Reset();
  {
    Result<Counted> res;
    res.set_value(Counted{});
    REQUIRE(Counted::default_ctor == 1);
    REQUIRE(Counted::move_ctor == 1);
    auto c = std::move(res).result();
    REQUIRE(Counted::move_ctor == 2);
  }
  REQUIRE(Counted::Alive() == 0);
  g_test_pass++;
}

// =============================================================================
// handle.h tests
// =============================================================================
void TestHandleState() {
  printf("[TEST] HandleState\n");

  TestHandle h;
  REQUIRE(h.GetState() == Handle::kUnschedule);
  h.SetState(Handle::kScheduled);
  REQUIRE(h.GetState() == Handle::kScheduled);
  h.SetState(Handle::kSuspend);
  REQUIRE(h.GetState() == Handle::kSuspend);
  g_test_pass++;
}

void TestHandleUniqueId() {
  printf("[TEST] HandleUniqueId\n");

  TestHandle h1, h2, h3;
  REQUIRE(h1.GetId() != h2.GetId());
  REQUIRE(h2.GetId() != h3.GetId());
  g_test_pass++;
}

// =============================================================================
// event.h tests
// =============================================================================
void TestEventStructure() {
  printf("[TEST] EventStructure\n");

  Event e{};
  e.fd = 42;
  e.flags = EPOLLIN | EPOLLOUT;
  TestHandle h;
  e.handle = &h;

  REQUIRE(e.fd == 42);
  REQUIRE(e.flags == (EPOLLIN | EPOLLOUT));
  REQUIRE(e.handle == &h);
  g_test_pass++;
}

// =============================================================================
// selector.h tests
// =============================================================================
void TestSelectorBasic() {
  printf("[TEST] SelectorBasic\n");

  auto sel = std::make_unique<Selector>();
  REQUIRE(sel->Stopped());

  int pfd[2];
  REQUIRE(pipe(pfd) == 0);
  fcntl(pfd[0], F_SETFL, O_NONBLOCK);

  TestHandle h;
  Event e{};
  e.fd = pfd[0];
  e.flags = EPOLLIN;
  e.handle = &h;

  sel->Join(e);
  REQUIRE(!sel->Stopped());

  [[maybe_unused]] ssize_t w = write(pfd[1], "x", 1);

  auto events = sel->Select(std::chrono::milliseconds(100));
  REQUIRE(events.size() == 1);

  sel->Quit(e);
  REQUIRE(sel->Stopped());

  close(pfd[0]);
  close(pfd[1]);

  g_test_pass++;
}

// =============================================================================
// coro.h tests
// =============================================================================
Coro<int> Square(int x) { co_return x * x; }

template <size_t N>
Coro<> CoroDepthN(std::vector<int>& result) {
  result.push_back(N);
  if constexpr (N > 0) {
    co_await CoroDepthN<N - 1>(result);
    result.push_back(N * 10);
  }
}

void TestCoroSimpleAwait() {
  printf("[TEST] CoroSimpleAwait\n");

  std::vector<int> result;
  Run(CoroDepthN<0>(result));
  REQUIRE(result.size() == 1);
  REQUIRE(result[0] == 0);
  g_test_pass++;
}

void TestCoroNested() {
  printf("[TEST] CoroNested\n");

  std::vector<int> result;
  Run(CoroDepthN<2>(result));
  std::vector<int> expected{2, 1, 0, 10, 20};
  REQUIRE(result == expected);
  g_test_pass++;
}

void TestCoroResultValue() {
  printf("[TEST] CoroResultValue\n");

  auto squareSum = [](int x, int y) -> Coro<int> {
    auto x2 = co_await Square(x);
    auto y2 = co_await Square(y);
    co_return x2 + y2;
  };
  REQUIRE(Run(squareSum(3, 4)) == 25);
  g_test_pass++;
}

void TestCoroFibonacci() {
  printf("[TEST] CoroFibonacci\n");

  std::function<Coro<size_t>(size_t)> fib = [&](size_t n) -> Coro<size_t> {
    if (n <= 1) co_return n;
    co_return co_await fib(n - 1) + co_await fib(n - 2);
  };

  REQUIRE(Run(fib(0)) == 0);
  REQUIRE(Run(fib(1)) == 1);
  REQUIRE(Run(fib(2)) == 1);
  REQUIRE(Run(fib(12)) == 144);
  g_test_pass++;
}

void TestCoroForLoop() {
  printf("[TEST] CoroForLoop\n");

  auto seq = [](int n) -> Coro<int> {
    int result = 1, sign = -1;
    for (int i = 2; i <= n; ++i) {
      result += (co_await Square(i)) * sign;
      sign *= -1;
    }
    co_return result;
  };

  REQUIRE(Run(seq(1)) == 1);
  REQUIRE(Run(seq(10)) == -55);
  REQUIRE(Run(seq(100)) == -5050);
  g_test_pass++;
}

void TestCoroException() {
  printf("[TEST] CoroException\n");

  auto throwing = []() -> Coro<int> {
    throw std::runtime_error("error");
    co_return 0;
  };
  REQUIRE_THROWS(Run(throwing()));
  g_test_pass++;
}

void TestCoroMoveSemantics() {
  printf("[TEST] CoroMoveSemantics\n");

  Counted::Reset();
  auto build = []() -> Coro<Counted> { co_return Counted{}; };

  auto c = Run(build());
  REQUIRE(Counted::Alive() >= 1);
  g_test_pass++;
}

// =============================================================================
// future.h tests
// =============================================================================
void TestFutureBasic() {
  printf("[TEST] FutureBasic\n");

  auto coro = []() -> Coro<int> { co_return 123; };
  Future<Coro<int>> fut{coro()};
  REQUIRE(fut.valid());

  IO::Get().Run();
  REQUIRE(fut.done());
  REQUIRE(fut.result() == 123);
  g_test_pass++;
}

// =============================================================================
// io.h tests
// =============================================================================
void TestIOSingleton() {
  printf("[TEST] IOSingleton\n");

  auto& io1 = IO::Get();
  auto& io2 = IO::Get();
  REQUIRE(&io1 == &io2);
  g_test_pass++;
}

void TestIOTime() {
  printf("[TEST] IOTime\n");

  auto& io = IO::Get();
  auto t1 = io.Time();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  auto t2 = io.Time();
  REQUIRE(t2 >= t1);
  g_test_pass++;
}

void TestIOCancelHandle() {
  printf("[TEST] IOCancelHandle\n");

  TestHandle h;
  auto& io = IO::Get();

  io.Call(h);
  REQUIRE(h.GetState() == Handle::kScheduled);

  io.Cancel(h);
  REQUIRE(h.GetState() == Handle::kUnschedule);
  REQUIRE(io.IsCancelled(h));

  io.Run();
  REQUIRE(!h.ran);
  g_test_pass++;
}

void TestHandleCancelMethod() {
  printf("[TEST] HandleCancelMethod\n");

  TestHandle h;
  auto& io = IO::Get();

  io.Call(h);
  h.cancel();
  REQUIRE(h.GetState() == Handle::kUnschedule);

  io.Run();
  REQUIRE(!h.ran);
  g_test_pass++;
}

// =============================================================================
// Comprehensive IO Cancellation Tests
// =============================================================================

/**
 * Test: Cancel a delayed task before its scheduled time
 * Verifies that run() is never called but stop() is called exactly once
 */
void TestCancelDelayedTaskBeforeTime() {
  printf("[TEST] CancelDelayedTaskBeforeTime\n");

  TrackedHandle h;
  auto& io = IO::Get();

  // Schedule for 10ms in the future (short delay)
  io.Call(std::chrono::milliseconds(10), h);
  REQUIRE(h.GetState() == Handle::kScheduled);

  // Cancel immediately (before the scheduled time)
  io.Cancel(h);
  REQUIRE(h.GetState() == Handle::kUnschedule);
  REQUIRE(io.IsCancelled(h));

  // Wait for the scheduled time to pass
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  // Now process - the task should be processed from schedule queue
  io.Runone();
  io.Cancel();

  REQUIRE(h.run_count == 0);
  REQUIRE(h.stop_count == 1);
  REQUIRE(!io.IsCancelled(h));  // Should be cleared after processing
  g_test_pass++;
}

/**
 * Test: Cancel a handle already in the ready queue
 * Verifies it does not execute and is stopped correctly
 */
void TestCancelHandleInReadyQueue() {
  printf("[TEST] CancelHandleInReadyQueue\n");

  TrackedHandle h;
  auto& io = IO::Get();

  // Add to ready queue (immediate execution)
  io.Call(h);
  REQUIRE(h.GetState() == Handle::kScheduled);

  // Cancel before processing
  io.Cancel(h);
  REQUIRE(h.GetState() == Handle::kUnschedule);
  REQUIRE(io.IsCancelled(h));

  // Process the queues
  io.Runone();
  io.Cancel();

  REQUIRE(h.run_count == 0);
  REQUIRE(h.stop_count == 1);
  g_test_pass++;
}

/**
 * Test: Cancel the same handle multiple times
 * Ensures no crashes, double-stopping, or inconsistent state
 */
void TestCancelHandleMultipleTimes() {
  printf("[TEST] CancelHandleMultipleTimes\n");

  TrackedHandle h;
  auto& io = IO::Get();

  io.Call(h);
  REQUIRE(h.GetState() == Handle::kScheduled);

  // Cancel multiple times
  io.Cancel(h);
  io.Cancel(h);
  io.Cancel(h);

  REQUIRE(h.GetState() == Handle::kUnschedule);

  io.Runone();
  io.Cancel();

  REQUIRE(h.run_count == 0);
  REQUIRE(h.stop_count == 1);  // Should only be stopped once
  g_test_pass++;
}

/**
 * Test: Cancel a handle that has already been executed
 * Should be a no-op or behave safely
 */
void TestCancelAlreadyExecutedHandle() {
  printf("[TEST] CancelAlreadyExecutedHandle\n");

  TrackedHandle h;
  auto& io = IO::Get();

  io.Call(h);
  io.Runone();

  REQUIRE(h.run_count == 1);
  REQUIRE(h.GetState() == Handle::kUnschedule);

  // Now try to cancel the already-executed handle
  io.Cancel(h);

  // Should be a no-op since handle is already unscheduled
  REQUIRE(!io.IsCancelled(h));

  io.Cancel();
  REQUIRE(h.stop_count == 0);  // stop() should not be called for already-executed handles
  g_test_pass++;
}

/**
 * Test: Cancel a handle that was never scheduled
 * Should be a no-op
 */
void TestCancelNeverScheduledHandle() {
  printf("[TEST] CancelNeverScheduledHandle\n");

  TrackedHandle h;
  auto& io = IO::Get();

  // Cancel without scheduling
  io.Cancel(h);

  REQUIRE(h.GetState() == Handle::kUnschedule);
  REQUIRE(!io.IsCancelled(h));  // Should not be in cancelled set since it wasn't scheduled

  io.Cancel();
  REQUIRE(h.run_count == 0);
  REQUIRE(h.stop_count == 0);
  g_test_pass++;
}

/**
 * Test: Self-cancellation (handle cancels itself during run)
 * Note: This tests what happens if a handle tries to cancel itself during execution.
 * The cancellation should be a no-op since the state is already kUnschedule.
 */
void TestHandleSelfCancel() {
  printf("[TEST] HandleSelfCancel\n");

  TrackedHandle h;
  h.self_cancel = true;
  auto& io = IO::Get();

  io.Call(h);
  io.Runone();

  REQUIRE(h.run_count == 1);
  // Self-cancellation during run() should be a no-op
  // because the handle's state is set to kUnschedule before run() is called
  REQUIRE(!io.IsCancelled(h));

  io.Cancel();
  REQUIRE(h.stop_count == 0);
  g_test_pass++;
}

/**
 * Test: Mix of cancelled and non-cancelled delayed handles
 */
void TestMixedDelayedHandles() {
  printf("[TEST] MixedDelayedHandles\n");

  TrackedHandle h1, h2, h3;
  auto& io = IO::Get();

  // Schedule all with same short delay
  io.Call(std::chrono::milliseconds(1), h1);
  io.Call(std::chrono::milliseconds(1), h2);
  io.Call(std::chrono::milliseconds(1), h3);

  // Cancel only h2
  io.Cancel(h2);

  // Wait for the delay
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Note: Runone() processes all due scheduled tasks and all ready tasks in one call
  io.Runone();
  io.Cancel();

  REQUIRE(h1.run_count == 1);
  REQUIRE(h1.stop_count == 0);
  REQUIRE(h2.run_count == 0);
  REQUIRE(h2.stop_count == 1);
  REQUIRE(h3.run_count == 1);
  REQUIRE(h3.stop_count == 0);
  g_test_pass++;
}

/**
 * Stress test: Schedule many handles, cancel a subset in different phases
 */
void TestStressCancellation() {
  printf("[TEST] StressCancellation\n");

  constexpr int kNumHandles = 100;
  std::vector<TrackedHandle> handles(kNumHandles);
  auto& io = IO::Get();

  // Schedule half for immediate, half for delayed execution
  for (int i = 0; i < kNumHandles; ++i) {
    if (i % 2 == 0) {
      io.Call(handles[i]);
    } else {
      io.Call(std::chrono::milliseconds(1), handles[i]);
    }
  }

  // Cancel every third handle
  for (int i = 0; i < kNumHandles; i += 3) {
    io.Cancel(handles[i]);
  }

  // Wait for delayed tasks to be due
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Process - Note: Runone() processes all due scheduled and all ready tasks
  io.Runone();
  io.Cancel();

  // Verify results
  int total_run = 0;
  int total_stop = 0;
  for (int i = 0; i < kNumHandles; ++i) {
    if (i % 3 == 0) {
      // These were cancelled
      REQUIRE(handles[i].run_count == 0);
      REQUIRE(handles[i].stop_count == 1);
      total_stop++;
    } else {
      // These should have run
      REQUIRE(handles[i].run_count == 1);
      REQUIRE(handles[i].stop_count == 0);
      total_run++;
    }
  }

  // Sanity check the counts
  int expected_cancelled = (kNumHandles + 2) / 3;  // Ceiling division
  REQUIRE(total_stop == expected_cancelled);
  REQUIRE(total_run == kNumHandles - expected_cancelled);

  g_test_pass++;
}

/**
 * Test: Cancel during different phases (before schedule, after schedule, before run)
 */
void TestCancelDifferentPhases() {
  printf("[TEST] CancelDifferentPhases\n");

  TrackedHandle h1, h2, h3, h4;
  auto& io = IO::Get();

  // h1: Cancel before scheduling (should be no-op)
  io.Cancel(h1);
  io.Call(h1);  // Schedule after cancel attempt - should work normally

  // h2: Schedule then cancel
  io.Call(h2);
  io.Cancel(h2);

  // h3: Schedule delayed then cancel immediately (use short delay)
  io.Call(std::chrono::milliseconds(10), h3);
  io.Cancel(h3);

  // h4: Normal execution without cancellation
  io.Call(h4);

  // Wait for h3's scheduled time to pass
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  // Note: Runone() processes all due scheduled tasks and all ready tasks in one call
  io.Runone();
  io.Cancel();

  // h1 was "cancelled" before scheduling, then scheduled - should run
  REQUIRE(h1.run_count == 1);
  REQUIRE(h1.stop_count == 0);

  // h2 was scheduled then cancelled - should not run, should stop
  REQUIRE(h2.run_count == 0);
  REQUIRE(h2.stop_count == 1);

  // h3 was delayed-scheduled then cancelled - should not run, should stop
  REQUIRE(h3.run_count == 0);
  REQUIRE(h3.stop_count == 1);

  // h4 normal execution
  REQUIRE(h4.run_count == 1);
  REQUIRE(h4.stop_count == 0);

  g_test_pass++;
}

/**
 * Test: IsCancelled returns false after handle is processed
 */
void TestIsCancelledAfterProcessing() {
  printf("[TEST] IsCancelledAfterProcessing\n");

  TrackedHandle h;
  auto& io = IO::Get();

  io.Call(h);
  io.Cancel(h);

  REQUIRE(io.IsCancelled(h));

  io.Runone();
  io.Cancel();

  REQUIRE(!io.IsCancelled(h));  // Should be cleared after processing
  g_test_pass++;
}

// =============================================================================
// client.h tests
// =============================================================================
void TestClientInvalidAddress() {
  printf("[TEST] ClientInvalidAddress\n");

  REQUIRE_THROWS(Run([]() -> Coro<> {
    Client c("256.256.256.256", 80);
    Stream s = co_await c.Connect();
  }()));
  g_test_pass++;
}

void TestClientInvalidHostname() {
  printf("[TEST] ClientInvalidHostname\n");

  REQUIRE_THROWS(Run([]() -> Coro<> {
    Client c("invalid.host.local", 80);
    Stream s = co_await c.Connect();
  }()));
  g_test_pass++;
}

// =============================================================================
// Coroutine-based Server/Client/Stream Tests
// These tests exercise the async IO primitives using the coroutine framework
// with epoll (Selector) and IO scheduler integration.
// =============================================================================

/**
 * Test: Client connect using coroutine framework with raw socket server
 * Verifies that Client.Connect() and Stream work correctly with epoll-based IO
 *
 * This test exercises:
 * - Client class (client.h)
 * - Stream class (stream.h)
 * - Coroutine framework (coro.h)
 * - IO scheduler (io.h)
 * - Selector/epoll integration (selector.h)
 */
void TestCoroClientConnect() {
  printf("[TEST] CoroClientConnect\n");

  int port = GetPort();
  REQUIRE(port > 0);

  // Start a simple server using raw sockets (to test client independently)
  std::atomic<bool> serverReady{false};
  std::atomic<bool> accepted{false};
  std::atomic<bool> clientDone{false};

  std::thread serverThread([&]() {
    int listenFd = socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd < 0) return;

    int opt = 1;
    setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);

    if (bind(listenFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
      close(listenFd);
      return;
    }

    listen(listenFd, 1);
    serverReady = true;

    int clientFd = accept(listenFd, nullptr, nullptr);
    if (clientFd >= 0) {
      accepted = true;
      // Echo any data
      char buf[256];
      ssize_t n = recv(clientFd, buf, sizeof(buf), 0);
      if (n > 0) send(clientFd, buf, n, 0);
      // Wait for client to finish reading before closing
      while (!clientDone) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      close(clientFd);
    }
    close(listenFd);
  });

  // Wait for server
  while (!serverReady) std::this_thread::sleep_for(std::chrono::milliseconds(5));

  // Use coroutine-based client with epoll-based async IO
  std::string result;
  Run([&]() -> Coro<> {
    auto client = Client("127.0.0.1", port);
    auto stream = co_await client.Connect();

    std::string msg("connect test");
    co_await stream.Write(msg.data(), msg.size());

    char buf[256];
    size_t size = co_await stream.Read(buf, sizeof(buf));
    result = std::string(buf, size);
    clientDone = true;
  }());

  serverThread.join();

  REQUIRE(accepted.load());
  REQUIRE(result == "connect test");
  g_test_pass++;
}

/**
 * Test: Stream data copy pattern
 * Verifies that data from Read() can be safely copied to a string
 *
 * This test exercises:
 * - Stream::Read() with async IO
 * - Data lifetime and buffer management
 * - Coroutine suspension and resumption via epoll
 */
void TestCoroStreamDataCopy() {
  printf("[TEST] CoroStreamDataCopy\n");

  int port = GetPort();
  REQUIRE(port > 0);

  std::atomic<bool> serverReady{false};
  std::atomic<bool> clientDone{false};

  std::thread serverThread([&]() {
    int listenFd = socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd < 0) return;

    int opt = 1;
    setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);

    if (bind(listenFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
      close(listenFd);
      return;
    }

    listen(listenFd, 1);
    serverReady = true;

    int clientFd = accept(listenFd, nullptr, nullptr);
    if (clientFd >= 0) {
      // Send test data
      const char* test_data = "stream data copy test with longer message";
      send(clientFd, test_data, strlen(test_data), 0);
      // Wait for client to finish
      while (!clientDone) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      close(clientFd);
    }
    close(listenFd);
  });

  while (!serverReady) std::this_thread::sleep_for(std::chrono::milliseconds(5));

  std::string received;
  Run([&]() -> Coro<> {
    auto client = Client("127.0.0.1", port);
    auto stream = co_await client.Connect();

    char buf[256];
    size_t size = co_await stream.Read(buf, sizeof(buf));
    // Copy data immediately to string - this is the safe pattern
    received = std::string(buf, size);
    clientDone = true;
  }());

  serverThread.join();

  REQUIRE(received == "stream data copy test with longer message");
  g_test_pass++;
}

/**
 * Test: Echo server and client (modeled after asyncio project)
 * Simple echo test that:
 * - Server reads data and echoes it back
 * - Client sends a message and receives the echo
 * - Server is cancelled after client completes
 *
 * This follows the pattern from asyncio's task_test.cpp "echo server & client" test.
 */
void TestCoroServerClientEcho() {
  printf("[TEST] CoroServerClientEcho\n");

  int port = GetPort();
  REQUIRE(port > 0);

  bool is_called = false;
  std::string received_message;
  constexpr std::string_view message = "hello world!";

  Run([&]() -> Coro<> {
    // Echo handler - reads data and echoes it back
    auto handle_echo = [&](Stream stream) -> Coro<> {
      char buf[256];
      size_t size = co_await stream.Read(buf, sizeof(buf));
      if (size > 0) {
        // Copy data and echo back (like asyncio pattern)
        co_await stream.Write(buf, size);
      }
    };

    // Echo server coroutine
    auto echo_server = [&]() -> Coro<> {
      auto server = Server("127.0.0.1", port, handle_echo);
      server.Start();
      co_await server.Wait();
    };

    // Echo client coroutine
    auto echo_client = [&]() -> Coro<> {
      auto client = Client("127.0.0.1", port);
      auto stream = co_await client.Connect();

      // Send message
      co_await stream.Write(message.data(), message.size());

      // Read echo
      char buf[256];
      size_t size = co_await stream.Read(buf, sizeof(buf));
      // Store result for verification outside coroutine
      received_message = std::string(buf, size);
      is_called = true;
    };

    // Start server as background task
    auto srv = Future(echo_server());

    // Run client
    co_await echo_client();

    // Cancel server
    srv.Cancel();
  }());

  REQUIRE(is_called);
  REQUIRE(received_message == message);
  g_test_pass++;
}

void TestCoroServerMultiClientEcho() {
  printf("[TEST] CoroServerMultiClientEcho\n");

  int port = GetPort();
  REQUIRE(port > 0);

  constexpr int kNumClients = 64;
  std::array<bool, kNumClients> is_called{};
  std::array<std::string, kNumClients> received_messages;
  constexpr std::string_view message = "hello world!";

  Run([&]() -> Coro<> {
    auto handle_echo = [&](Stream stream) -> Coro<> {
      char buf[256];
      size_t size = co_await stream.Read(buf, sizeof(buf));
      if (size > 0) {
        co_await stream.Write(buf, size);
      }
    };

    auto echo_server = [&]() -> Coro<> {
      auto server = Server("127.0.0.1", port, handle_echo);
      server.Start();
      co_await server.Wait();
    };

    auto echo_client = [&](int id) -> Coro<> {
      auto client = Client("127.0.0.1", port);
      auto stream = co_await client.Connect();
      co_await stream.Write(message.data(), message.size());
      char buf[256];
      size_t size = co_await stream.Read(buf, sizeof(buf));
      received_messages[id] = std::string(buf, size);
      is_called[id] = true;
    };

    auto srv = Future(echo_server());

    for (int i = 0; i < kNumClients; ++i) {
      co_await echo_client(i);
    }

    srv.Cancel();
  }());

  for (int i = 0; i < kNumClients; ++i) {
    REQUIRE(is_called[i]);
    REQUIRE(received_messages[i] == message);
  }
  g_test_pass++;
}

/**
 * @brief Test large data transfer (10MB) following asyncio pattern
 *
 * This test verifies that the Stream can handle large data transfers
 * reliably. The pattern follows asyncio's echo test but with 10MB of data.
 */
void TestCoroLargeDataTransfer() {
  printf("[TEST] CoroLargeDataTransfer\n");

  int port = GetPort();
  REQUIRE(port > 0);

  constexpr size_t kTotalSize = 10 * 1024 * 1024;  // 10 MB
  constexpr size_t kChunkSize = 64 * 1024;         // 64 KB chunks
  printf("  Transferring %zu bytes (%.1f MB)...\n", kTotalSize, kTotalSize / (1024.0 * 1024.0));

  // Create a pattern buffer
  std::vector<char> send_buffer(kChunkSize);
  for (size_t i = 0; i < kChunkSize; ++i) {
    send_buffer[i] = static_cast<char>(i % 256);
  }

  size_t server_bytes_received = 0;
  size_t client_bytes_received = 0;
  bool test_completed = false;

  Run([&]() -> Coro<> {
    // Echo handler - receives data and echoes it back in chunks
    auto handle_echo = [&](Stream stream) -> Coro<> {
      std::vector<char> received_data;
      received_data.reserve(kTotalSize);
      char buf[kChunkSize];

      // Read all data until we have kTotalSize bytes
      while (received_data.size() < kTotalSize) {
        size_t size = co_await stream.Read(buf, sizeof(buf));
        if (size == 0) break;  // EOF
        received_data.insert(received_data.end(), buf, buf + size);
      }

      server_bytes_received = received_data.size();

      // Echo all received data back in chunks
      size_t sent = 0;
      while (sent < received_data.size()) {
        size_t to_send = std::min(kChunkSize, received_data.size() - sent);
        size_t written = co_await stream.Write(received_data.data() + sent, to_send);
        if (written == 0) break;
        sent += written;
      }
    };

    // Echo server coroutine
    auto echo_server = [&]() -> Coro<> {
      auto server = Server("127.0.0.1", port, handle_echo);
      server.Start();
      co_await server.Wait();
    };

    // Echo client coroutine
    auto echo_client = [&]() -> Coro<> {
      auto client = Client("127.0.0.1", port);
      auto stream = co_await client.Connect();

      // Send kTotalSize bytes in chunks
      size_t total_sent = 0;
      while (total_sent < kTotalSize) {
        size_t to_send = std::min(kChunkSize, kTotalSize - total_sent);
        size_t written = co_await stream.Write(send_buffer.data(), to_send);
        if (written == 0) break;
        total_sent += written;
      }

      // Read all echoed data
      std::vector<char> received_data;
      received_data.reserve(kTotalSize);
      char buf[kChunkSize];
      while (received_data.size() < kTotalSize) {
        size_t size = co_await stream.Read(buf, sizeof(buf));
        if (size == 0) break;  // EOF
        received_data.insert(received_data.end(), buf, buf + size);
      }

      client_bytes_received = received_data.size();

      // Verify data integrity
      bool integrity_ok = (received_data.size() == kTotalSize);
      if (integrity_ok) {
        for (size_t i = 0; i < kTotalSize; ++i) {
          if (received_data[i] != static_cast<char>(i % 256)) {
            integrity_ok = false;
            break;
          }
        }
      }

      test_completed = integrity_ok;
    };

    // Start server as background task
    auto srv = Future(echo_server());

    // Run client
    co_await echo_client();

    // Cancel server
    srv.Cancel();
  }());

  printf("  Server received: %zu bytes\n", server_bytes_received);
  printf("  Client received: %zu bytes\n", client_bytes_received);

  REQUIRE(server_bytes_received == kTotalSize);
  REQUIRE(client_bytes_received == kTotalSize);
  REQUIRE(test_completed);
  g_test_pass++;
}

// =============================================================================
// Edge Case Tests
// =============================================================================

/**
 * @brief Test single byte data transfer
 * Edge case: minimum non-empty data transfer
 */
void TestCoroSingleByteTransfer() {
  printf("[TEST] CoroSingleByteTransfer\n");

  int port = GetPort();
  REQUIRE(port > 0);

  char received_byte = 0;
  bool test_completed = false;

  Run([&]() -> Coro<> {
    auto handle_echo = [&](Stream stream) -> Coro<> {
      char buf[1];
      size_t size = co_await stream.Read(buf, sizeof(buf));
      if (size > 0) {
        co_await stream.Write(buf, size);
      }
    };

    auto echo_server = [&]() -> Coro<> {
      auto server = Server("127.0.0.1", port, handle_echo);
      server.Start();
      co_await server.Wait();
    };

    auto echo_client = [&]() -> Coro<> {
      auto client = Client("127.0.0.1", port);
      auto stream = co_await client.Connect();

      char send_byte = 'X';
      co_await stream.Write(&send_byte, 1);

      char buf[1];
      size_t size = co_await stream.Read(buf, sizeof(buf));
      if (size == 1) {
        received_byte = buf[0];
        test_completed = true;
      }
    };

    auto srv = Future(echo_server());
    co_await echo_client();
    srv.Cancel();
  }());

  REQUIRE(test_completed);
  REQUIRE(received_byte == 'X');
  g_test_pass++;
}

/**
 * @brief Test binary data with all byte values (0-255)
 * Edge case: ensures no byte value corruption
 */
void TestCoroBinaryDataTransfer() {
  printf("[TEST] CoroBinaryDataTransfer\n");

  int port = GetPort();
  REQUIRE(port > 0);

  constexpr size_t kDataSize = 256;  // All byte values 0-255
  std::vector<char> received_data;
  bool test_completed = false;

  Run([&]() -> Coro<> {
    auto handle_echo = [&](Stream stream) -> Coro<> {
      char buf[512];
      size_t total = 0;
      while (total < kDataSize) {
        size_t size = co_await stream.Read(buf + total, sizeof(buf) - total);
        if (size == 0) break;
        total += size;
      }
      co_await stream.Write(buf, total);
    };

    auto echo_server = [&]() -> Coro<> {
      auto server = Server("127.0.0.1", port, handle_echo);
      server.Start();
      co_await server.Wait();
    };

    auto echo_client = [&]() -> Coro<> {
      auto client = Client("127.0.0.1", port);
      auto stream = co_await client.Connect();

      // Create buffer with all byte values 0-255
      char send_buf[kDataSize];
      for (size_t i = 0; i < kDataSize; ++i) {
        send_buf[i] = static_cast<char>(i);
      }
      co_await stream.Write(send_buf, kDataSize);

      char buf[512];
      size_t total = 0;
      while (total < kDataSize) {
        size_t size = co_await stream.Read(buf + total, sizeof(buf) - total);
        if (size == 0) break;
        total += size;
      }

      received_data.assign(buf, buf + total);
      test_completed = (total == kDataSize);
    };

    auto srv = Future(echo_server());
    co_await echo_client();
    srv.Cancel();
  }());

  REQUIRE(test_completed);
  REQUIRE(received_data.size() == kDataSize);
  for (size_t i = 0; i < kDataSize; ++i) {
    REQUIRE(received_data[i] == static_cast<char>(i));
  }
  g_test_pass++;
}

/**
 * @brief Test rapid connect/disconnect cycles
 * Edge case: connection churn stress test
 */
void TestCoroRapidConnectDisconnect() {
  printf("[TEST] CoroRapidConnectDisconnect\n");

  int port = GetPort();
  REQUIRE(port > 0);

  constexpr int kNumCycles = 50;
  int successful_cycles = 0;

  Run([&]() -> Coro<> {
    auto handle_echo = [&](Stream stream) -> Coro<> {
      char buf[16];
      size_t size = co_await stream.Read(buf, sizeof(buf));
      if (size > 0) {
        co_await stream.Write(buf, size);
      }
    };

    auto echo_server = [&]() -> Coro<> {
      auto server = Server("127.0.0.1", port, handle_echo);
      server.Start();
      co_await server.Wait();
    };

    auto srv = Future(echo_server());

    for (int i = 0; i < kNumCycles; ++i) {
      auto client = Client("127.0.0.1", port);
      auto stream = co_await client.Connect();

      char msg = 'A' + (i % 26);
      co_await stream.Write(&msg, 1);

      char buf[1];
      size_t size = co_await stream.Read(buf, sizeof(buf));
      if (size == 1 && buf[0] == msg) {
        successful_cycles++;
      }
      // Stream goes out of scope, connection closes
    }

    srv.Cancel();
  }());

  printf("  Successful cycles: %d/%d\n", successful_cycles, kNumCycles);
  REQUIRE(successful_cycles == kNumCycles);
  g_test_pass++;
}

// =============================================================================
// Stress Tests
// =============================================================================

/**
 * @brief Test very large data transfer (128MB)
 * Stress test: large data transfer to stress buffer management
 */
void TestCoroVeryLargeDataTransfer() {
  printf("[TEST] CoroVeryLargeDataTransfer\n");

  int port = GetPort();
  REQUIRE(port > 0);

  constexpr size_t kTotalSize = 128 * 1024 * 1024;  // 128 MB
  constexpr size_t kChunkSize = 256 * 1024;          // 256 KB chunks
  printf("  Transferring %zu bytes (%.1f MB)...\n", kTotalSize, kTotalSize / (1024.0 * 1024.0));

  // Create a pattern buffer
  std::vector<char> send_buffer(kChunkSize);
  for (size_t i = 0; i < kChunkSize; ++i) {
    send_buffer[i] = static_cast<char>(i % 256);
  }

  size_t server_bytes_received = 0;
  size_t client_bytes_received = 0;
  bool test_completed = false;

  Run([&]() -> Coro<> {
    auto handle_echo = [&](Stream stream) -> Coro<> {
      std::vector<char> received_data;
      received_data.reserve(kTotalSize);
      char buf[kChunkSize];

      while (received_data.size() < kTotalSize) {
        size_t size = co_await stream.Read(buf, sizeof(buf));
        if (size == 0) break;
        received_data.insert(received_data.end(), buf, buf + size);
      }

      server_bytes_received = received_data.size();

      // Echo all received data back
      size_t sent = 0;
      while (sent < received_data.size()) {
        size_t to_send = std::min(kChunkSize, received_data.size() - sent);
        size_t written = co_await stream.Write(received_data.data() + sent, to_send);
        if (written == 0) break;
        sent += written;
      }
    };

    auto echo_server = [&]() -> Coro<> {
      auto server = Server("127.0.0.1", port, handle_echo);
      server.Start();
      co_await server.Wait();
    };

    auto echo_client = [&]() -> Coro<> {
      auto client = Client("127.0.0.1", port);
      auto stream = co_await client.Connect();

      size_t total_sent = 0;
      while (total_sent < kTotalSize) {
        size_t to_send = std::min(kChunkSize, kTotalSize - total_sent);
        size_t written = co_await stream.Write(send_buffer.data(), to_send);
        if (written == 0) break;
        total_sent += written;
      }

      std::vector<char> received_data;
      received_data.reserve(kTotalSize);
      char buf[kChunkSize];
      while (received_data.size() < kTotalSize) {
        size_t size = co_await stream.Read(buf, sizeof(buf));
        if (size == 0) break;
        received_data.insert(received_data.end(), buf, buf + size);
      }

      client_bytes_received = received_data.size();

      // Verify data integrity (sample check for performance)
      bool integrity_ok = (received_data.size() == kTotalSize);
      if (integrity_ok) {
        // Check every 1MB
        for (size_t i = 0; i < kTotalSize && integrity_ok; i += 1024 * 1024) {
          if (received_data[i] != static_cast<char>(i % 256)) {
            integrity_ok = false;
          }
        }
      }

      test_completed = integrity_ok;
    };

    auto srv = Future(echo_server());
    co_await echo_client();
    srv.Cancel();
  }());

  printf("  Server received: %zu bytes (%.1f MB)\n", server_bytes_received, server_bytes_received / (1024.0 * 1024.0));
  printf("  Client received: %zu bytes (%.1f MB)\n", client_bytes_received, client_bytes_received / (1024.0 * 1024.0));

  REQUIRE(server_bytes_received == kTotalSize);
  REQUIRE(client_bytes_received == kTotalSize);
  REQUIRE(test_completed);
  g_test_pass++;
}

/**
 * @brief Test concurrent multiple clients
 * Stress test: multiple clients connecting and transferring simultaneously
 */
void TestCoroConcurrentMultiClient() {
  printf("[TEST] CoroConcurrentMultiClient\n");

  int port = GetPort();
  REQUIRE(port > 0);

  constexpr int kNumClients = 32;
  std::array<bool, kNumClients> is_called{};
  std::array<std::string, kNumClients> received_messages;
  std::atomic<int> completed_clients{0};

  Run([&]() -> Coro<> {
    auto handle_echo = [&](Stream stream) -> Coro<> {
      char buf[256];
      size_t size = co_await stream.Read(buf, sizeof(buf));
      if (size > 0) {
        co_await stream.Write(buf, size);
      }
    };

    auto echo_server = [&]() -> Coro<> {
      auto server = Server("127.0.0.1", port, handle_echo);
      server.Start();
      co_await server.Wait();
    };

    auto echo_client = [&](int id) -> Coro<> {
      auto client = Client("127.0.0.1", port);
      auto stream = co_await client.Connect();

      // Each client sends a unique message
      std::string msg = "client_" + std::to_string(id);
      co_await stream.Write(msg.data(), msg.size());

      char buf[256];
      size_t size = co_await stream.Read(buf, sizeof(buf));
      received_messages[id] = std::string(buf, size);
      is_called[id] = true;
      completed_clients++;
    };

    auto srv = Future(echo_server());

    // Start all clients as background tasks (concurrent)
    std::vector<Future<Coro<>>> client_futures;
    for (int i = 0; i < kNumClients; ++i) {
      client_futures.emplace_back(echo_client(i));
    }

    // Wait for all clients to complete by awaiting each future
    for (int i = 0; i < kNumClients; ++i) {
      co_await std::move(client_futures[i]);
    }

    srv.Cancel();
  }());

  printf("  Completed clients: %d/%d\n", completed_clients.load(), kNumClients);

  for (int i = 0; i < kNumClients; ++i) {
    REQUIRE(is_called[i]);
    std::string expected = "client_" + std::to_string(i);
    REQUIRE(received_messages[i] == expected);
  }
  g_test_pass++;
}

/**
 * @brief Test high-throughput small messages
 * Stress test: many small messages in rapid succession
 */
void TestCoroHighThroughputSmallMessages() {
  printf("[TEST] CoroHighThroughputSmallMessages\n");

  int port = GetPort();
  REQUIRE(port > 0);

  constexpr int kNumMessages = 1000;
  constexpr size_t kMessageSize = 64;
  int messages_echoed = 0;

  Run([&]() -> Coro<> {
    auto handle_echo = [&](Stream stream) -> Coro<> {
      char buf[kMessageSize];
      for (int i = 0; i < kNumMessages; ++i) {
        size_t size = co_await stream.Read(buf, sizeof(buf));
        if (size == 0) break;
        size_t written = co_await stream.Write(buf, size);
        if (written == 0) break;
      }
    };

    auto echo_server = [&]() -> Coro<> {
      auto server = Server("127.0.0.1", port, handle_echo);
      server.Start();
      co_await server.Wait();
    };

    auto echo_client = [&]() -> Coro<> {
      auto client = Client("127.0.0.1", port);
      auto stream = co_await client.Connect();

      char send_buf[kMessageSize];
      char recv_buf[kMessageSize];
      std::memset(send_buf, 'A', kMessageSize);

      for (int i = 0; i < kNumMessages; ++i) {
        send_buf[0] = static_cast<char>(i % 256);  // Unique first byte
        co_await stream.Write(send_buf, kMessageSize);

        size_t total = 0;
        while (total < kMessageSize) {
          size_t size = co_await stream.Read(recv_buf + total, kMessageSize - total);
          if (size == 0) break;
          total += size;
        }

        if (total == kMessageSize && recv_buf[0] == send_buf[0]) {
          messages_echoed++;
        }
      }
    };

    auto srv = Future(echo_server());
    co_await echo_client();
    srv.Cancel();
  }());

  printf("  Messages echoed: %d/%d\n", messages_echoed, kNumMessages);
  REQUIRE(messages_echoed == kNumMessages);
  g_test_pass++;
}

/**
 * @brief Test repeated data transfers with integrity verification
 * Stress test: transfer data 256 times (more than 128) with full integrity check
 */
void TestCoroRepeatedTransfersWithIntegrity() {
  printf("[TEST] CoroRepeatedTransfersWithIntegrity\n");

  int port = GetPort();
  REQUIRE(port > 0);

  constexpr int kNumTransfers = 256;  // More than 128 as requested
  constexpr size_t kTransferSize = 4096;  // 4KB per transfer
  int successful_transfers = 0;
  int integrity_failures = 0;

  Run([&]() -> Coro<> {
    auto handle_echo = [&](Stream stream) -> Coro<> {
      char buf[kTransferSize];
      for (int i = 0; i < kNumTransfers; ++i) {
        // Read the full transfer
        size_t total_read = 0;
        while (total_read < kTransferSize) {
          size_t size = co_await stream.Read(buf + total_read, kTransferSize - total_read);
          if (size == 0) co_return;
          total_read += size;
        }
        // Echo it back
        size_t total_written = 0;
        while (total_written < kTransferSize) {
          size_t written = co_await stream.Write(buf + total_written, kTransferSize - total_written);
          if (written == 0) co_return;
          total_written += written;
        }
      }
    };

    auto echo_server = [&]() -> Coro<> {
      auto server = Server("127.0.0.1", port, handle_echo);
      server.Start();
      co_await server.Wait();
    };

    auto echo_client = [&]() -> Coro<> {
      auto client = Client("127.0.0.1", port);
      auto stream = co_await client.Connect();

      std::vector<char> send_buf(kTransferSize);
      std::vector<char> recv_buf(kTransferSize);

      for (int transfer = 0; transfer < kNumTransfers; ++transfer) {
        // Fill with unique pattern based on transfer number
        // Pattern: each byte = (transfer_num + byte_position) % 256
        for (size_t i = 0; i < kTransferSize; ++i) {
          send_buf[i] = static_cast<char>((transfer + i) % 256);
        }

        // Send the data
        size_t total_written = 0;
        while (total_written < kTransferSize) {
          size_t written = co_await stream.Write(send_buf.data() + total_written, kTransferSize - total_written);
          if (written == 0) co_return;
          total_written += written;
        }

        // Receive the echo
        size_t total_read = 0;
        while (total_read < kTransferSize) {
          size_t size = co_await stream.Read(recv_buf.data() + total_read, kTransferSize - total_read);
          if (size == 0) co_return;
          total_read += size;
        }

        // Verify data integrity - check every byte
        bool integrity_ok = true;
        for (size_t i = 0; i < kTransferSize; ++i) {
          if (recv_buf[i] != send_buf[i]) {
            integrity_ok = false;
            break;
          }
        }

        if (integrity_ok) {
          successful_transfers++;
        } else {
          integrity_failures++;
        }
      }
    };

    auto srv = Future(echo_server());
    co_await echo_client();
    srv.Cancel();
  }());

  printf("  Transfers completed: %d/%d\n", successful_transfers, kNumTransfers);
  printf("  Integrity failures: %d\n", integrity_failures);
  printf("  Total data transferred: %.1f MB\n",
         (kNumTransfers * kTransferSize * 2.0) / (1024.0 * 1024.0));

  REQUIRE(successful_transfers == kNumTransfers);
  REQUIRE(integrity_failures == 0);
  g_test_pass++;
}

// =============================================================================
// Delayed Server Start with Client Retry Test
// =============================================================================

/**
 * @brief Test delayed server start with client retry
 *
 * This test verifies that when the server starts after the client's first
 * connection attempt fails, the client can retry connecting until the server
 * is up, and then successfully exchange data with integrity verification.
 *
 * Pattern:
 * - Client starts immediately and attempts to connect (first attempt fails)
 * - Server starts only after client's first connection failure
 * - Client retries connection attempts every 200ms (up to 32 retries)
 * - Once connected, client sends a test message and receives echo
 * - Verify data integrity of the echoed message
 */
void TestCoroDelayedServerStartWithClientRetry() {
  printf("[TEST] CoroDelayedServerStartWithClientRetry\n");

  int port = GetPort();
  REQUIRE(port > 0);

  constexpr std::string_view kTestMessage = "delayed server test";
  std::string received_message;
  bool client_connected = false;
  bool test_completed = false;
  int retry_count = 0;

  Run([&]() -> Coro<> {
    // Echo handler - reads data and echoes it back
    auto handle_echo = [](Stream stream) -> Coro<> {
      char buf[256];
      size_t size = co_await stream.Read(buf, sizeof(buf));
      if (size > 0) {
        co_await stream.Write(buf, size);
      }
    };

    // Server coroutine
    auto start_server = [&]() -> Coro<> {
      auto server = Server("127.0.0.1", port, handle_echo);
      server.Start();
      co_await server.Wait();
    };

    // Client coroutine with connection retry logic
    constexpr int kMaxRetries = 32;
    constexpr auto kRetryDelay = std::chrono::milliseconds(200);
    std::optional<Future<Coro<>>> srv;

    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
      try {
        auto client = Client("127.0.0.1", port);
        auto stream = co_await client.Connect();

        // Connection successful
        client_connected = true;
        retry_count = attempt;

        // Send test message
        co_await stream.Write(kTestMessage.data(), kTestMessage.size());

        // Read echo
        char buf[256];
        size_t size = co_await stream.Read(buf, sizeof(buf));
        received_message = std::string(buf, size);
        test_completed = true;
        break;
      } catch (...) {
        // Start server after first connection failure
        if (attempt == 0) {
          srv.emplace(start_server());
        }
      }
      co_await Sleep(kRetryDelay);
    }

    // Fail if max retries exhausted without connection
    if (!client_connected) {
      SPDLOG_ERROR("  [FAIL] Max retries ({}) exhausted", kMaxRetries);
    }

    // Cancel server
    if (srv) {
      srv->Cancel();
    }
  }());

  printf("  Connection retries: %d\n", retry_count);
  printf("  Client connected: %s\n", client_connected ? "yes" : "no");

  REQUIRE(client_connected);
  REQUIRE(test_completed);
  REQUIRE(received_message == kTestMessage);
  g_test_pass++;
}

// =============================================================================
// Main
// =============================================================================
int main() {
  printf("=== IO Unit Tests ===\n\n");

  // defer.h
  TestDeferBasic();
  TestDeferException();

  // result.h
  TestResultValue();
  TestResultException();
  TestResultVoid();
  TestResultMoveSemantics();

  // handle.h
  TestHandleState();
  TestHandleUniqueId();

  // event.h
  TestEventStructure();

  // selector.h
  TestSelectorBasic();

  // coro.h
  TestCoroSimpleAwait();
  TestCoroNested();
  TestCoroResultValue();
  TestCoroFibonacci();
  TestCoroForLoop();
  TestCoroException();
  TestCoroMoveSemantics();

  // future.h
  TestFutureBasic();

  // io.h
  TestIOSingleton();
  TestIOTime();
  TestIOCancelHandle();
  TestHandleCancelMethod();

  // io.h cancellation tests
  TestCancelDelayedTaskBeforeTime();
  TestCancelHandleInReadyQueue();
  TestCancelHandleMultipleTimes();
  TestCancelAlreadyExecutedHandle();
  TestCancelNeverScheduledHandle();
  TestHandleSelfCancel();
  TestMixedDelayedHandles();
  TestStressCancellation();
  TestCancelDifferentPhases();
  TestIsCancelledAfterProcessing();

  // client.h error handling
  TestClientInvalidAddress();
  TestClientInvalidHostname();

  // Coroutine-based Client/Stream/Server tests
  // These tests exercise: client.h, server.h, stream.h, selector.h, io.h
  TestCoroClientConnect();
  TestCoroStreamDataCopy();
  TestCoroServerClientEcho();
  TestCoroServerMultiClientEcho();
  TestCoroLargeDataTransfer();

  // Edge case tests
  TestCoroSingleByteTransfer();
  TestCoroBinaryDataTransfer();
  TestCoroRapidConnectDisconnect();

  // Stress tests
  TestCoroVeryLargeDataTransfer();
  TestCoroConcurrentMultiClient();
  TestCoroHighThroughputSmallMessages();
  TestCoroRepeatedTransfersWithIntegrity();

  // Delayed server start with client retry test
  TestCoroDelayedServerStartWithClientRetry();

  printf("\n=== Summary ===\n");
  printf("PASSED: %d\n", g_test_pass);
  printf("FAILED: %d\n", g_test_fail);

  if (g_test_fail > 0) {
    printf("\n[FAILED]\n");
    return 1;
  }

  printf("\n[SUCCESS]\n");
  return 0;
}
