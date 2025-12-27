# Echo

This example demonstrates how to build asynchronous network applications using
[C++20 coroutines](https://en.cppreference.com/w/cpp/language/coroutines). The
`main.cu` file implements a TCP echo server and client that showcase coroutine-based
async I/O patterns. By using `co_await` and `co_return`, coroutines enable an
event-driven programming model that avoids *callback hell* and eliminates complex
global state management, resulting in cleaner, more maintainable asynchronous code.

## Event Loop Implementation

Unlike Python's [asyncio](https://docs.python.org/3/library/asyncio.html) or
JavaScript's [async/await](https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Asynchronous/Promises),
C++20 coroutines do not include a built-in runtime for task scheduling or I/O polling.
This project implements a lightweight event loop with two core components:

- **Scheduler**: Manages coroutine lifecycle using a priority queue and timer
- **Selector**: Polls for I/O readiness (e.g., epoll, kqueue) and resumes waiting coroutines

This architecture maps naturally to **RDMA (Remote Direct Memory Access)** programming
with libraries like [libfabric](https://ofiwg.github.io/libfabric/). In RDMA workflows,
applications poll a **completion queue (CQ)** to check operation status—a pattern
that traditionally requires complex callback chains. With this event loop, the
scheduler manages coroutine execution while CQ polling serves as the selector,
allowing RDMA operations to be written as sequential, readable code. See
[sendrecv](../sendrecv) for RDMA-based examples.

```
Event Loop Flow

                         ┌───────┐
                         │ Start │
                         └───┬───┘
                             │
                             ▼
                   ┌─────────────────────┐
              ┌───►│ Get next coroutine  │
              │    └──────────┬──────────┘
              │               │
              │               ▼
              │    ┌─────────────────────┐
              │    │  Resume coroutine   │
              │    └──────────┬──────────┘
              │               │
              │    ┌──────────┴──────────┐
              │    │                     │
              │ co_await              co_return
              │    │                     │
              │    ▼                     ▼
              │ ┌──────────────┐   ┌───────────┐
              │ │ Register I/O │   │  Complete │
              │ │ with Selector│   └───────────┘
              │ └──────┬───────┘
              │        │
              │        ▼
              │ ┌──────────────┐
              │ │  Poll I/O    │
              │ │  readiness   │
              │ └──────┬───────┘
              │        │
              │        ▼
              │ ┌──────────────┐
              └─┤ I/O ready,   │
                │ re-schedule  │
                └──────────────┘
```
