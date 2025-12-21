# Echo

This example demonstrates how to use [C++20 coroutines](https://en.cppreference.com/w/cpp/language/coroutines)
for asynchronous I/O and is not directly tied to RDMA itself. The example in `main.cu`
demonstrates the behavior of the echo server and client using C++20 coroutine syntax.
Coroutines introduce a modern event-driven programming model that helps mitigate
*callback hell* and reduces reliance on global state management, making
asynchronous workflows clearer and more maintainable.

While coroutines provide compiler-level syntax, they do not include a standardized
runtime for scheduling or I/O polling. Developers must implement a **scheduler** and
**selector** to manage coroutine execution and event readiness.

This design is particularly relevant for **RDMA APIs**, such as those provided by
[libfabric](https://ofiwg.github.io/libfabric/), where operations are typically
asynchronous and applications often poll a **completion queue (CQ)** to check whether
RDMA send/receive operations have completed. Traditional callback-based implementations
for CQ polling can lead to complex code with many callbacks and global state. By
leveraging C++20 coroutines, these workflows can be expressed sequentially, with the
scheduler acting as the event loop and the CQ polling as the I/O selector.

In this project, we implement a lightweight **scheduler** and **RDMA selector**,
enabling developers to write coroutine-based logic using `co_await` and `co_return`,
similar to Python’s [asyncio](https://docs.python.org/3/library/asyncio.html) or
JavaScript’s [async/await](https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Asynchronous/Promises).
The Echo example demonstrates a coroutine-based TCP echo server and client, showing
how C++20 coroutines can simplify event-driven network programming. RDMA-based
examples using an event loop, such as [sendrecv](../sendrecv), are provided in
later examples.
