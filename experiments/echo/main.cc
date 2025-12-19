/**
 * @file main.cc
 * @brief Echo server/client example using coroutine-based IO primitives
 *
 * This example demonstrates a single process running an async echo server
 * and client. The server handles one echo request and the client sends
 * a message, receives the echo, and prints it.
 *
 * Key points about the implementation:
 * - Stream objects must remain alive while their coroutines are executing
 * - Data returned from stream.Read() is valid until the next Read() call
 * - Always copy data from Read() into a local string before further operations
 * - The server runs as a background task while the client connects and communicates
 * - After the client completes, fut.Cancel() is used to gracefully cancel the server
 */

#include <cstdlib>
#include <string>
#include <io/coro.h>
#include <io/runner.h>
#include <io/server.h>
#include <io/client.h>
#include <io/future.h>
#include <io/stream.h>
#include <iostream>


Coro<> Start(int argc, char *argv[]) {
  // Echo handler: reads data from the stream and writes it back
  auto handle_echo = [&](Stream stream) -> Coro<> {
    char buf[1024];
    size_t size = co_await stream.Read(buf, sizeof(buf));
    std::string msg(buf, size);
    std::cout << "handle_echo: " << msg << std::endl;
    co_await stream.Write(msg.data(), msg.size());
  };

  // Echo server: binds to localhost:8888 and handles connections
  auto echo_server = [&]() -> Coro<> {
    auto server = Server("127.0.0.1", 8888, handle_echo);
    server.Start();
    co_await server.Wait();
  };

  // Echo client: connects to the server, sends a message, and receives the echo
  auto echo_client = [&]() -> Coro<> {
    auto client = Client("127.0.0.1", 8888);
    auto stream = co_await client.Connect();

    std::string send_msg("hello world");
    co_await stream.Write(send_msg.data(), send_msg.size());

    char buf[1024];
    size_t size = co_await stream.Read(buf, sizeof(buf));
    std::string recv_msg(buf, size);
    std::cout << "echo msg: " << recv_msg << std::endl;

    // Verify the echo worked correctly
    if (recv_msg == send_msg) {
      std::cout << "Echo successful!" << std::endl;
    } else {
      std::cerr << "Echo mismatch!" << std::endl;
    }
  };

  // Start the server as a background task
  auto fut = Future(echo_server());

  // Run the client
  co_await echo_client();

  // After client completes, cancel the server coroutine
  std::cout << "Client completed, cancelling server..." << std::endl;
  fut.Cancel();
}

int main(int argc, char* argv[]) { Run(Start(argc, argv)); }
