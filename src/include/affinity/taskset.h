#pragma once

#include <errno.h>
#include <sched.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <vector>

#define TASKSET_CHECK(exp)                                              \
  do {                                                                  \
    auto rc = (exp);                                                    \
    if (rc < 0) {                                                       \
      auto msg = fmt::format(#exp " fail. error: {}", strerror(errno)); \
      SPDLOG_ERROR(msg);                                                \
      throw std::runtime_error(msg);                                    \
    }                                                                   \
  } while (0)

struct Taskset {
  /**
   * @brief Bind current process to a specific CPU core
   * @param cpu CPU core ID to bind to
   * @throws std::runtime_error if sched_setaffinity fails
   */
  inline static void Set(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);  // Bind process to 'cpu'

    pid_t pid = getpid();
    TASKSET_CHECK(sched_setaffinity(pid, sizeof(mask), &mask));
  }

  /**
   * @brief Bind current process to multiple CPU cores
   * @param cpus Vector of CPU core IDs to bind to
   * @throws std::runtime_error if sched_setaffinity fails
   */
  inline static void Set(std::vector<int> cpus) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (auto cpu : cpus) CPU_SET(cpu, &mask);

    pid_t pid = getpid();
    TASKSET_CHECK(sched_setaffinity(pid, sizeof(mask), &mask));
  }
};
