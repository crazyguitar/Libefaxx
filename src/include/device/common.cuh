/**
 * @file common.cuh
 * @brief Common CUDA utilities and error checking macros
 */
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

constexpr size_t kAlign = 128;

/**
 * @brief CUDA error checking macro
 * @param exp CUDA function call to check
 */
#define CUDA_CHECK(exp)                                                                                    \
  do {                                                                                                     \
    cudaError_t err = (exp);                                                                               \
    if (err != cudaSuccess) {                                                                              \
      SPDLOG_CRITICAL("[{}:{}] " #exp " got CUDA error: {}", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1);                                                                                             \
    }                                                                                                      \
  } while (0)

#define CU_CHECK(exp)                                                                                            \
  do {                                                                                                           \
    CUresult rc = (exp);                                                                                         \
    if (rc != CUDA_SUCCESS) {                                                                                    \
      const char* err_str = nullptr;                                                                             \
      cuGetErrorString(rc, &err_str);                                                                            \
      SPDLOG_ERROR("{} failed with {} ({})", #exp, static_cast<int>(rc), (err_str ? err_str : "Unknown error")); \
      exit(1);                                                                                                   \
    }                                                                                                            \
  } while (0)

/**
 * @brief Launch kernel with config and error checking
 */
#define LAUNCH_KERNEL(cfg, kernel, ...) CUDA_CHECK(cudaLaunchKernelEx(cfg, kernel, ##__VA_ARGS__))
