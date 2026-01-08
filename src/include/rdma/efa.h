/**
 * @file efa.h
 * @brief Shared EFA constants and utilities
 */
#pragma once
#include <spdlog/spdlog.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace rdma {

static constexpr size_t kMaxAddrSize = 64;
static constexpr size_t kAddrSize = 32;

inline std::string Addr2Str(const char* addr) {
  std::string out;
  for (size_t i = 0; i < kAddrSize; ++i) out += fmt::format("{:02x}", static_cast<uint8_t>(addr[i]));
  return out;
}

inline void Str2Addr(const std::string& addr, char* bytes) noexcept {
  for (size_t i = 0; i < kAddrSize; ++i) std::sscanf(addr.c_str() + 2 * i, "%02hhx", &bytes[i]);
}

}  // namespace rdma
