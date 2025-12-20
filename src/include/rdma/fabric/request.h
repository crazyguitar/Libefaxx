#pragma once
#include <stdint.h>

#include <algorithm>
#include <tuple>
#include <vector>

enum class DeviceRequestType : uint32_t { kPut = 0, kGet = 1, kCount = 2 };

struct DeviceRequest {
  uint64_t type;
  uint64_t rank;
  uint64_t size;
  uint64_t addr;
  uint64_t imm;
};

/** @brief Merge contiguous device requests by (rank, type, addr) */
static inline std::vector<DeviceRequest> Merge(std::vector<DeviceRequest>& reqs) {
  if (reqs.empty()) return {};
  auto cmp = [](const auto& a, const auto& b) { return std::tie(a.rank, a.type, a.addr) < std::tie(b.rank, b.type, b.addr); };
  std::sort(reqs.begin(), reqs.end(), cmp);
  std::vector<DeviceRequest> result;
  result.reserve(reqs.size());
  result.push_back(std::move(reqs[0]));
  for (size_t i = 1; i < reqs.size(); ++i) {
    auto& last = result.back();
    auto& cur = reqs[i];
    if (cur.rank == last.rank && cur.type == last.type && cur.addr == last.addr + last.size) {
      last.size += cur.size;
    } else {
      result.push_back(std::move(cur));
    }
  }
  reqs.clear();
  return result;
}
