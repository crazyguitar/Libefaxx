#pragma once
#include <stdint.h>

enum class DeviceRequestType : uint32_t { kPut = 0, kGet = 1, kCount = 2 };

struct DeviceRequest {
  uint64_t type;
  uint64_t size;
  uint64_t src;
  uint64_t dst;
};
