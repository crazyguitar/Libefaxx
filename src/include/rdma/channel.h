/**
 * @file channel.h
 * @brief Shared channel constants
 */
#pragma once
#include <cstddef>

namespace rdma {

static constexpr size_t kChunkSize = 1 << 20;  // 1MB

}  // namespace rdma
