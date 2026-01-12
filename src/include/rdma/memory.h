/**
 * @file memory.h
 * @brief Shared symmetric memory utilities
 *
 * Common utilities for symmetric memory used by both IB and Fabric implementations
 */
#pragma once

#include <cstdint>

namespace rdma {

/**
 * @brief Encode immediate data with channel index
 * @param imm_data User immediate data
 * @param ch Channel index
 * @return Encoded immediate data with channel in lower 8 bits
 */
static constexpr uint64_t EncodeImmdata(uint64_t imm_data, size_t ch) noexcept { return (imm_data << 8) | (ch & 0xFF); }

/**
 * @brief Decode channel index from immediate data
 * @param encoded Encoded immediate data
 * @return Channel index
 */
static constexpr size_t DecodeChannel(uint64_t encoded) noexcept { return encoded & 0xFF; }

/**
 * @brief Decode user immediate data
 * @param encoded Encoded immediate data
 * @return Original user immediate data
 */
static constexpr uint64_t DecodeImmdata(uint64_t encoded) noexcept { return encoded >> 8; }

}  // namespace rdma
