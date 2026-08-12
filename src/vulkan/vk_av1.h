#pragma once

/**
 * @file vk_av1.h
 * @brief Parsing an AV1 sequence header into the Std structs a Vulkan Video AV1
 * decode session's parameters need (namespace nxe::rhi).
 */

#include "core/rendering/rhi/vulkan/vk_common.h"

namespace nxe::rhi {

struct Av1SequenceHeader {
  bool valid = false;
  StdVideoAV1SequenceHeader header{};
  StdVideoAV1ColorConfig color{};
  StdVideoAV1TimingInfo timing{};
};

[[nodiscard]] bool parse_av1_sequence_header(const u8 *data, u32 size,
                                             Av1SequenceHeader &out);

[[nodiscard]] bool parse_av1_codec_private(const u8 *data, u32 size,
                                           Av1SequenceHeader &out);

} // namespace nxe::rhi
