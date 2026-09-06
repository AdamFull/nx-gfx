#pragma once

#include "core/rendering/rhi/image/image.h"

#include <span>

namespace nxe::rhi {

[[nodiscard]] bool is_dds(std::span<const u8> bytes) noexcept;

/// Validates and decodes a DDS container into tightly packed image
/// subresources. Both legacy headers and DX10 extended headers are supported.
[[nodiscard]] nx::result<ImageData, ImageError>
decode_dds(std::span<const u8> bytes);

} // namespace nxe::rhi
