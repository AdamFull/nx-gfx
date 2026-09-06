#pragma once

#include "core/rendering/rhi/image/image.h"

#include <span>

namespace nxe::rhi {

[[nodiscard]] bool is_ktx(std::span<const u8> bytes) noexcept;

/// Validates and decodes a KTX 1.0 container into tightly packed image
/// subresources, including opposite-endian uncompressed payloads.
[[nodiscard]] nx::result<ImageData, ImageError>
decode_ktx(std::span<const u8> bytes);

} // namespace nxe::rhi
