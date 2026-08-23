#pragma once

#include "core/foundation/strings/utf8_string.h"
#include "core/rendering/rhi/image/image.h"

#include <span>

namespace nxe::rhi {

/// A decoded image as an nva container: its layout in an INFO chunk, its pixels
/// in a DATA chunk. Deterministic, so a cook is reproducible.
[[nodiscard]] nx::vector<u8> encode_texture(const ImageData &image);

/// Reads a container built by encode_texture into @p out. False when the bytes
/// are not a valid texture container.
[[nodiscard]] bool decode_texture(std::span<const u8> bytes, ImageData &out);

[[nodiscard]] nx::string cooked_texture_path(nx::string_view source);

}
