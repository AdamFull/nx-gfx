#pragma once

#include "core/foundation/containers/blob.h"
#include "core/foundation/core/foundation.h"
#include "core/foundation/core/result.h"
#include "core/foundation/platform/filesystem.h"
#include "core/foundation/strings/utf8_string_view.h"
#include "core/rendering/rhi/descs.h"
#include "core/rendering/rhi/types.h"

#include <span>

namespace nxe::rhi {

enum class ImageError : u8 {
  Unknown = 0,
  Io,
  Empty,             ///< Zero-byte input.
  UnknownContainer,
  UnsupportedFormat,
  Corrupt,
  Count
};

[[nodiscard]] nx::string_view to_string(ImageError error) noexcept;

struct ImageSubresource {
  u64 offset = 0; ///< Byte offset into ImageData::pixels.
  u64 size = 0;   ///< Byte length of this subresource.
  u32 mip = 0;
  u32 layer = 0;
  u32 width = 0;
  u32 height = 0;
  u32 depth = 1;
  u32 row_pitch = 0; ///< Bytes per block-row.

  u32 x = 0;
  u32 y = 0;
  u32 z = 0;
};

struct ImageData {
  Format format = Format::Unknown;
  TextureDimension dimension = TextureDimension::Tex2D;
  u32 width = 1;
  u32 height = 1;
  u32 depth = 1;
  u32 mip_levels = 1;
  u32 array_layers = 1;
  nx::blob<u8> pixels;
  nx::vector<ImageSubresource> subresources;

  [[nodiscard]] bool valid() const noexcept {
    return format != Format::Unknown && !pixels.empty() &&
           !subresources.empty();
  }

  [[nodiscard]] TextureDesc
  texture_desc(nx::string_view name = "image",
               TextureUsageFlags usage = TextureUsage::Sampled |
                                         TextureUsage::CopyDst) const noexcept;
};

[[nodiscard]] nx::result<ImageData, ImageError>
load_image_from_memory(std::span<const u8> bytes);

[[nodiscard]] nx::result<ImageData, ImageError>
load_image(nx::fs::path_view path);

void generate_mip_chain(ImageData &image);

}
