#pragma once

#include "core/foundation/containers/blob.h"
#include "core/foundation/core/foundation.h"
#include "core/foundation/core/result.h"
#include "core/foundation/platform/filesystem.h"
#include "core/foundation/strings/utf8_string_view.h"
#include "rendering/rhi/descs.h"
#include "rendering/rhi/texture_data.h"
#include "rendering/rhi/types.h"

#include <span>

namespace nxe::rhi {

enum class ImageError : u8 {
  Unknown = 0,
  Io,
  Empty, ///< Zero-byte input.
  UnknownContainer,
  UnsupportedFormat,
  Corrupt,
  Count
};

[[nodiscard]] nx::string_view to_string(ImageError error) noexcept;

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

  // An upload view of the decoded texels; valid while this image lives.
  operator TextureData() const noexcept {
    return {format,
            {pixels.data(), pixels.size()},
            {subresources.data(), subresources.size()}};
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

} // namespace nxe::rhi
