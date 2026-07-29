#pragma once

/**
 * @file image.h
 * @brief CPU-side image decoding for DDS, KTX 1.0 and the stb formats.
 *
 * Decoding turns a file's bytes into an ImageData: a tightly-packed pixel blob
 * plus the per-subresource layout the uploader needs, and the rhi::Format the
 * texture must be created with. It is a pure CPU operation - no GPU, and no
 * filesystem beyond load_image()'s convenience read - so it is safe to run on a
 * job thread and hand the result to the uploader afterwards.
 *
 */

#include "foundation/containers/blob.h"
#include "foundation/core/foundation.h"
#include "foundation/core/result.h"
#include "foundation/platform/filesystem.h"
#include "foundation/strings/utf8_string_view.h"
#include "rendering/rhi/descs.h"
#include "rendering/rhi/types.h"

#include <span>

namespace nxe::rhi {

enum class ImageError : u8 {
  Unknown = 0,
  Io,                ///< load_image()'s file read failed.
  Empty,             ///< Zero-byte input.
  UnknownContainer,  ///< Not DDS, KTX, nor anything stb recognises.
  UnsupportedFormat, ///< A pixel format with no rhi::Format equivalent.
  Corrupt,           ///< Header or pixel data is malformed or truncated.
  Count
};

[[nodiscard]] nx::string_view to_string(ImageError error) noexcept;

/// One mip level of one array layer, addressing a run of ImageData::pixels.
struct ImageSubresource {
  u64 offset = 0; ///< Byte offset into ImageData::pixels.
  u64 size = 0;   ///< Byte length of this subresource.
  u32 mip = 0;
  u32 layer = 0; ///< Array index / cube face; 0 for a plain 2D image.
  u32 width = 0;
  u32 height = 0;
  u32 depth = 1;
  u32 row_pitch = 0; ///< Bytes per block-row.

  u32 x = 0;
  u32 y = 0;
  u32 z = 0;
};

/// A fully decoded image, ready to hand to the uploader.
struct ImageData {
  Format format = Format::Unknown;
  TextureDimension dimension = TextureDimension::Tex2D;
  u32 width = 1;
  u32 height = 1;
  u32 depth = 1; ///< 3D depth; 1 for 2D/cube/array.
  u32 mip_levels = 1;
  u32 array_layers = 1; ///< Cube = 6 (*N), array length, else 1.
  nx::blob<u8> pixels;
  nx::vector<ImageSubresource> subresources;

  [[nodiscard]] bool valid() const noexcept {
    return format != Format::Unknown && !pixels.empty() &&
           !subresources.empty();
  }

  /// A texture descriptor matching this image. The default usage - sampled and
  /// copy-destination - is exactly what the uploader fills into.
  [[nodiscard]] TextureDesc
  texture_desc(nx::string_view name = "image",
               TextureUsageFlags usage = TextureUsage::Sampled |
                                         TextureUsage::CopyDst) const noexcept;
};

/// Decodes an image already resident in memory. The span is read only during
/// the call; nothing is retained.
[[nodiscard]] nx::result<ImageData, ImageError>
load_image_from_memory(std::span<const u8> bytes);

/// Reads @p path in full, then decodes it.
[[nodiscard]] nx::result<ImageData, ImageError>
load_image(nx::fs::path_view path);

} // namespace nxe::rhi
