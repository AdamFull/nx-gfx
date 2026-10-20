#pragma once

#include "rendering/rhi/image/image.h"

#include <cstring>
#include <limits>

namespace nxe::rhi::detail {

using ImageResult = nx::result<ImageData, ImageError>;

/// Sequential cursor over encoded image bytes, bounds-checked on every read.
struct ImageReader {
  const u8 *data = nullptr;
  usize size = 0;
  usize pos = 0;

  [[nodiscard]] bool ok(const usize count) const noexcept {
    return pos <= size && count <= size - pos;
  }

  [[nodiscard]] usize remaining() const noexcept {
    return pos <= size ? size - pos : 0;
  }

  bool read(void *destination, const usize count) noexcept {
    if (!ok(count))
      return false;
    std::memcpy(destination, data + pos, count);
    pos += count;
    return true;
  }

  template <typename T> bool read(T &value) noexcept {
    return read(&value, sizeof(T));
  }

  bool skip(const usize count) noexcept {
    if (!ok(count))
      return false;
    pos += count;
    return true;
  }

  bool align4() noexcept { return skip((4u - (pos & 3u)) & 3u); }
};

[[nodiscard]] inline u32 mip_dimension(const u32 base, const u32 mip) noexcept {
  return mip >= 32 ? 1u : nx::max(1u, base >> mip);
}

[[nodiscard]] inline bool checked_add(const u64 a, const u64 b,
                                      u64 &result) noexcept {
  if (b > std::numeric_limits<u64>::max() - a)
    return false;
  result = a + b;
  return true;
}

[[nodiscard]] inline bool checked_align4(const u64 value,
                                         u64 &aligned) noexcept {
  return checked_add(value, (4u - (value & 3u)) & 3u, aligned);
}

struct SubresourceLayout {
  u64 row_pitch = 0;
  u64 size = 0;
};

[[nodiscard]] inline bool
subresource_layout(const Format format, const u32 width, const u32 height,
                   const u32 depth, SubresourceLayout &layout) noexcept {
  if (width == 0 || height == 0 || depth == 0)
    return false;

  u32 block_width = 0;
  u32 block_height = 0;
  format_block_extent(format, block_width, block_height);
  const u64 texel_size = format_texel_size(format);
  if (block_width == 0 || block_height == 0 || texel_size == 0)
    return false;

  const u64 blocks_x = (nx::cast<u64>(width) + block_width - 1) / block_width;
  const u64 blocks_y =
      (nx::cast<u64>(height) + block_height - 1) / block_height;
  if (!checked_mul(blocks_x, texel_size, layout.row_pitch) ||
      layout.row_pitch > std::numeric_limits<u32>::max())
    return false;

  u64 slice_size = 0;
  return checked_mul(layout.row_pitch, blocks_y, slice_size) &&
         checked_mul(slice_size, depth, layout.size);
}

[[nodiscard]] inline u32 maximum_mip_levels(const u32 width, const u32 height,
                                            const u32 depth) noexcept {
  u32 largest = nx::max(width, nx::max(height, depth));
  u32 levels = 0;
  while (largest != 0) {
    ++levels;
    largest >>= 1;
  }
  return levels;
}

[[nodiscard]] inline bool image_total_size(const Format format, const u32 width,
                                           const u32 height, const u32 depth,
                                           const u32 mip_levels,
                                           const u32 layers, const bool is_3d,
                                           u64 &total) noexcept {
  if (width == 0 || height == 0 || depth == 0 || mip_levels == 0 ||
      layers == 0 || mip_levels > maximum_mip_levels(width, height, depth))
    return false;

  total = 0;
  for (u32 mip = 0; mip < mip_levels; ++mip) {
    SubresourceLayout layout;
    if (!subresource_layout(format, mip_dimension(width, mip),
                            mip_dimension(height, mip),
                            is_3d ? mip_dimension(depth, mip) : 1u, layout) ||
        !checked_add(total, layout.size, total))
      return false;
  }
  return checked_mul(total, layers, total);
}

inline void append_subresource(ImageData &image, const u64 offset,
                               const u32 mip, const u32 layer, const u32 width,
                               const u32 height, const u32 depth) {
  ImageSubresource subresource;
  subresource.offset = offset;
  subresource.size =
      format_subresource_size(image.format, width, height, depth);
  subresource.mip = mip;
  subresource.layer = layer;
  subresource.width = width;
  subresource.height = height;
  subresource.depth = depth;
  subresource.row_pitch = nx::cast<u32>(format_row_pitch(image.format, width));
  image.subresources.push_back(subresource);
}

[[nodiscard]] inline TextureDimension
image_dimension(const bool cube, const bool is_3d, const u32 layers) noexcept {
  if (cube)
    return TextureDimension::TexCube;
  if (is_3d)
    return TextureDimension::Tex3D;
  return layers > 1 ? TextureDimension::Tex2DArray : TextureDimension::Tex2D;
}

} // namespace nxe::rhi::detail
