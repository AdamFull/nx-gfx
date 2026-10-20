#include "nx/gfx/upload.h"

namespace nx::gfx {

bool validate_image_upload(
    const image_upload_desc &target, const std::span<const u8> src,
    const std::span<const image_subresource> layout) noexcept {
  u32 block_width = 0;
  u32 block_height = 0;
  format_block_extent(target.format, block_width, block_height);
  const u64 bytes_per_block = format_texel_size(target.format);
  if (target.format == format::Unknown || target.width == 0 ||
      target.height == 0 || target.depth == 0 || target.mip_levels == 0 ||
      target.array_layers == 0 || block_width == 0 || block_height == 0 ||
      bytes_per_block == 0 || layout.empty())
    return false;

  for (const image_subresource &sub : layout) {
    if (sub.size == 0 || sub.offset > src.size() ||
        sub.size > src.size() - sub.offset || sub.mip >= target.mip_levels ||
        sub.layer >= target.array_layers || sub.width == 0 || sub.height == 0 ||
        sub.depth == 0)
      return false;

    const auto mip_extent = [mip = sub.mip](const u32 extent) {
      return mip >= 32u ? 1u : nx::max(1u, extent >> mip);
    };
    const u32 mip_width = mip_extent(target.width);
    const u32 mip_height = mip_extent(target.height);
    const u32 mip_depth = mip_extent(target.depth);
    if (sub.x > mip_width || sub.width > mip_width - sub.x ||
        sub.y > mip_height || sub.height > mip_height - sub.y ||
        sub.z > mip_depth || sub.depth > mip_depth - sub.z)
      return false;

    // Compressed copies start on block boundaries. Their final partial block
    // is legal only when it reaches the edge of the selected mip.
    if (sub.x % block_width != 0 || sub.y % block_height != 0 ||
        (sub.width % block_width != 0 && sub.x + sub.width != mip_width) ||
        (sub.height % block_height != 0 && sub.y + sub.height != mip_height))
      return false;

    const u64 tight_pitch = format_row_pitch(target.format, sub.width);
    if (sub.row_pitch < tight_pitch || sub.row_pitch % bytes_per_block != 0)
      return false;
    const u64 blocks_per_row = sub.row_pitch / bytes_per_block;
    if (blocks_per_row > std::numeric_limits<u32>::max() / block_width)
      return false;

    const u64 rows =
        (nx::cast<u64>(sub.height) + block_height - 1u) / block_height;
    u64 bytes_per_slice = 0;
    u64 required = 0;
    if (!detail::checked_mul(sub.row_pitch, rows, bytes_per_slice) ||
        !detail::checked_mul(bytes_per_slice, sub.depth, required) ||
        sub.size != required)
      return false;
  }
  return true;
}

} // namespace nx::gfx
