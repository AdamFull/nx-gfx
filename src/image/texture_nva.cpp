#include "core/rendering/rhi/image/texture_nva.h"

#include "core/foundation/serialization/nva.h"

#include <cstring>
#include <limits>

namespace nxe::rhi {
namespace {

constexpr u32 FORMAT = nx::nva::fourcc('T', 'E', 'X', ' ');
constexpr u32 CHUNK_INFO = nx::nva::fourcc('I', 'N', 'F', 'O');
constexpr u32 CHUNK_DATA = nx::nva::fourcc('D', 'A', 'T', 'A');
constexpr u16 VERSION = 1;

[[nodiscard]] u32 dim_at(const u32 base, const u32 mip) noexcept {
  return nx::max(1u, base >> mip);
}

[[nodiscard]] u32 maximum_mip_levels(const u32 width, const u32 height,
                                     const u32 depth) noexcept {
  u32 largest = nx::max(width, nx::max(height, depth));
  u32 levels = 0;
  while (largest != 0) {
    ++levels;
    largest >>= 1;
  }
  return levels;
}

[[nodiscard]] bool subresource_layout(const Format format, const u32 width,
                                      const u32 height, const u32 depth,
                                      u32 &row_pitch, u64 &size) noexcept {
  u32 block_width = 0;
  u32 block_height = 0;
  format_block_extent(format, block_width, block_height);
  const u64 texel_size = format_texel_size(format);
  if (block_width == 0 || block_height == 0 || texel_size == 0)
    return false;

  const u64 blocks_x = (nx::cast<u64>(width) + block_width - 1) / block_width;
  const u64 blocks_y =
      (nx::cast<u64>(height) + block_height - 1) / block_height;
  const u64 row = blocks_x * texel_size;
  if (row > std::numeric_limits<u32>::max() ||
      blocks_y > std::numeric_limits<u64>::max() / row)
    return false;
  const u64 slice = row * blocks_y;
  if (depth > std::numeric_limits<u64>::max() / slice)
    return false;
  row_pitch = nx::cast<u32>(row);
  size = slice * depth;
  return true;
}

} // namespace

nx::vector<u8> encode_texture(const ImageData &image) {
  nx::nva::Writer info;
  info.u32v(nx::cast<u32>(image.format));
  info.u32v(nx::cast<u32>(image.dimension));
  info.u32v(image.width);
  info.u32v(image.height);
  info.u32v(image.depth);
  info.u32v(image.mip_levels);
  info.u32v(image.array_layers);
  info.u32v(nx::cast<u32>(image.subresources.size()));
  for (const ImageSubresource &s : image.subresources) {
    info.u64v(s.offset);
    info.u64v(s.size);
    info.u32v(s.mip);
    info.u32v(s.layer);
    info.u32v(s.width);
    info.u32v(s.height);
    info.u32v(s.depth);
    info.u32v(s.row_pitch);
    info.u32v(s.x);
    info.u32v(s.y);
    info.u32v(s.z);
  }

  nx::nva::ChunkWriter w(FORMAT, VERSION);
  w.add(CHUNK_INFO, info.span());
  w.add(CHUNK_DATA, {image.pixels.data(), image.pixels.size()});
  return w.finish();
}

bool decode_texture(const std::span<const u8> bytes, ImageData &out) {
  const std::optional<nx::nva::View> view = nx::nva::View::open(bytes, FORMAT);
  if (!view || view->version() != VERSION)
    return false;
  const std::span<const u8> info = view->chunk(CHUNK_INFO);
  const std::span<const u8> data = view->chunk(CHUNK_DATA);
  if (info.empty() || data.empty())
    return false;

  nx::nva::Reader r(info);
  const u32 format = r.u32v();
  const u32 dimension = r.u32v();
  ImageData decoded;
  decoded.format = nx::cast<Format>(format);
  decoded.dimension = nx::cast<TextureDimension>(dimension);
  decoded.width = r.u32v();
  decoded.height = r.u32v();
  decoded.depth = r.u32v();
  decoded.mip_levels = r.u32v();
  decoded.array_layers = r.u32v();

  if (!r.ok() || format == nx::cast<u32>(Format::Unknown) ||
      format >= nx::cast<u32>(Format::Count) ||
      dimension >= nx::cast<u32>(TextureDimension::Count) ||
      decoded.width == 0 || decoded.height == 0 || decoded.depth == 0 ||
      decoded.mip_levels == 0 ||
      decoded.mip_levels >
          maximum_mip_levels(decoded.width, decoded.height, decoded.depth) ||
      decoded.array_layers == 0 ||
      decoded.array_layers > MAX_IMAGE_ARRAY_LAYERS)
    return false;

  switch (decoded.dimension) {
  case TextureDimension::Tex2D:
    if (decoded.depth != 1 || decoded.array_layers != 1)
      return false;
    break;
  case TextureDimension::Tex2DArray:
    if (decoded.depth != 1)
      return false;
    break;
  case TextureDimension::TexCube:
    if (decoded.depth != 1 || decoded.width != decoded.height ||
        decoded.array_layers % 6u != 0)
      return false;
    break;
  case TextureDimension::Tex3D:
    if (decoded.array_layers != 1)
      return false;
    break;
  default:
    return false;
  }

  const u32 count = r.u32v();
  const u64 layers =
      decoded.dimension == TextureDimension::Tex3D ? 1u : decoded.array_layers;
  const u64 expected_count = layers * decoded.mip_levels;
  if (count != expected_count || !r.require_items(count, 52))
    return false;
  decoded.subresources.reserve(count);
  nx::vector<u8> seen(count, 0);
  for (u32 i = 0; i < count; ++i) {
    ImageSubresource s;
    s.offset = r.u64v();
    s.size = r.u64v();
    s.mip = r.u32v();
    s.layer = r.u32v();
    s.width = r.u32v();
    s.height = r.u32v();
    s.depth = r.u32v();
    s.row_pitch = r.u32v();
    s.x = r.u32v();
    s.y = r.u32v();
    s.z = r.u32v();
    if (!r.ok() || s.mip >= decoded.mip_levels || s.layer >= layers ||
        s.x != 0 || s.y != 0 || s.z != 0)
      return false;

    const u32 expected_width = dim_at(decoded.width, s.mip);
    const u32 expected_height = dim_at(decoded.height, s.mip);
    const u32 expected_depth = decoded.dimension == TextureDimension::Tex3D
                                   ? dim_at(decoded.depth, s.mip)
                                   : 1u;
    u32 expected_pitch = 0;
    u64 expected_size = 0;
    if (s.width != expected_width || s.height != expected_height ||
        s.depth != expected_depth ||
        !subresource_layout(decoded.format, expected_width, expected_height,
                            expected_depth, expected_pitch, expected_size) ||
        s.row_pitch != expected_pitch || s.size != expected_size ||
        s.offset > data.size() || s.size > data.size() - s.offset)
      return false;

    const usize key = nx::cast<usize>(s.layer) * decoded.mip_levels + s.mip;
    if (seen[key] != 0)
      return false;
    seen[key] = 1;
    decoded.subresources.push_back(s);
  }

  decoded.pixels = nx::blob<u8>(data.size());
  std::memcpy(decoded.pixels.data(), data.data(), data.size());

  if (!r.ok() || !decoded.valid())
    return false;
  out = std::move(decoded);
  return true;
}

nx::string cooked_texture_path(const nx::string_view source) {
  nx::string out(source);
  out += ".nxb";
  return out;
}

} // namespace nxe::rhi
