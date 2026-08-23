#include "core/rendering/rhi/image/texture_nva.h"

#include "core/foundation/serialization/nva.h"

#include <cstring>

namespace nxe::rhi {
namespace {

constexpr u32 FORMAT = nx::nva::fourcc('T', 'E', 'X', ' ');
constexpr u32 CHUNK_INFO = nx::nva::fourcc('I', 'N', 'F', 'O');
constexpr u32 CHUNK_DATA = nx::nva::fourcc('D', 'A', 'T', 'A');
constexpr u16 VERSION = 1;

}

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
  if (!view)
    return false;
  const std::span<const u8> info = view->chunk(CHUNK_INFO);
  const std::span<const u8> data = view->chunk(CHUNK_DATA);
  if (info.empty())
    return false;

  nx::nva::Reader r(info);
  out.format = nx::cast<Format>(r.u32v());
  out.dimension = nx::cast<TextureDimension>(r.u32v());
  out.width = r.u32v();
  out.height = r.u32v();
  out.depth = r.u32v();
  out.mip_levels = r.u32v();
  out.array_layers = r.u32v();

  const u32 count = r.u32v();
  out.subresources.clear();
  out.subresources.reserve(count);
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
    out.subresources.push_back(s);
  }

  out.pixels = nx::blob<u8>(data.size());
  if (!data.empty())
    std::memcpy(out.pixels.data(), data.data(), data.size());

  return r.ok() && out.valid();
}

nx::string cooked_texture_path(const nx::string_view source) {
  nx::string out(source);
  out += ".nxb";
  return out;
}

}
