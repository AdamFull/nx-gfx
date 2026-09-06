#include "core/rendering/rhi/image/texture_nva.h"

#include "core/foundation/serialization/nva.h"

#include <cstring>
#include <limits>

namespace nxe::rhi {
namespace {

constexpr u32 FORMAT = nx::nva::fourcc('T', 'E', 'X', ' ');
constexpr u32 CHUNK_INFO = nx::nva::fourcc('I', 'N', 'F', 'O');
constexpr u32 CHUNK_DATA = nx::nva::fourcc('D', 'A', 'T', 'A');
constexpr u32 CHUNK_KTX2 = nx::nva::fourcc('K', 'T', 'X', '2');
constexpr u16 VERSION_LEGACY = 1;
constexpr u16 VERSION_BASIS = 2;
constexpr u16 VERSION = 3;
constexpr u32 BASIS_FLAG_ALPHA = 1u << 0;
constexpr u32 BASIS_FLAG_SRGB = 1u << 1;
constexpr u32 BASIS_FLAG_UASTC = 1u << 2;

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

[[nodiscard]] bool decode_native_info(const std::span<const u8> info,
                                      const u64 data_size, const bool has_kind,
                                      ImageData &out) {
  if (info.empty() || data_size == 0)
    return false;

  nx::nva::Reader r(info);
  if (has_kind && r.u32v() != nx::cast<u32>(TexturePayloadKind::Native))
    return false;
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
        s.offset > data_size || s.size > data_size - s.offset)
      return false;

    const usize key = nx::cast<usize>(s.layer) * decoded.mip_levels + s.mip;
    if (seen[key] != 0)
      return false;
    seen[key] = 1;
    decoded.subresources.push_back(s);
  }

  if (!r.ok() || r.remaining() != 0)
    return false;
  out = std::move(decoded);
  return true;
}

void copy_payload(const std::span<const u8> source, nx::blob<u8> &out,
                  const nx::cancellation_token &cancellation) {
  constexpr usize COPY_CHUNK = 256u * 1024u;
  nx::blob<u8> copied(source.size());
  for (usize offset = 0; offset < source.size();) {
    cancellation.throw_if_requested();
    const usize count = nx::min(COPY_CHUNK, source.size() - offset);
    std::memcpy(copied.data() + offset, source.data() + offset, count);
    offset += count;
  }
  cancellation.throw_if_requested();
  out = std::move(copied);
}

[[nodiscard]] bool
decode_native_payload(const std::span<const u8> info,
                      const std::span<const u8> data, const bool has_kind,
                      ImageData &out,
                      const nx::cancellation_token &cancellation) {
  ImageData decoded;
  if (!decode_native_info(info, data.size(), has_kind, decoded))
    return false;
  copy_payload(data, decoded.pixels, cancellation);
  if (!decoded.valid())
    return false;
  out = std::move(decoded);
  return true;
}

[[nodiscard]] bool read_exact(nx::fs::stream &stream, void *destination,
                              usize size,
                              const nx::cancellation_token &cancellation = {}) {
  constexpr usize CANCEL_CHUNK = 256u * 1024u;
  auto *bytes = static_cast<u8 *>(destination);
  usize read = 0;
  while (read < size) {
    cancellation.throw_if_requested();
    const usize request = nx::min(size - read, CANCEL_CHUNK);
    const usize count = stream.read(bytes + read, request);
    if (count == 0)
      return false;
    read += count;
  }
  cancellation.throw_if_requested();
  return true;
}

[[nodiscard]] u32 read_u32(const u8 *bytes) noexcept {
  return nx::cast<u32>(bytes[0]) | (nx::cast<u32>(bytes[1]) << 8) |
         (nx::cast<u32>(bytes[2]) << 16) | (nx::cast<u32>(bytes[3]) << 24);
}

} // namespace

nx::vector<u8> encode_texture(const ImageData &image) {
  if (!image.valid() || image.pixels.size() > std::numeric_limits<u32>::max())
    return {};

  const u32 layers =
      image.dimension == TextureDimension::Tex3D ? 1u : image.array_layers;
  if (image.subresources.size() != nx::cast<usize>(layers) * image.mip_levels)
    return {};

  ImageData stored;
  stored.format = image.format;
  stored.dimension = image.dimension;
  stored.width = image.width;
  stored.height = image.height;
  stored.depth = image.depth;
  stored.mip_levels = image.mip_levels;
  stored.array_layers = image.array_layers;
  stored.pixels = nx::blob<u8>(image.pixels.size());
  stored.subresources.resize(image.subresources.size());

  nx::vector<const ImageSubresource *> sources(image.subresources.size(),
                                               nullptr);
  for (const ImageSubresource &subresource : image.subresources) {
    if (subresource.mip >= image.mip_levels || subresource.layer >= layers)
      return {};
    const usize key =
        nx::cast<usize>(subresource.layer) * image.mip_levels + subresource.mip;
    if (sources[key] != nullptr)
      return {};
    sources[key] = &subresource;
  }

  // INFO remains canonical layer-major/mip-major while DATA is coarse-first.
  // The explicit offsets bridge the two orders for full decoding and let the
  // streaming reader consume the ZIP entry strictly forwards.
  u64 stored_offset = 0;
  for (u32 mip = image.mip_levels; mip-- > 0;) {
    for (u32 layer = 0; layer < layers; ++layer) {
      const ImageSubresource *const source =
          sources[nx::cast<usize>(layer) * image.mip_levels + mip];
      if (source == nullptr || source->offset > image.pixels.size() ||
          source->size > image.pixels.size() - source->offset ||
          source->size > stored.pixels.size() - stored_offset)
        return {};
      std::memcpy(stored.pixels.data() + nx::cast<usize>(stored_offset),
                  image.pixels.data() + nx::cast<usize>(source->offset),
                  nx::cast<usize>(source->size));
      ImageSubresource encoded = *source;
      encoded.offset = stored_offset;
      stored.subresources[nx::cast<usize>(layer) * image.mip_levels + mip] =
          encoded;
      stored_offset += source->size;
    }
  }
  if (stored_offset != stored.pixels.size())
    return {};

  nx::nva::Writer info;
  info.u32v(nx::cast<u32>(TexturePayloadKind::Native));
  info.u32v(nx::cast<u32>(stored.format));
  info.u32v(nx::cast<u32>(stored.dimension));
  info.u32v(stored.width);
  info.u32v(stored.height);
  info.u32v(stored.depth);
  info.u32v(stored.mip_levels);
  info.u32v(stored.array_layers);
  info.u32v(nx::cast<u32>(stored.subresources.size()));
  for (const ImageSubresource &s : stored.subresources) {
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
  w.add(CHUNK_DATA, {stored.pixels.data(), stored.pixels.size()});
  return w.finish();
}

nx::vector<u8> encode_basis_texture(const std::span<const u8> basis_ktx2) {
  BasisTextureInfo texture_info;
  if (!inspect_basis_texture(basis_ktx2, texture_info))
    return {};

  nx::nva::Writer info;
  info.u32v(nx::cast<u32>(TexturePayloadKind::BasisKtx2));
  info.u32v(nx::cast<u32>(texture_info.dimension));
  info.u32v(texture_info.width);
  info.u32v(texture_info.height);
  info.u32v(1);
  info.u32v(texture_info.mip_levels);
  info.u32v(texture_info.array_layers);
  info.u32v((texture_info.has_alpha ? BASIS_FLAG_ALPHA : 0u) |
            (texture_info.srgb ? BASIS_FLAG_SRGB : 0u) |
            (texture_info.encoding == BasisTextureEncoding::Uastc
                 ? BASIS_FLAG_UASTC
                 : 0u));

  nx::nva::ChunkWriter writer(FORMAT, VERSION_BASIS);
  writer.add(CHUNK_INFO, info.span());
  writer.add(CHUNK_KTX2, basis_ktx2);
  return writer.finish();
}

bool decode_texture_asset(const std::span<const u8> bytes, TextureAsset &out,
                          const nx::cancellation_token &cancellation) {
  cancellation.throw_if_requested();
  const std::optional<nx::nva::View> view = nx::nva::View::open(bytes, FORMAT);
  if (!view || (view->version() != VERSION_LEGACY &&
                view->version() != VERSION_BASIS && view->version() != VERSION))
    return false;
  const std::span<const u8> info = view->chunk(CHUNK_INFO);
  if (info.empty())
    return false;

  TextureAsset decoded;
  if (view->version() == VERSION_LEGACY) {
    if (!decode_native_payload(info, view->chunk(CHUNK_DATA), false,
                               decoded.native, cancellation))
      return false;
    decoded.kind = TexturePayloadKind::Native;
  } else {
    nx::nva::Reader reader(info);
    const u32 kind = reader.u32v();
    if (!reader.ok() || kind >= nx::cast<u32>(TexturePayloadKind::Count))
      return false;
    decoded.kind = nx::cast<TexturePayloadKind>(kind);
    if (decoded.kind == TexturePayloadKind::Native) {
      if (!decode_native_payload(info, view->chunk(CHUNK_DATA), true,
                                 decoded.native, cancellation))
        return false;
    } else {
      const u32 dimension = reader.u32v();
      BasisTextureInfo stored;
      stored.dimension = nx::cast<TextureDimension>(dimension);
      stored.width = reader.u32v();
      stored.height = reader.u32v();
      const u32 depth = reader.u32v();
      stored.mip_levels = reader.u32v();
      stored.array_layers = reader.u32v();
      const u32 flags = reader.u32v();
      stored.has_alpha = (flags & BASIS_FLAG_ALPHA) != 0;
      stored.srgb = (flags & BASIS_FLAG_SRGB) != 0;
      stored.encoding = (flags & BASIS_FLAG_UASTC) != 0
                            ? BasisTextureEncoding::Uastc
                            : BasisTextureEncoding::Etc1s;
      const std::span<const u8> payload = view->chunk(CHUNK_KTX2);
      BasisTextureInfo inspected;
      if (!reader.ok() || reader.remaining() != 0 ||
          dimension >= nx::cast<u32>(TextureDimension::Count) || depth != 1 ||
          (flags & ~(BASIS_FLAG_ALPHA | BASIS_FLAG_SRGB | BASIS_FLAG_UASTC)) !=
              0 ||
          payload.empty() || !inspect_basis_texture(payload, inspected) ||
          stored.dimension != inspected.dimension ||
          stored.width != inspected.width ||
          stored.height != inspected.height ||
          stored.mip_levels != inspected.mip_levels ||
          stored.array_layers != inspected.array_layers ||
          stored.encoding != inspected.encoding ||
          stored.has_alpha != inspected.has_alpha ||
          stored.srgb != inspected.srgb)
        return false;
      decoded.basis_info = stored;
      copy_payload(payload, decoded.basis_ktx2, cancellation);
    }
  }

  if (!decoded.valid())
    return false;
  out = std::move(decoded);
  return true;
}

bool open_texture_mip_stream(nx::fs::stream_ptr stream, TextureMipStream &out,
                             const nx::cancellation_token &cancellation) {
  cancellation.throw_if_requested();
  if (!stream || stream->size() < 16 ||
      !stream->seek(0, nx::fs::seek_origin::begin))
    return false;

  u8 header[16]{};
  if (!read_exact(*stream, header, sizeof(header), cancellation) ||
      read_u32(header) != nx::nva::MAGIC || read_u32(header + 4) != FORMAT ||
      (read_u32(header + 8) & 0xffffu) != VERSION)
    return false;
  const u32 chunk_count = read_u32(header + 12);
  if (chunk_count == 0 || chunk_count > 64)
    return false;
  const u64 table_size = nx::cast<u64>(chunk_count) * 12u;
  if (table_size > nx::cast<u64>(stream->size()) - sizeof(header))
    return false;

  nx::blob<u8> table(nx::cast<usize>(table_size));
  if (!read_exact(*stream, table.data(), table.size(), cancellation))
    return false;
  u32 info_offset = 0, info_size = 0, data_offset = 0, data_size = 0;
  for (u32 index = 0; index < chunk_count; ++index) {
    const u8 *const entry = table.data() + nx::cast<usize>(index) * 12u;
    const u32 id = read_u32(entry);
    const u32 offset = read_u32(entry + 4);
    const u32 size = read_u32(entry + 8);
    if (offset < sizeof(header) + table_size ||
        offset > nx::cast<u64>(stream->size()) ||
        size > nx::cast<u64>(stream->size()) - offset)
      return false;
    if (id == CHUNK_INFO) {
      if (info_size != 0)
        return false;
      info_offset = offset;
      info_size = size;
    } else if (id == CHUNK_DATA) {
      if (data_size != 0)
        return false;
      data_offset = offset;
      data_size = size;
    }
  }
  if (info_size == 0 || data_size == 0 ||
      !stream->seek(info_offset, nx::fs::seek_origin::begin))
    return false;
  nx::blob<u8> info(info_size);
  if (!read_exact(*stream, info.data(), info.size(), cancellation))
    return false;

  ImageData layout;
  if (!decode_native_info({info.data(), info.size()}, data_size, true,
                          layout) ||
      layout.dimension != TextureDimension::Tex2D || layout.array_layers != 1 ||
      layout.subresources.size() != layout.mip_levels)
    return false;

  u64 expected_offset = 0;
  for (u32 mip = layout.mip_levels; mip-- > 0;) {
    const ImageSubresource &subresource = layout.subresources[mip];
    if (subresource.mip != mip || subresource.layer != 0 ||
        subresource.offset != expected_offset ||
        subresource.size > data_size - expected_offset)
      return false;
    expected_offset += subresource.size;
  }
  if (expected_offset != data_size ||
      !stream->seek(data_offset, nx::fs::seek_origin::begin))
    return false;

  TextureMipStream opened;
  opened.m_stream = std::move(stream);
  opened.m_layout = std::move(layout);
  opened.m_data_offset = data_offset;
  opened.m_next_mip = opened.m_layout.mip_levels - 1u;
  out = std::move(opened);
  return true;
}

bool TextureMipStream::read_through(
    const u32 finest_mip, ImageData &out,
    const nx::cancellation_token &cancellation) {
  cancellation.throw_if_requested();
  if (!valid() || finest_mip > m_next_mip)
    return false;
  const u32 coarsest_mip = m_next_mip;
  const u64 file_offset = m_layout.subresources[coarsest_mip].offset;
  u64 size = 0;
  for (u32 mip = finest_mip; mip <= coarsest_mip; ++mip) {
    if (m_layout.subresources[mip].size >
        std::numeric_limits<u64>::max() - size)
      return false;
    size += m_layout.subresources[mip].size;
  }
  if (size == 0 || size > std::numeric_limits<usize>::max() ||
      m_data_offset >
          nx::cast<u64>(std::numeric_limits<i64>::max()) - file_offset ||
      !m_stream->seek(nx::cast<i64>(m_data_offset + file_offset),
                      nx::fs::seek_origin::begin))
    return false;

  ImageData image;
  image.format = m_layout.format;
  image.dimension = m_layout.dimension;
  image.width = m_layout.width;
  image.height = m_layout.height;
  image.depth = m_layout.depth;
  image.mip_levels = m_layout.mip_levels;
  image.array_layers = m_layout.array_layers;
  image.pixels = nx::blob<u8>(nx::cast<usize>(size));
  if (!read_exact(*m_stream, image.pixels.data(), image.pixels.size(),
                  cancellation))
    return false;
  image.subresources.reserve(coarsest_mip - finest_mip + 1u);
  for (u32 mip = finest_mip; mip <= coarsest_mip; ++mip) {
    ImageSubresource subresource = m_layout.subresources[mip];
    subresource.offset -= file_offset;
    image.subresources.push_back(subresource);
  }

  m_next_mip = finest_mip == 0 ? ~u32{0} : finest_mip - 1u;
  if (m_next_mip == ~u32{0})
    m_stream.reset();
  out = std::move(image);
  return true;
}

bool decode_texture(const std::span<const u8> bytes, ImageData &out) {
  TextureAsset asset;
  if (!decode_texture_asset(bytes, asset) ||
      asset.kind != TexturePayloadKind::Native)
    return false;
  out = std::move(asset.native);
  return true;
}

nx::string cooked_texture_path(const nx::string_view source) {
  nx::string out(source);
  out += ".nxb";
  return out;
}

} // namespace nxe::rhi
