#include "rendering/rhi/image/ktx2.h"

#include <zstd.h>

#include <cstring>
#include <limits>

namespace nxe::rhi {
namespace {

constexpr u8 KTX2_MAGIC[12] = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32,
                               0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
constexpr usize HEADER_SIZE = 80;
constexpr usize LEVEL_INDEX_OFFSET = HEADER_SIZE;
constexpr usize LEVEL_INDEX_SIZE = 24;
constexpr u8 KHR_DF_MODEL_ETC1S = 163;
constexpr u8 KHR_DF_MODEL_UASTC = 166;
constexpr u8 KHR_DF_TRANSFER_LINEAR = 1;
constexpr u8 KHR_DF_TRANSFER_SRGB = 2;

struct Header {
  u32 vk_format = 0;
  u32 type_size = 0;
  u32 pixel_width = 0;
  u32 pixel_height = 0;
  u32 pixel_depth = 0;
  u32 layer_count = 0;
  u32 face_count = 0;
  u32 level_count = 0;
  u32 supercompression = 0;
  u32 dfd_offset = 0;
  u32 dfd_length = 0;
  u32 kvd_offset = 0;
  u32 kvd_length = 0;
  u64 sgd_offset = 0;
  u64 sgd_length = 0;
};

struct Level {
  u64 offset = 0;
  u64 length = 0;
  u64 decoded_length = 0;
};

struct Parsed {
  Header header;
  Ktx2Info info;
  nx::vector<Level> levels;
};

[[nodiscard]] u32 read_u32(const std::span<const u8> bytes,
                           const usize offset) noexcept {
  return nx::cast<u32>(bytes[offset]) |
         (nx::cast<u32>(bytes[offset + 1]) << 8) |
         (nx::cast<u32>(bytes[offset + 2]) << 16) |
         (nx::cast<u32>(bytes[offset + 3]) << 24);
}

[[nodiscard]] u64 read_u64(const std::span<const u8> bytes,
                           const usize offset) noexcept {
  return nx::cast<u64>(read_u32(bytes, offset)) |
         (nx::cast<u64>(read_u32(bytes, offset + 4)) << 32);
}

[[nodiscard]] bool range_valid(const u64 offset, const u64 length,
                               const usize size) noexcept {
  return offset <= size && length <= nx::cast<u64>(size) - offset;
}

[[nodiscard]] bool checked_add(const u64 a, const u64 b, u64 &result) noexcept {
  if (b > std::numeric_limits<u64>::max() - a)
    return false;
  result = a + b;
  return true;
}

[[nodiscard]] bool checked_mul(const u64 a, const u64 b, u64 &result) noexcept {
  if (a != 0 && b > std::numeric_limits<u64>::max() / a)
    return false;
  result = a * b;
  return true;
}

[[nodiscard]] u32 dim_at(const u32 base, const u32 mip) noexcept {
  return mip >= 32 ? 1u : nx::max(1u, base >> mip);
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

// KTX2 intentionally names its primary format field after Vulkan. Keeping the
// numeric mapping local avoids leaking Vulkan API types into the portable image
// API while retaining the specification's stable enum values.
[[nodiscard]] Format format_from_vk(const u32 format) noexcept {
  switch (format) {
  case 9:
    return Format::R8_UNORM;
  case 10:
    return Format::R8_SNORM;
  case 13:
    return Format::R8_UINT;
  case 14:
    return Format::R8_SINT;
  case 16:
    return Format::RG8_UNORM;
  case 20:
    return Format::RG8_UINT;
  case 37:
    return Format::RGBA8_UNORM;
  case 38:
    return Format::RGBA8_SNORM;
  case 41:
    return Format::RGBA8_UINT;
  case 43:
    return Format::RGBA8_SRGB;
  case 44:
    return Format::BGRA8_UNORM;
  case 50:
    return Format::BGRA8_SRGB;
  case 64:
    return Format::RGB10A2_UNORM;
  case 70:
    return Format::R16_UNORM;
  case 74:
    return Format::R16_UINT;
  case 75:
    return Format::R16_SINT;
  case 76:
    return Format::R16_FLOAT;
  case 77:
    return Format::RG16_UNORM;
  case 83:
    return Format::RG16_FLOAT;
  case 91:
    return Format::RGBA16_UNORM;
  case 97:
    return Format::RGBA16_FLOAT;
  case 98:
    return Format::R32_UINT;
  case 99:
    return Format::R32_SINT;
  case 100:
    return Format::R32_FLOAT;
  case 101:
    return Format::RG32_UINT;
  case 103:
    return Format::RG32_FLOAT;
  case 107:
    return Format::RGBA32_UINT;
  case 109:
    return Format::RGBA32_FLOAT;
  case 122:
    return Format::RG11B10_FLOAT;
  case 124:
    return Format::D16_UNORM;
  case 126:
    return Format::D32_FLOAT;
  case 129:
    return Format::D24_UNORM_S8_UINT;
  case 130:
    return Format::D32_FLOAT_S8_UINT;
  case 131:
    return Format::BC1_RGB_UNORM;
  case 132:
    return Format::BC1_RGB_SRGB;
  case 133:
    return Format::BC1_RGBA_UNORM;
  case 134:
    return Format::BC1_RGBA_SRGB;
  case 135:
    return Format::BC2_RGBA_UNORM;
  case 136:
    return Format::BC2_RGBA_SRGB;
  case 137:
    return Format::BC3_RGBA_UNORM;
  case 138:
    return Format::BC3_RGBA_SRGB;
  case 139:
    return Format::BC4_R_UNORM;
  case 141:
    return Format::BC5_RG_UNORM;
  case 143:
    return Format::BC6H_RGB_UFLOAT;
  case 144:
    return Format::BC6H_RGB_SFLOAT;
  case 145:
    return Format::BC7_RGBA_UNORM;
  case 146:
    return Format::BC7_RGBA_SRGB;
  case 157:
    return Format::ASTC_4x4_UNORM;
  case 158:
    return Format::ASTC_4x4_SRGB;
  case 165:
    return Format::ASTC_6x6_UNORM;
  case 166:
    return Format::ASTC_6x6_SRGB;
  case 171:
    return Format::ASTC_8x8_UNORM;
  case 172:
    return Format::ASTC_8x8_SRGB;
  default:
    return Format::Unknown;
  }
}

[[nodiscard]] u32 expected_type_size(const Format format) noexcept {
  if (format_is_compressed(format))
    return 1;
  switch (format) {
  case Format::R16_UNORM:
  case Format::R16_FLOAT:
  case Format::R16_UINT:
  case Format::R16_SINT:
  case Format::RG16_FLOAT:
  case Format::RG16_UNORM:
  case Format::RGBA16_FLOAT:
  case Format::RGBA16_UNORM:
  case Format::D16_UNORM:
    return 2;
  case Format::R32_FLOAT:
  case Format::R32_UINT:
  case Format::R32_SINT:
  case Format::RG32_FLOAT:
  case Format::RG32_UINT:
  case Format::RGBA32_FLOAT:
  case Format::RGBA32_UINT:
  case Format::RGB10A2_UNORM:
  case Format::RG11B10_FLOAT:
  case Format::D32_FLOAT:
  case Format::D24_UNORM_S8_UINT:
  case Format::D32_FLOAT_S8_UINT:
    return 4;
  default:
    return 1;
  }
}

[[nodiscard]] bool metadata_supported(const std::span<const u8> bytes,
                                      const Header &header,
                                      const TextureDimension dimension,
                                      const bool one_dimensional) {
  if (header.kvd_length == 0)
    return header.kvd_offset == 0;
  if ((header.kvd_offset & 3u) != 0 ||
      !range_valid(header.kvd_offset, header.kvd_length, bytes.size()))
    return false;

  usize cursor = header.kvd_offset;
  const usize end = cursor + header.kvd_length;
  bool saw_orientation = false;
  bool saw_swizzle = false;
  nx::string_view previous_key;
  while (cursor < end) {
    if (end - cursor < sizeof(u32))
      return false;
    const u32 pair_size = read_u32(bytes, cursor);
    cursor += sizeof(u32);
    if (pair_size < 2 || pair_size > end - cursor)
      return false;
    const usize pair_end = cursor + pair_size;
    usize separator = cursor;
    while (separator < pair_end && bytes[separator] != 0)
      ++separator;
    if (separator == cursor || separator == pair_end)
      return false;
    const nx::string_view key(
        reinterpret_cast<const char *>(bytes.data() + cursor),
        separator - cursor);
    if (!previous_key.empty() && key.compare(previous_key) <= 0)
      return false;
    previous_key = key;
    const std::span<const u8> value =
        bytes.subspan(separator + 1, pair_end - separator - 1);

    if (key == "KTXorientation") {
      if (saw_orientation)
        return false;
      saw_orientation = true;
      usize length = value.size();
      if (length != 0 && value[length - 1] == 0)
        --length;
      const nx::string_view orientation(
          reinterpret_cast<const char *>(value.data()), length);
      const nx::string_view expected = one_dimensional ? nx::string_view("r")
                                       : dimension == TextureDimension::Tex3D
                                           ? nx::string_view("rdi")
                                           : nx::string_view("rd");
      if (orientation != expected)
        return false;
    } else if (key == "KTXswizzle") {
      if (saw_swizzle)
        return false;
      saw_swizzle = true;
      usize length = value.size();
      if (length != 0 && value[length - 1] == 0)
        --length;
      if (length != 4 ||
          std::memcmp(value.data(), "rgba", sizeof("rgba") - 1) != 0)
        return false;
    } else if (header.vk_format != 0 &&
               (key == "KTXglFormat" || key == "KTXdxgiFormat__" ||
                key == "KTXmetalPixelFormat")) {
      return false;
    }

    cursor = pair_end;
    const usize padding = (4u - (pair_size & 3u)) & 3u;
    if (padding > end - cursor)
      return false;
    cursor += padding;
  }
  return cursor == end;
}

[[nodiscard]] nx::result<Parsed, ImageError>
parse(const std::span<const u8> bytes) noexcept {
  if (!is_ktx2(bytes))
    return nx::err(ImageError::UnknownContainer);
  if (bytes.size() < HEADER_SIZE)
    return nx::err(ImageError::Corrupt);

  Parsed parsed;
  Header &h = parsed.header;
  h.vk_format = read_u32(bytes, 12);
  h.type_size = read_u32(bytes, 16);
  h.pixel_width = read_u32(bytes, 20);
  h.pixel_height = read_u32(bytes, 24);
  h.pixel_depth = read_u32(bytes, 28);
  h.layer_count = read_u32(bytes, 32);
  h.face_count = read_u32(bytes, 36);
  h.level_count = read_u32(bytes, 40);
  h.supercompression = read_u32(bytes, 44);
  h.dfd_offset = read_u32(bytes, 48);
  h.dfd_length = read_u32(bytes, 52);
  h.kvd_offset = read_u32(bytes, 56);
  h.kvd_length = read_u32(bytes, 60);
  h.sgd_offset = read_u64(bytes, 64);
  h.sgd_length = read_u64(bytes, 72);

  const u32 stored_levels = nx::max(1u, h.level_count);
  u64 index_bytes = 0;
  u64 index_end = 0;
  if (!checked_mul(stored_levels, LEVEL_INDEX_SIZE, index_bytes) ||
      !checked_add(LEVEL_INDEX_OFFSET, index_bytes, index_end) ||
      index_end > bytes.size() || h.pixel_width == 0 ||
      (h.face_count != 1 && h.face_count != 6) || h.type_size == 0 ||
      h.supercompression > nx::cast<u32>(Ktx2Supercompression::Zlib))
    return nx::err(ImageError::Corrupt);

  const bool cube = h.face_count == 6;
  const bool is_3d = h.pixel_depth != 0;
  if ((cube && (h.pixel_height == 0 || is_3d)) ||
      (is_3d && (h.pixel_height == 0 || h.layer_count != 0)) ||
      (h.pixel_height == 0 && h.pixel_depth != 0))
    return nx::err(ImageError::Corrupt);

  u32 layers = nx::max(1u, h.layer_count);
  if (cube && layers > std::numeric_limits<u32>::max() / 6u)
    return nx::err(ImageError::Corrupt);
  if (cube)
    layers *= 6;
  if (layers > MAX_IMAGE_ARRAY_LAYERS)
    return nx::err(ImageError::Corrupt);

  Ktx2Info &info = parsed.info;
  info.width = h.pixel_width;
  info.height = nx::max(1u, h.pixel_height);
  info.depth = nx::max(1u, h.pixel_depth);
  info.mip_levels = stored_levels;
  info.array_layers = layers;
  info.dimension = cube                 ? TextureDimension::TexCube
                   : is_3d              ? TextureDimension::Tex3D
                   : h.layer_count != 0 ? TextureDimension::Tex2DArray
                                        : TextureDimension::Tex2D;
  info.supercompression = nx::cast<Ktx2Supercompression>(h.supercompression);
  if (cube && info.width != info.height)
    return nx::err(ImageError::Corrupt);
  if (info.mip_levels > maximum_mip_levels(info.width, info.height, info.depth))
    return nx::err(ImageError::Corrupt);

  if (h.dfd_length < 28 || (h.dfd_offset & 3u) != 0 ||
      !range_valid(h.dfd_offset, h.dfd_length, bytes.size()) ||
      read_u32(bytes, h.dfd_offset) != h.dfd_length)
    return nx::err(ImageError::Corrupt);
  const u32 descriptor_size = read_u32(bytes, h.dfd_offset + 8) >> 16;
  if (descriptor_size < 24 || descriptor_size > h.dfd_length - 4)
    return nx::err(ImageError::Corrupt);
  const u8 color_model = bytes[h.dfd_offset + 12];
  const u8 transfer = bytes[h.dfd_offset + 14];
  if (transfer != KHR_DF_TRANSFER_LINEAR && transfer != KHR_DF_TRANSFER_SRGB)
    return nx::err(ImageError::UnsupportedFormat);
  info.srgb = transfer == KHR_DF_TRANSFER_SRGB;

  if (h.vk_format == 0) {
    if (color_model == KHR_DF_MODEL_ETC1S &&
        info.supercompression == Ktx2Supercompression::BasisLz) {
      info.payload = Ktx2PayloadKind::BasisEtc1s;
    } else if (color_model == KHR_DF_MODEL_UASTC &&
               (info.supercompression == Ktx2Supercompression::None ||
                info.supercompression == Ktx2Supercompression::Zstd)) {
      info.payload = Ktx2PayloadKind::BasisUastc;
    } else {
      return nx::err(ImageError::UnsupportedFormat);
    }
  } else {
    info.payload = Ktx2PayloadKind::Native;
    info.format = format_from_vk(h.vk_format);
    if (info.format == Format::Unknown ||
        (info.supercompression != Ktx2Supercompression::None &&
         info.supercompression != Ktx2Supercompression::Zstd))
      return nx::err(ImageError::UnsupportedFormat);
    if (h.level_count == 0 && format_is_compressed(info.format))
      return nx::err(ImageError::Corrupt);
    if (h.type_size != expected_type_size(info.format))
      return nx::err(ImageError::Corrupt);
    if (info.srgb != format_is_srgb(info.format))
      return nx::err(ImageError::Corrupt);

    u32 block_width = 0;
    u32 block_height = 0;
    format_block_extent(info.format, block_width, block_height);
    const u8 dfd_block_width = nx::cast<u8>(bytes[h.dfd_offset + 16] + 1u);
    const u8 dfd_block_height = nx::cast<u8>(bytes[h.dfd_offset + 17] + 1u);
    const u8 dfd_block_depth = nx::cast<u8>(bytes[h.dfd_offset + 18] + 1u);
    const u8 bytes_per_block = bytes[h.dfd_offset + 20];
    if (dfd_block_width != block_width || dfd_block_height != block_height ||
        dfd_block_depth != 1 ||
        (bytes_per_block != 0 &&
         bytes_per_block != format_texel_size(info.format)))
      return nx::err(ImageError::Corrupt);
  }

  if (!metadata_supported(bytes, h, info.dimension, h.pixel_height == 0))
    return nx::err(ImageError::UnsupportedFormat);
  if ((h.sgd_length == 0) != (h.sgd_offset == 0) ||
      !range_valid(h.sgd_offset, h.sgd_length, bytes.size()))
    return nx::err(ImageError::Corrupt);
  if (info.payload == Ktx2PayloadKind::BasisEtc1s && h.sgd_length == 0)
    return nx::err(ImageError::Corrupt);
  if (info.payload != Ktx2PayloadKind::BasisEtc1s && h.sgd_length != 0)
    return nx::err(ImageError::Corrupt);

  parsed.levels.reserve(stored_levels);
  u64 decoded_total = 0;
  for (u32 mip = 0; mip < stored_levels; ++mip) {
    const usize base =
        LEVEL_INDEX_OFFSET + nx::cast<usize>(mip) * LEVEL_INDEX_SIZE;
    Level level{read_u64(bytes, base), read_u64(bytes, base + 8),
                read_u64(bytes, base + 16)};
    u64 alignment = 1;
    if (info.supercompression == Ktx2Supercompression::None) {
      alignment = info.payload == Ktx2PayloadKind::BasisUastc
                      ? 16u
                      : nx::max<u64>(4u, format_texel_size(info.format));
    }
    if (level.offset % alignment != 0 || level.length == 0 ||
        !range_valid(level.offset, level.length, bytes.size()))
      return nx::err(ImageError::Corrupt);

    if (info.payload == Ktx2PayloadKind::Native) {
      const u32 width = dim_at(info.width, mip);
      const u32 height = dim_at(info.height, mip);
      const u32 depth = info.dimension == TextureDimension::Tex3D
                            ? dim_at(info.depth, mip)
                            : 1u;
      u64 expected = format_subresource_size(info.format, width, height, depth);
      if (!checked_mul(expected,
                       info.dimension == TextureDimension::Tex3D ? 1u : layers,
                       expected) ||
          expected == 0 || level.decoded_length != expected ||
          (info.supercompression == Ktx2Supercompression::None &&
           level.length != expected) ||
          !checked_add(decoded_total, expected, decoded_total))
        return nx::err(ImageError::Corrupt);
    } else if (info.payload == Ktx2PayloadKind::BasisEtc1s &&
               level.decoded_length != 0) {
      return nx::err(ImageError::Corrupt);
    }
    parsed.levels.push_back(level);
  }

  if (info.payload == Ktx2PayloadKind::Native &&
      decoded_total > std::numeric_limits<usize>::max())
    return nx::err(ImageError::Corrupt);
  return parsed;
}

} // namespace

bool is_ktx2(const std::span<const u8> bytes) noexcept {
  return bytes.size() >= sizeof(KTX2_MAGIC) &&
         std::memcmp(bytes.data(), KTX2_MAGIC, sizeof(KTX2_MAGIC)) == 0;
}

nx::result<Ktx2Info, ImageError>
inspect_ktx2(const std::span<const u8> bytes) noexcept {
  auto parsed = parse(bytes);
  if (!parsed)
    return nx::err(parsed.error());
  return parsed->info;
}

nx::result<ImageData, ImageError>
decode_native_ktx2(const std::span<const u8> bytes) {
  auto result = parse(bytes);
  if (!result)
    return nx::err(result.error());
  Parsed parsed = std::move(result.value());
  const Ktx2Info &info = parsed.info;
  if (info.payload != Ktx2PayloadKind::Native)
    return nx::err(ImageError::UnsupportedFormat);

  u64 total = 0;
  for (const Level &level : parsed.levels)
    if (!checked_add(total, level.decoded_length, total))
      return nx::err(ImageError::Corrupt);
  if (total == 0 || total > std::numeric_limits<usize>::max())
    return nx::err(ImageError::Corrupt);

  ImageData image;
  image.format = info.format;
  image.dimension = info.dimension;
  image.width = info.width;
  image.height = info.height;
  image.depth = info.depth;
  image.mip_levels = info.mip_levels;
  image.array_layers = info.array_layers;
  image.pixels = nx::blob<u8>(nx::cast<usize>(total));
  image.subresources.reserve(image.dimension == TextureDimension::Tex3D
                                 ? image.mip_levels
                                 : nx::cast<usize>(image.mip_levels) *
                                       image.array_layers);

  u64 destination_offset = 0;
  for (u32 mip = 0; mip < image.mip_levels; ++mip) {
    const Level &level = parsed.levels[mip];
    u8 *const destination =
        image.pixels.data() + nx::cast<usize>(destination_offset);
    if (info.supercompression == Ktx2Supercompression::None) {
      std::memcpy(destination, bytes.data() + nx::cast<usize>(level.offset),
                  nx::cast<usize>(level.length));
    } else {
      const usize decoded =
          ZSTD_decompress(destination, nx::cast<usize>(level.decoded_length),
                          bytes.data() + nx::cast<usize>(level.offset),
                          nx::cast<usize>(level.length));
      if (ZSTD_isError(decoded) || decoded != level.decoded_length)
        return nx::err(ImageError::Corrupt);
    }

    const u32 width = dim_at(image.width, mip);
    const u32 height = dim_at(image.height, mip);
    const u32 depth = image.dimension == TextureDimension::Tex3D
                          ? dim_at(image.depth, mip)
                          : 1u;
    const u64 per_layer =
        format_subresource_size(image.format, width, height, depth);
    const u32 layers =
        image.dimension == TextureDimension::Tex3D ? 1u : image.array_layers;
    for (u32 layer = 0; layer < layers; ++layer) {
      image.subresources.push_back(
          {destination_offset + per_layer * layer, per_layer, mip, layer, width,
           height, depth,
           nx::cast<u32>(format_row_pitch(image.format, width))});
    }
    destination_offset += level.decoded_length;
  }

  if (!image.valid())
    return nx::err(ImageError::Corrupt);
  return image;
}

} // namespace nxe::rhi
