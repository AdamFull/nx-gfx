#include "core/rendering/rhi/image/ktx.h"

#include "core/rendering/rhi/image/image_decode.h"

#include <cstring>
#include <limits>

namespace nxe::rhi {
namespace {

using namespace detail;

constexpr u8 KTX_MAGIC[12] = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31,
                              0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
constexpr u32 KTX_ENDIAN_NATIVE = 0x04030201;
constexpr u32 KTX_ENDIAN_REVERSED = 0x01020304;

struct KtxHeader {
  u32 endianness, gl_type, gl_type_size, gl_format, gl_internal_format,
      gl_base_internal_format, pixel_width, pixel_height, pixel_depth,
      array_elements, faces, mip_levels, key_value_bytes;
};
static_assert(sizeof(KtxHeader) == 52, "KTX 1.0 header must be 52 bytes");

struct KtxLevelLayout {
  u32 width = 0;
  u32 height = 0;
  u32 depth = 0;
  u32 rows = 0;
  u64 decoded_row = 0;
  u64 encoded_row = 0;
  u64 decoded_per_layer = 0;
  u64 encoded_per_layer = 0;
};

[[nodiscard]] u32 swap32(const u32 value) noexcept {
  return ((value & 0xFF000000u) >> 24) | ((value & 0x00FF0000u) >> 8) |
         ((value & 0x0000FF00u) << 8) | ((value & 0x000000FFu) << 24);
}

[[nodiscard]] Format format_from_gl(const u32 format) noexcept {
  switch (format) {
  case 0x8058:
    return Format::RGBA8_UNORM;
  case 0x8C43:
    return Format::RGBA8_SRGB;
  case 0x8229:
    return Format::R8_UNORM;
  case 0x822B:
    return Format::RG8_UNORM;
  case 0x822A:
    return Format::R16_UNORM;
  case 0x822D:
    return Format::R16_FLOAT;
  case 0x822F:
    return Format::RG16_FLOAT;
  case 0x881A:
    return Format::RGBA16_FLOAT;
  case 0x822E:
    return Format::R32_FLOAT;
  case 0x8230:
    return Format::RG32_FLOAT;
  case 0x8814:
    return Format::RGBA32_FLOAT;
  case 0x83F0:
    return Format::BC1_RGB_UNORM;
  case 0x8C4C:
    return Format::BC1_RGB_SRGB;
  case 0x83F1:
    return Format::BC1_RGBA_UNORM;
  case 0x8C4D:
    return Format::BC1_RGBA_SRGB;
  case 0x83F2:
    return Format::BC2_RGBA_UNORM;
  case 0x8C4E:
    return Format::BC2_RGBA_SRGB;
  case 0x83F3:
    return Format::BC3_RGBA_UNORM;
  case 0x8C4F:
    return Format::BC3_RGBA_SRGB;
  case 0x8DBB:
    return Format::BC4_R_UNORM;
  case 0x8DBD:
    return Format::BC5_RG_UNORM;
  case 0x8E8F:
    return Format::BC6H_RGB_UFLOAT;
  case 0x8E8E:
    return Format::BC6H_RGB_SFLOAT;
  case 0x8E8C:
    return Format::BC7_RGBA_UNORM;
  case 0x8E8D:
    return Format::BC7_RGBA_SRGB;
  case 0x93B0:
    return Format::ASTC_4x4_UNORM;
  case 0x93D0:
    return Format::ASTC_4x4_SRGB;
  case 0x93B4:
    return Format::ASTC_6x6_UNORM;
  case 0x93D4:
    return Format::ASTC_6x6_SRGB;
  case 0x93B7:
    return Format::ASTC_8x8_UNORM;
  case 0x93D7:
    return Format::ASTC_8x8_SRGB;
  default:
    return Format::Unknown;
  }
}

[[nodiscard]] bool calculate_level_layout(const Format format, const u32 width,
                                          const u32 height, const u32 depth,
                                          KtxLevelLayout &level) noexcept {
  SubresourceLayout decoded;
  if (!subresource_layout(format, width, height, depth, decoded))
    return false;

  u32 block_width = 0;
  u32 block_height = 0;
  format_block_extent(format, block_width, block_height);
  const u64 rows = (nx::cast<u64>(height) + block_height - 1) / block_height;
  if (rows > std::numeric_limits<u32>::max())
    return false;

  level.width = width;
  level.height = height;
  level.depth = depth;
  level.rows = nx::cast<u32>(rows);
  level.decoded_row = decoded.row_pitch;
  level.decoded_per_layer = decoded.size;
  level.encoded_row = decoded.row_pitch;
  if (!format_is_compressed(format) &&
      !checked_align4(level.encoded_row, level.encoded_row))
    return false;

  u64 encoded_slice = 0;
  return checked_mul(level.encoded_row, rows, encoded_slice) &&
         checked_mul(encoded_slice, depth, level.encoded_per_layer);
}

void copy_row(u8 *destination, const u8 *source, const usize bytes,
              const u32 element_size, const bool swapped) noexcept {
  std::memcpy(destination, source, bytes);
  if (!swapped || element_size <= 1)
    return;

  for (usize element = 0; element < bytes; element += element_size) {
    for (u32 left = 0, right = element_size - 1; left < right;
         ++left, --right) {
      const u8 value = destination[element + left];
      destination[element + left] = destination[element + right];
      destination[element + right] = value;
    }
  }
}

} // namespace

bool is_ktx(const std::span<const u8> bytes) noexcept {
  return bytes.size() >= sizeof(KTX_MAGIC) &&
         std::memcmp(bytes.data(), KTX_MAGIC, sizeof(KTX_MAGIC)) == 0;
}

ImageResult decode_ktx(const std::span<const u8> bytes) {
  if (!is_ktx(bytes))
    return nx::err(ImageError::UnknownContainer);

  ImageReader reader{bytes.data(), bytes.size(), sizeof(KTX_MAGIC)};
  KtxHeader header{};
  if (!reader.read(header))
    return nx::err(ImageError::Corrupt);

  if (header.endianness != KTX_ENDIAN_NATIVE &&
      header.endianness != KTX_ENDIAN_REVERSED)
    return nx::err(ImageError::Corrupt);

  const bool swapped = header.endianness == KTX_ENDIAN_REVERSED;
  if (swapped) {
    header.gl_type = swap32(header.gl_type);
    header.gl_type_size = swap32(header.gl_type_size);
    header.gl_format = swap32(header.gl_format);
    header.gl_internal_format = swap32(header.gl_internal_format);
    header.gl_base_internal_format = swap32(header.gl_base_internal_format);
    header.pixel_width = swap32(header.pixel_width);
    header.pixel_height = swap32(header.pixel_height);
    header.pixel_depth = swap32(header.pixel_depth);
    header.array_elements = swap32(header.array_elements);
    header.faces = swap32(header.faces);
    header.mip_levels = swap32(header.mip_levels);
    header.key_value_bytes = swap32(header.key_value_bytes);
  }

  ImageData image;
  image.format = format_from_gl(header.gl_internal_format);
  if (image.format == Format::Unknown)
    return nx::err(ImageError::UnsupportedFormat);

  const bool compressed = format_is_compressed(image.format);
  if ((compressed && (header.gl_type != 0 || header.gl_format != 0 ||
                      header.gl_type_size != 1)) ||
      (!compressed && (header.gl_type == 0 || header.gl_format == 0 ||
                       (header.gl_type_size != 1 && header.gl_type_size != 2 &&
                        header.gl_type_size != 4 && header.gl_type_size != 8))))
    return nx::err(ImageError::Corrupt);

  if (header.pixel_width == 0 || (header.faces != 1 && header.faces != 6) ||
      (header.key_value_bytes & 3u) != 0)
    return nx::err(ImageError::Corrupt);

  const bool cube = header.faces == 6;
  const bool is_3d = header.pixel_depth != 0;
  if ((cube && (header.pixel_height == 0 || is_3d)) ||
      (is_3d && (header.pixel_height == 0 || header.array_elements != 0)))
    return nx::err(ImageError::Corrupt);

  u32 layers = header.array_elements == 0 ? 1 : header.array_elements;
  if (cube && layers > std::numeric_limits<u32>::max() / 6u)
    return nx::err(ImageError::Corrupt);
  if (cube)
    layers *= 6;
  if (layers > MAX_IMAGE_ARRAY_LAYERS)
    return nx::err(ImageError::Corrupt);

  image.width = header.pixel_width;
  image.height = header.pixel_height == 0 ? 1 : header.pixel_height;
  image.depth = is_3d ? header.pixel_depth : 1;
  image.mip_levels = header.mip_levels == 0 ? 1 : header.mip_levels;
  image.array_layers = layers;
  image.dimension = image_dimension(cube, is_3d, layers);

  if (cube && image.width != image.height)
    return nx::err(ImageError::Corrupt);
  if (!reader.skip(header.key_value_bytes))
    return nx::err(ImageError::Corrupt);

  if (image.mip_levels >
      maximum_mip_levels(image.width, image.height, image.depth))
    return nx::err(ImageError::Corrupt);

  KtxLevelLayout levels[32]{};
  ImageReader scan = reader;
  u64 total = 0;
  const bool non_array_cube = cube && header.array_elements == 0;
  for (u32 mip = 0; mip < image.mip_levels; ++mip) {
    u32 image_size = 0;
    if (!scan.read(image_size))
      return nx::err(ImageError::Corrupt);
    if (swapped)
      image_size = swap32(image_size);

    KtxLevelLayout &level = levels[mip];
    const u32 width = mip_dimension(image.width, mip);
    const u32 height = mip_dimension(image.height, mip);
    const u32 depth = is_3d ? mip_dimension(image.depth, mip) : 1u;
    if (!calculate_level_layout(image.format, width, height, depth, level) ||
        level.decoded_row % header.gl_type_size != 0)
      return nx::err(ImageError::Corrupt);

    u64 expected_image_size = level.encoded_per_layer;
    if (!non_array_cube &&
        !checked_mul(expected_image_size, layers, expected_image_size))
      return nx::err(ImageError::Corrupt);
    if (expected_image_size > std::numeric_limits<u32>::max() ||
        image_size != expected_image_size)
      return nx::err(ImageError::Corrupt);

    for (u32 layer = 0; layer < layers; ++layer) {
      if (level.encoded_per_layer > scan.remaining() ||
          !scan.skip(nx::cast<usize>(level.encoded_per_layer)))
        return nx::err(ImageError::Corrupt);
      if (non_array_cube && !scan.align4())
        return nx::err(ImageError::Corrupt);
    }
    if (!scan.align4())
      return nx::err(ImageError::Corrupt);

    u64 decoded_level = 0;
    if (!checked_mul(level.decoded_per_layer, layers, decoded_level) ||
        !checked_add(total, decoded_level, total))
      return nx::err(ImageError::Corrupt);
  }

  if (total > std::numeric_limits<usize>::max())
    return nx::err(ImageError::Corrupt);
  image.pixels = nx::blob<u8>(nx::cast<usize>(total));
  image.subresources.reserve(
      nx::cast<usize>(nx::cast<u64>(layers) * image.mip_levels));

  // KTX stores mip-major data. Decode its aligned rows into the RHI's tightly
  // packed subresources after validating the entire encoded layout.
  u64 offset = 0;
  for (u32 mip = 0; mip < image.mip_levels; ++mip) {
    if (!reader.skip(sizeof(u32)))
      return nx::err(ImageError::Corrupt);

    const KtxLevelLayout &level = levels[mip];
    for (u32 layer = 0; layer < layers; ++layer) {
      const u64 destination_layer = offset + level.decoded_per_layer * layer;
      for (u32 z = 0; z < level.depth; ++z) {
        for (u32 row = 0; row < level.rows; ++row) {
          const u64 source_offset =
              level.encoded_row * (nx::cast<u64>(z) * level.rows + row);
          const u64 destination_offset =
              destination_layer +
              level.decoded_row * (nx::cast<u64>(z) * level.rows + row);
          copy_row(image.pixels.data() + nx::cast<usize>(destination_offset),
                   reader.data + reader.pos + nx::cast<usize>(source_offset),
                   nx::cast<usize>(level.decoded_row), header.gl_type_size,
                   swapped);
        }
      }
      if (!reader.skip(nx::cast<usize>(level.encoded_per_layer)) ||
          (non_array_cube && !reader.align4()))
        return nx::err(ImageError::Corrupt);
      append_subresource(image, destination_layer, mip, layer, level.width,
                         level.height, level.depth);
    }
    if (!reader.align4())
      return nx::err(ImageError::Corrupt);
    offset += level.decoded_per_layer * layers;
  }
  return image;
}

} // namespace nxe::rhi
