
#include "core/rendering/rhi/image/image.h"

#include "core/foundation/diagnostics/log.h"
#include "core/foundation/diagnostics/profiler.h"

#include <stb_image.h>
#include <webp/decode.h>

#include <cstring>
#include <limits>
#include <memory>

namespace nxe::rhi {
namespace {

const nx::log::Category log_img = nx::log::category("image");

using Result = nx::result<ImageData, ImageError>;

/// Sequential cursor over the input bytes, bounds-checked on every read.
struct Reader {
  const u8 *data = nullptr;
  usize size = 0;
  usize pos = 0;

  [[nodiscard]] bool ok(const usize n) const noexcept {
    return pos <= size && n <= size - pos;
  }
  [[nodiscard]] usize remaining() const noexcept { return size - pos; }

  bool read(void *dst, const usize n) noexcept {
    if (!ok(n))
      return false;
    std::memcpy(dst, data + pos, n);
    pos += n;
    return true;
  }
  template <typename T> bool read(T &value) noexcept {
    return read(&value, sizeof(T));
  }
  bool skip(const usize n) noexcept {
    if (!ok(n))
      return false;
    pos += n;
    return true;
  }
  bool align4() noexcept { return skip((4u - (pos & 3u)) & 3u); }
};

[[nodiscard]] u32 dim_at(const u32 base, const u32 mip) noexcept {
  const u32 v = base >> mip;
  return v == 0 ? 1u : v;
}

[[nodiscard]] bool checked_add(const u64 a, const u64 b, u64 &value) noexcept {
  if (b > std::numeric_limits<u64>::max() - a)
    return false;
  value = a + b;
  return true;
}

[[nodiscard]] bool checked_mul(const u64 a, const u64 b, u64 &value) noexcept {
  if (a != 0 && b > std::numeric_limits<u64>::max() / a)
    return false;
  value = a * b;
  return true;
}

[[nodiscard]] bool checked_align4(const u64 value, u64 &aligned) noexcept {
  return checked_add(value, (4u - (value & 3u)) & 3u, aligned);
}

struct SubresourceLayout {
  u64 row_pitch = 0;
  u64 size = 0;
};

[[nodiscard]] bool subresource_layout(const Format fmt, const u32 w,
                                      const u32 h, const u32 d,
                                      SubresourceLayout &layout) noexcept {
  if (w == 0 || h == 0 || d == 0)
    return false;

  u32 block_w = 0;
  u32 block_h = 0;
  format_block_extent(fmt, block_w, block_h);
  const u64 texel_size = format_texel_size(fmt);
  if (block_w == 0 || block_h == 0 || texel_size == 0)
    return false;

  const u64 blocks_x = (nx::cast<u64>(w) + block_w - 1) / block_w;
  const u64 blocks_y = (nx::cast<u64>(h) + block_h - 1) / block_h;
  if (!checked_mul(blocks_x, texel_size, layout.row_pitch) ||
      layout.row_pitch > std::numeric_limits<u32>::max())
    return false;

  u64 slice_size = 0;
  return checked_mul(layout.row_pitch, blocks_y, slice_size) &&
         checked_mul(slice_size, d, layout.size);
}

[[nodiscard]] u32 maximum_mip_levels(const u32 w, const u32 h,
                                     const u32 d) noexcept {
  u32 largest = nx::max(w, nx::max(h, d));
  u32 levels = 0;
  while (largest != 0) {
    ++levels;
    largest >>= 1;
  }
  return levels;
}

[[nodiscard]] bool image_total_size(const Format fmt, const u32 w, const u32 h,
                                    const u32 d, const u32 mips,
                                    const u32 layers, const bool is_3d,
                                    u64 &total) noexcept {
  if (w == 0 || h == 0 || d == 0 || mips == 0 || layers == 0 ||
      mips > maximum_mip_levels(w, h, d))
    return false;

  total = 0;
  for (u32 mip = 0; mip < mips; ++mip) {
    SubresourceLayout layout;
    if (!subresource_layout(fmt, dim_at(w, mip), dim_at(h, mip),
                            is_3d ? dim_at(d, mip) : 1u, layout) ||
        !checked_add(total, layout.size, total))
      return false;
  }
  return checked_mul(total, layers, total);
}

void push_subresource(ImageData &img, const u64 offset, const u32 mip,
                      const u32 layer, const u32 w, const u32 h, const u32 d) {
  ImageSubresource sr;
  sr.offset = offset;
  sr.size = format_subresource_size(img.format, w, h, d);
  sr.mip = mip;
  sr.layer = layer;
  sr.width = w;
  sr.height = h;
  sr.depth = d;
  sr.row_pitch = nx::cast<u32>(format_row_pitch(img.format, w));
  img.subresources.push_back(sr);
}

TextureDimension pick_dimension(const bool cube, const bool is_3d,
                                const u32 layers) noexcept {
  if (cube)
    return TextureDimension::TexCube;
  if (is_3d)
    return TextureDimension::Tex3D;
  return layers > 1 ? TextureDimension::Tex2DArray : TextureDimension::Tex2D;
}

constexpr u8 DDS_MAGIC[4] = {'D', 'D', 'S', ' '};
constexpr u8 WEBP_RIFF[4] = {'R', 'I', 'F', 'F'};
constexpr u8 WEBP_FORM[4] = {'W', 'E', 'B', 'P'};

[[nodiscard]] bool is_webp(const std::span<const u8> bytes) noexcept {
  return bytes.size() >= 12 &&
         std::memcmp(bytes.data(), WEBP_RIFF, sizeof(WEBP_RIFF)) == 0 &&
         std::memcmp(bytes.data() + 8, WEBP_FORM, sizeof(WEBP_FORM)) == 0;
}

constexpr u32 make_fourcc(const char a, const char b, const char c,
                          const char d) noexcept {
  return nx::cast<u32>(nx::cast<u8>(a)) |
         (nx::cast<u32>(nx::cast<u8>(b)) << 8) |
         (nx::cast<u32>(nx::cast<u8>(c)) << 16) |
         (nx::cast<u32>(nx::cast<u8>(d)) << 24);
}

enum : u32 {
  DDPF_ALPHAPIXELS = 0x1,
  DDPF_FOURCC = 0x4,
  DDPF_RGB = 0x40,
  DDSCAPS2_CUBEMAP = 0x200,
  DDSCAPS2_VOLUME = 0x200000,
  DDS_DIM_TEXTURE3D = 4,
  DDS_MISC_TEXTURECUBE = 0x4,
};

struct DdsPixelFormat {
  u32 size, flags, fourcc, rgb_bit_count, r_mask, g_mask, b_mask, a_mask;
};

struct DdsHeader {
  u32 size, flags, height, width, pitch_or_linear_size, depth, mip_count;
  u32 reserved1[11];
  DdsPixelFormat ddspf;
  u32 caps, caps2, caps3, caps4, reserved2;
};
static_assert(sizeof(DdsHeader) == 124, "DDS header must be 124 bytes");

struct DdsHeaderDX10 {
  u32 dxgi_format, resource_dimension, misc_flag, array_size, misc_flags2;
};
static_assert(sizeof(DdsHeaderDX10) == 20, "DX10 header must be 20 bytes");

Format format_from_dxgi(const u32 dxgi) noexcept {
  switch (dxgi) {
  case 28:
    return Format::RGBA8_UNORM;
  case 29:
    return Format::RGBA8_SRGB;
  case 31:
    return Format::RGBA8_SNORM;
  case 30:
    return Format::RGBA8_UINT;
  case 87:
    return Format::BGRA8_UNORM;
  case 91:
    return Format::BGRA8_SRGB;
  case 61:
    return Format::R8_UNORM;
  case 49:
    return Format::RG8_UNORM;
  case 56:
    return Format::R16_UNORM;
  case 54:
    return Format::R16_FLOAT;
  case 35:
    return Format::RG16_UNORM;
  case 34:
    return Format::RG16_FLOAT;
  case 11:
    return Format::RGBA16_UNORM;
  case 10:
    return Format::RGBA16_FLOAT;
  case 41:
    return Format::R32_FLOAT;
  case 16:
    return Format::RG32_FLOAT;
  case 2:
    return Format::RGBA32_FLOAT;
  case 24:
    return Format::RGB10A2_UNORM;
  case 26:
    return Format::RG11B10_FLOAT;
  case 71:
    return Format::BC1_RGBA_UNORM;
  case 72:
    return Format::BC1_RGBA_SRGB;
  case 74:
    return Format::BC2_RGBA_UNORM;
  case 75:
    return Format::BC2_RGBA_SRGB;
  case 77:
    return Format::BC3_RGBA_UNORM;
  case 78:
    return Format::BC3_RGBA_SRGB;
  case 80:
    return Format::BC4_R_UNORM;
  case 83:
    return Format::BC5_RG_UNORM;
  case 95:
    return Format::BC6H_RGB_UFLOAT;
  case 96:
    return Format::BC6H_RGB_SFLOAT;
  case 98:
    return Format::BC7_RGBA_UNORM;
  case 99:
    return Format::BC7_RGBA_SRGB;
  default:
    return Format::Unknown;
  }
}

Format format_from_dds_legacy(const DdsPixelFormat &pf) noexcept {
  if (pf.flags & DDPF_FOURCC) {
    if (pf.fourcc == make_fourcc('D', 'X', 'T', '1'))
      return Format::BC1_RGBA_UNORM;
    if (pf.fourcc == make_fourcc('D', 'X', 'T', '3'))
      return Format::BC2_RGBA_UNORM;
    if (pf.fourcc == make_fourcc('D', 'X', 'T', '5'))
      return Format::BC3_RGBA_UNORM;
    if (pf.fourcc == make_fourcc('A', 'T', 'I', '1') ||
        pf.fourcc == make_fourcc('B', 'C', '4', 'U'))
      return Format::BC4_R_UNORM;
    if (pf.fourcc == make_fourcc('A', 'T', 'I', '2') ||
        pf.fourcc == make_fourcc('B', 'C', '5', 'U'))
      return Format::BC5_RG_UNORM;
    return Format::Unknown;
  }
  if (pf.flags & DDPF_RGB && pf.rgb_bit_count == 32) {
    if (pf.r_mask == 0x000000FF)
      return Format::RGBA8_UNORM;
    if (pf.r_mask == 0x00FF0000)
      return Format::BGRA8_UNORM;
  }
  return Format::Unknown;
}

Result parse_dds(Reader &r) {
  DdsHeader header{};
  if (!r.read(header) || header.size != sizeof(DdsHeader) ||
      header.ddspf.size != sizeof(DdsPixelFormat))
    return nx::err(ImageError::Corrupt);

  ImageData img;
  bool cube = false;
  bool is_3d = false;
  u32 layers = 1;

  const bool dx10 = (header.ddspf.flags & DDPF_FOURCC) &&
                    header.ddspf.fourcc == make_fourcc('D', 'X', '1', '0');
  if (dx10) {
    DdsHeaderDX10 ext{};
    if (!r.read(ext))
      return nx::err(ImageError::Corrupt);
    if (ext.array_size == 0 || (ext.resource_dimension != 3 &&
                                ext.resource_dimension != DDS_DIM_TEXTURE3D))
      return nx::err(ImageError::Corrupt);
    img.format = format_from_dxgi(ext.dxgi_format);
    layers = ext.array_size;
    cube = (ext.misc_flag & DDS_MISC_TEXTURECUBE) != 0;
    is_3d = ext.resource_dimension == DDS_DIM_TEXTURE3D;
    if (is_3d && (cube || layers != 1))
      return nx::err(ImageError::Corrupt);
  } else {
    img.format = format_from_dds_legacy(header.ddspf);
    cube = (header.caps2 & DDSCAPS2_CUBEMAP) != 0;
    is_3d = (header.caps2 & DDSCAPS2_VOLUME) != 0 || header.depth > 1;
  }

  if (img.format == Format::Unknown)
    return nx::err(ImageError::UnsupportedFormat);

  if (cube && layers > std::numeric_limits<u32>::max() / 6u)
    return nx::err(ImageError::Corrupt);
  if (cube)
    layers *= 6;
  if (layers > MAX_IMAGE_ARRAY_LAYERS)
    return nx::err(ImageError::Corrupt);

  img.width = header.width;
  img.height = header.height;
  img.depth = is_3d ? header.depth : 1;
  img.mip_levels = header.mip_count == 0 ? 1 : header.mip_count;
  img.array_layers = layers;
  img.dimension = pick_dimension(cube, is_3d, layers);

  if (cube && img.width != img.height)
    return nx::err(ImageError::Corrupt);

  u64 total = 0;
  if (!image_total_size(img.format, img.width, img.height, img.depth,
                        img.mip_levels, layers, is_3d, total) ||
      total > r.remaining())
    return nx::err(ImageError::Corrupt);

  img.pixels = nx::blob<u8>(nx::cast<usize>(total));
  img.subresources.reserve(
      nx::cast<usize>(nx::cast<u64>(layers) * img.mip_levels));

  u64 offset = 0;
  for (u32 layer = 0; layer < layers; ++layer) {
    for (u32 mip = 0; mip < img.mip_levels; ++mip) {
      const u32 w = dim_at(img.width, mip);
      const u32 h = dim_at(img.height, mip);
      const u32 d = is_3d ? dim_at(img.depth, mip) : 1u;
      const u64 sz = format_subresource_size(img.format, w, h, d);
      std::memcpy(img.pixels.data() + nx::cast<usize>(offset), r.data + r.pos,
                  nx::cast<usize>(sz));
      r.pos += nx::cast<usize>(sz);
      push_subresource(img, offset, mip, layer, w, h, d);
      offset += sz;
    }
  }
  return img;
}

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

u32 swap32(const u32 v) noexcept {
  return ((v & 0xFF000000u) >> 24) | ((v & 0x00FF0000u) >> 8) |
         ((v & 0x0000FF00u) << 8) | ((v & 0x000000FFu) << 24);
}

Format format_from_gl(const u32 gl) noexcept {
  switch (gl) {
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

[[nodiscard]] bool ktx_level_layout(const Format format, const u32 width,
                                    const u32 height, const u32 depth,
                                    KtxLevelLayout &level) noexcept {
  SubresourceLayout decoded;
  if (!subresource_layout(format, width, height, depth, decoded))
    return false;

  u32 block_w = 0;
  u32 block_h = 0;
  format_block_extent(format, block_w, block_h);
  const u64 rows = (nx::cast<u64>(height) + block_h - 1) / block_h;
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

void copy_ktx_row(u8 *dst, const u8 *src, const usize bytes,
                  const u32 element_size, const bool swapped) noexcept {
  std::memcpy(dst, src, bytes);
  if (!swapped || element_size <= 1)
    return;

  for (usize element = 0; element < bytes; element += element_size) {
    for (u32 left = 0, right = element_size - 1; left < right;
         ++left, --right) {
      const u8 value = dst[element + left];
      dst[element + left] = dst[element + right];
      dst[element + right] = value;
    }
  }
}

Result parse_ktx(Reader &r) {
  KtxHeader h{};
  if (!r.read(h))
    return nx::err(ImageError::Corrupt);

  if (h.endianness != KTX_ENDIAN_NATIVE && h.endianness != KTX_ENDIAN_REVERSED)
    return nx::err(ImageError::Corrupt);

  const bool swapped = h.endianness == KTX_ENDIAN_REVERSED;
  if (swapped) {
    h.gl_type = swap32(h.gl_type);
    h.gl_type_size = swap32(h.gl_type_size);
    h.gl_format = swap32(h.gl_format);
    h.gl_internal_format = swap32(h.gl_internal_format);
    h.gl_base_internal_format = swap32(h.gl_base_internal_format);
    h.pixel_width = swap32(h.pixel_width);
    h.pixel_height = swap32(h.pixel_height);
    h.pixel_depth = swap32(h.pixel_depth);
    h.array_elements = swap32(h.array_elements);
    h.faces = swap32(h.faces);
    h.mip_levels = swap32(h.mip_levels);
    h.key_value_bytes = swap32(h.key_value_bytes);
  }

  ImageData img;
  img.format = format_from_gl(h.gl_internal_format);
  if (img.format == Format::Unknown)
    return nx::err(ImageError::UnsupportedFormat);

  const bool compressed = format_is_compressed(img.format);
  if ((compressed &&
       (h.gl_type != 0 || h.gl_format != 0 || h.gl_type_size != 1)) ||
      (!compressed && (h.gl_type == 0 || h.gl_format == 0 ||
                       (h.gl_type_size != 1 && h.gl_type_size != 2 &&
                        h.gl_type_size != 4 && h.gl_type_size != 8))))
    return nx::err(ImageError::Corrupt);

  if (h.pixel_width == 0 || (h.faces != 1 && h.faces != 6) ||
      (h.key_value_bytes & 3u) != 0)
    return nx::err(ImageError::Corrupt);

  const bool cube = h.faces == 6;
  const bool is_3d = h.pixel_depth != 0;
  if ((cube && (h.pixel_height == 0 || is_3d)) ||
      (is_3d && (h.pixel_height == 0 || h.array_elements != 0)))
    return nx::err(ImageError::Corrupt);

  u32 layers = h.array_elements == 0 ? 1 : h.array_elements;
  if (cube && layers > std::numeric_limits<u32>::max() / 6u)
    return nx::err(ImageError::Corrupt);
  if (cube)
    layers *= 6;
  if (layers > MAX_IMAGE_ARRAY_LAYERS)
    return nx::err(ImageError::Corrupt);

  img.width = h.pixel_width;
  img.height = h.pixel_height == 0 ? 1 : h.pixel_height;
  img.depth = is_3d ? h.pixel_depth : 1;
  img.mip_levels = h.mip_levels == 0 ? 1 : h.mip_levels;
  img.array_layers = layers;
  img.dimension = pick_dimension(cube, is_3d, layers);

  if (cube && img.width != img.height)
    return nx::err(ImageError::Corrupt);
  if (!r.skip(h.key_value_bytes))
    return nx::err(ImageError::Corrupt);

  if (img.mip_levels > maximum_mip_levels(img.width, img.height, img.depth))
    return nx::err(ImageError::Corrupt);

  KtxLevelLayout levels[32]{};
  Reader scan = r;
  u64 total = 0;
  const bool non_array_cube = cube && h.array_elements == 0;
  for (u32 mip = 0; mip < img.mip_levels; ++mip) {
    u32 image_size = 0;
    if (!scan.read(image_size))
      return nx::err(ImageError::Corrupt);
    if (swapped)
      image_size = swap32(image_size);

    KtxLevelLayout &level = levels[mip];
    const u32 width = dim_at(img.width, mip);
    const u32 height = dim_at(img.height, mip);
    const u32 depth = is_3d ? dim_at(img.depth, mip) : 1u;
    if (!ktx_level_layout(img.format, width, height, depth, level) ||
        level.decoded_row % h.gl_type_size != 0)
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
  img.pixels = nx::blob<u8>(nx::cast<usize>(total));
  img.subresources.reserve(
      nx::cast<usize>(nx::cast<u64>(layers) * img.mip_levels));

  // KTX stores mip-major data. Decode its aligned rows into the RHI's tightly
  // packed subresources, validating the entire encoded layout before
  // allocation.
  u64 offset = 0;
  for (u32 mip = 0; mip < img.mip_levels; ++mip) {
    if (!r.skip(sizeof(u32)))
      return nx::err(ImageError::Corrupt);

    const KtxLevelLayout &level = levels[mip];
    for (u32 layer = 0; layer < layers; ++layer) {
      const u64 dst_layer = offset + level.decoded_per_layer * layer;
      for (u32 z = 0; z < level.depth; ++z) {
        for (u32 row = 0; row < level.rows; ++row) {
          const u64 src_offset =
              level.encoded_row * (nx::cast<u64>(z) * level.rows + row);
          const u64 dst_offset =
              dst_layer +
              level.decoded_row * (nx::cast<u64>(z) * level.rows + row);
          copy_ktx_row(img.pixels.data() + nx::cast<usize>(dst_offset),
                       r.data + r.pos + nx::cast<usize>(src_offset),
                       nx::cast<usize>(level.decoded_row), h.gl_type_size,
                       swapped);
        }
      }
      if (!r.skip(nx::cast<usize>(level.encoded_per_layer)) ||
          (non_array_cube && !r.align4()))
        return nx::err(ImageError::Corrupt);
      push_subresource(img, dst_layer, mip, layer, level.width, level.height,
                       level.depth);
    }
    if (!r.align4())
      return nx::err(ImageError::Corrupt);
    offset += level.decoded_per_layer * layers;
  }
  return img;
}

Result parse_webp(const std::span<const u8> bytes) {
  WebPBitstreamFeatures features{};
  const VP8StatusCode status =
      WebPGetFeatures(bytes.data(), bytes.size(), &features);
  if (status != VP8_STATUS_OK) {
    nx::loge(log_img, "WebP header decode failed: status {}",
             nx::cast<i32>(status));
    return nx::err(status == VP8_STATUS_UNSUPPORTED_FEATURE
                       ? ImageError::UnsupportedFormat
                       : ImageError::Corrupt);
  }
  if (features.has_animation != 0) {
    nx::loge(log_img, "animated WebP is not a static texture");
    return nx::err(ImageError::UnsupportedFormat);
  }
  if (features.width <= 0 || features.height <= 0)
    return nx::err(ImageError::Corrupt);

  const u32 width = nx::cast<u32>(features.width);
  const u32 height = nx::cast<u32>(features.height);
  SubresourceLayout layout;
  if (!subresource_layout(Format::RGBA8_UNORM, width, height, 1, layout) ||
      layout.size > std::numeric_limits<usize>::max())
    return nx::err(ImageError::Corrupt);

  ImageData img;
  img.format = Format::RGBA8_UNORM;
  img.width = width;
  img.height = height;
  img.pixels = nx::blob<u8>(nx::cast<usize>(layout.size));
  if (WebPDecodeRGBAInto(bytes.data(), bytes.size(), img.pixels.data(),
                         img.pixels.size(),
                         nx::cast<int>(layout.row_pitch)) == nullptr) {
    nx::loge(log_img, "WebP pixel decode failed");
    return nx::err(ImageError::Corrupt);
  }

  push_subresource(img, 0, 0, 0, width, height, 1);
  return img;
}

Result parse_stb(const std::span<const u8> bytes) {
  if (bytes.size() > nx::cast<usize>(std::numeric_limits<int>::max()))
    return nx::err(ImageError::Corrupt);
  const int len = nx::cast<int>(bytes.size());
  int w = 0, h = 0, comp = 0;

  ImageData img;
  const bool hdr = stbi_is_hdr_from_memory(bytes.data(), len) != 0;

  void *pixels = nullptr;
  u64 stride = 0;
  if (hdr) {
    pixels = stbi_loadf_from_memory(bytes.data(), len, &w, &h, &comp, 4);
    img.format = Format::RGBA32_FLOAT;
    stride = 16;
  } else {
    pixels = stbi_load_from_memory(bytes.data(), len, &w, &h, &comp, 4);
    img.format = Format::RGBA8_UNORM;
    stride = 4;
  }

  if (pixels == nullptr) {
    nx::loge(log_img, "stb decode failed: {}", stbi_failure_reason());
    return nx::err(ImageError::Corrupt);
  }
  const std::unique_ptr<void, decltype(&stbi_image_free)> decoded(
      pixels, &stbi_image_free);

  img.width = nx::cast<u32>(w);
  img.height = nx::cast<u32>(h);
  const u64 size = nx::cast<u64>(w) * nx::cast<u64>(h) * stride;
  img.pixels = nx::blob<u8>(size);
  std::memcpy(img.pixels.data(), decoded.get(), size);

  push_subresource(img, 0, 0, 0, img.width, img.height, 1);
  return img;
}

} // namespace

nx::string_view to_string(const ImageError error) noexcept {
  switch (error) {
  case ImageError::Io:
    return "file read failed";
  case ImageError::Empty:
    return "empty input";
  case ImageError::UnknownContainer:
    return "unrecognised container";
  case ImageError::UnsupportedFormat:
    return "unsupported pixel format";
  case ImageError::Corrupt:
    return "corrupt or truncated data";
  default:
    return "unknown error";
  }
}

bool validate_image_upload(
    const ImageUploadDesc &target, const std::span<const u8> src,
    const std::span<const ImageSubresource> layout) noexcept {
  u32 block_width = 0;
  u32 block_height = 0;
  format_block_extent(target.format, block_width, block_height);
  const u64 bytes_per_block = format_texel_size(target.format);
  if (target.format == Format::Unknown || target.width == 0 ||
      target.height == 0 || target.depth == 0 || target.mip_levels == 0 ||
      target.array_layers == 0 || block_width == 0 || block_height == 0 ||
      bytes_per_block == 0 || layout.empty())
    return false;

  for (const ImageSubresource &sub : layout) {
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
    if (!checked_mul(sub.row_pitch, rows, bytes_per_slice) ||
        !checked_mul(bytes_per_slice, sub.depth, required) ||
        sub.size != required)
      return false;
  }
  return true;
}

TextureDesc
ImageData::texture_desc(const nx::string_view name,
                        const TextureUsageFlags usage) const noexcept {
  TextureDesc desc;
  desc.name = name;
  desc.format = format;
  desc.dimension = dimension;
  desc.width = width;
  desc.height = height;
  desc.depth_or_layers = dimension == TextureDimension::Tex3D
                             ? depth
                             : (array_layers > 1 ? array_layers : 1);
  desc.mip_levels = mip_levels;
  desc.usage = usage;
  desc.memory = MemoryUsage::DeviceLocal;
  desc.initial_state = ResourceState::Undefined;
  return desc;
}

Result load_image_from_memory(const std::span<const u8> bytes) {
  NX_PROFILE_ZONE("image::decode");
  if (bytes.empty())
    return nx::err(ImageError::Empty);

  Reader reader{bytes.data(), bytes.size(), 0};

  if (bytes.size() >= sizeof(DDS_MAGIC) &&
      std::memcmp(bytes.data(), DDS_MAGIC, sizeof(DDS_MAGIC)) == 0) {
    reader.pos = sizeof(DDS_MAGIC);
    return parse_dds(reader);
  }
  if (bytes.size() >= sizeof(KTX_MAGIC) &&
      std::memcmp(bytes.data(), KTX_MAGIC, sizeof(KTX_MAGIC)) == 0) {
    reader.pos = sizeof(KTX_MAGIC);
    return parse_ktx(reader);
  }
  if (is_webp(bytes))
    return parse_webp(bytes);
  return parse_stb(bytes);
}

Result load_image(const nx::fs::path_view path) {
  auto data = nx::fs::file_read(path);
  if (!data) {
    nx::loge(log_img, "load_image('{}'): {}", path,
             nx::fs::to_string(data.error().kind));
    return nx::err(ImageError::Io);
  }
  return load_image_from_memory(
      std::span<const u8>(data->data(), data->size()));
}

} // namespace nxe::rhi
