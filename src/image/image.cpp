/**
 * @file image.cpp
 * @brief DDS, KTX 1.0 and stb_image decoding into ImageData.
 *
 * Each container parser fills the ImageData header (format, extent, mip/layer
 * counts), sizes the pixel blob to the tightly-packed total, then copies every
 * subresource into it while recording its offset. The on-disk order differs by
 * container - DDS is layer-major, KTX is mip-major - but every ImageSubresource
 * is self-describing, so the uploader does not care which produced it.
 */

#include "rendering/rhi/image/image.h"

#include "foundation/diagnostics/log.h"

#include <stb_image.h>

#include <cstring>

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
  void align4() noexcept { pos = (pos + 3u) & ~usize{3}; }
};

[[nodiscard]] u32 dim_at(const u32 base, const u32 mip) noexcept {
  const u32 v = base >> mip;
  return v == 0 ? 1u : v;
}

/// Sum of every subresource's tightly-packed size.
[[nodiscard]] u64 total_size(const Format fmt, const u32 w, const u32 h,
                             const u32 d, const u32 mips, const u32 layers,
                             const bool is_3d) noexcept {
  u64 total = 0;
  for (u32 layer = 0; layer < layers; ++layer)
    for (u32 mip = 0; mip < mips; ++mip)
      total += format_subresource_size(fmt, dim_at(w, mip), dim_at(h, mip),
                                       is_3d ? dim_at(d, mip) : 1u);
  return total;
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

// --- DDS -------------------------------------------------------------------

constexpr u8 DDS_MAGIC[4] = {'D', 'D', 'S', ' '};

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

/// Legacy (pre-DX10) DDS: map the FourCC or RGB masks to an rhi::Format.
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
    // Distinguish RGBA from BGRA by which mask carries red.
    if (pf.r_mask == 0x000000FF)
      return Format::RGBA8_UNORM;
    if (pf.r_mask == 0x00FF0000)
      return Format::BGRA8_UNORM;
  }
  return Format::Unknown;
}

Result parse_dds(Reader &r) {
  DdsHeader header{};
  if (!r.read(header) || header.size != sizeof(DdsHeader))
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
    img.format = format_from_dxgi(ext.dxgi_format);
    layers = ext.array_size == 0 ? 1 : ext.array_size;
    cube = (ext.misc_flag & DDS_MISC_TEXTURECUBE) != 0;
    is_3d = ext.resource_dimension == DDS_DIM_TEXTURE3D;
  } else {
    img.format = format_from_dds_legacy(header.ddspf);
    cube = (header.caps2 & DDSCAPS2_CUBEMAP) != 0;
    is_3d = (header.caps2 & DDSCAPS2_VOLUME) != 0 || header.depth > 1;
  }

  if (img.format == Format::Unknown)
    return nx::err(ImageError::UnsupportedFormat);

  if (cube)
    layers *= 6;

  img.width = header.width;
  img.height = header.height == 0 ? 1 : header.height;
  img.depth = is_3d ? (header.depth == 0 ? 1 : header.depth) : 1;
  img.mip_levels = header.mip_count == 0 ? 1 : header.mip_count;
  img.array_layers = layers;
  img.dimension = pick_dimension(cube, is_3d, layers);

  const u64 total = total_size(img.format, img.width, img.height, img.depth,
                               img.mip_levels, layers, is_3d);
  if (!r.ok(total))
    return nx::err(ImageError::Corrupt);

  img.pixels = nx::blob<u8>(total);
  img.subresources.reserve(nx::cast<usize>(layers) * img.mip_levels);

  // DDS stores each array slice as a full mip chain: layer-major, mip-minor.
  u64 offset = 0;
  for (u32 layer = 0; layer < layers; ++layer) {
    for (u32 mip = 0; mip < img.mip_levels; ++mip) {
      const u32 w = dim_at(img.width, mip);
      const u32 h = dim_at(img.height, mip);
      const u32 d = is_3d ? dim_at(img.depth, mip) : 1u;
      const u64 sz = format_subresource_size(img.format, w, h, d);
      std::memcpy(img.pixels.data() + offset, r.data + r.pos, sz);
      r.pos += sz;
      push_subresource(img, offset, mip, layer, w, h, d);
      offset += sz;
    }
  }
  return img;
}

// --- KTX 1.0 ---------------------------------------------------------------

constexpr u8 KTX_MAGIC[12] = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31,
                              0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
constexpr u32 KTX_ENDIAN_REVERSED = 0x01020304;

struct KtxHeader {
  u32 endianness, gl_type, gl_type_size, gl_format, gl_internal_format,
      gl_base_internal_format, pixel_width, pixel_height, pixel_depth,
      array_elements, faces, mip_levels, key_value_bytes;
};
static_assert(sizeof(KtxHeader) == 52, "KTX 1.0 header must be 52 bytes");

u32 swap32(const u32 v) noexcept {
  return ((v & 0xFF000000u) >> 24) | ((v & 0x00FF0000u) >> 8) |
         ((v & 0x0000FF00u) << 8) | ((v & 0x000000FFu) << 24);
}

Format format_from_gl(const u32 gl) noexcept {
  switch (gl) {
  case 0x8058:
    return Format::RGBA8_UNORM; // RGBA8
  case 0x8C43:
    return Format::RGBA8_SRGB; // SRGB8_ALPHA8
  case 0x8229:
    return Format::R8_UNORM; // R8
  case 0x822B:
    return Format::RG8_UNORM; // RG8
  case 0x822A:
    return Format::R16_UNORM; // R16
  case 0x822D:
    return Format::R16_FLOAT; // R16F
  case 0x822F:
    return Format::RG16_FLOAT; // RG16F
  case 0x881A:
    return Format::RGBA16_FLOAT; // RGBA16F
  case 0x822E:
    return Format::R32_FLOAT; // R32F
  case 0x8230:
    return Format::RG32_FLOAT; // RG32F
  case 0x8814:
    return Format::RGBA32_FLOAT; // RGBA32F
  case 0x83F0:
    return Format::BC1_RGB_UNORM; // RGB_S3TC_DXT1
  case 0x8C4C:
    return Format::BC1_RGB_SRGB; // SRGB_S3TC_DXT1
  case 0x83F1:
    return Format::BC1_RGBA_UNORM; // RGBA_S3TC_DXT1
  case 0x8C4D:
    return Format::BC1_RGBA_SRGB; // SRGB_ALPHA_S3TC_DXT1
  case 0x83F2:
    return Format::BC2_RGBA_UNORM; // RGBA_S3TC_DXT3
  case 0x8C4E:
    return Format::BC2_RGBA_SRGB; // SRGB_ALPHA_S3TC_DXT3
  case 0x83F3:
    return Format::BC3_RGBA_UNORM; // RGBA_S3TC_DXT5
  case 0x8C4F:
    return Format::BC3_RGBA_SRGB; // SRGB_ALPHA_S3TC_DXT5
  case 0x8DBB:
    return Format::BC4_R_UNORM; // RED_RGTC1
  case 0x8DBD:
    return Format::BC5_RG_UNORM; // RG_RGTC2
  case 0x8E8F:
    return Format::BC6H_RGB_UFLOAT; // BPTC_UNSIGNED_FLOAT
  case 0x8E8E:
    return Format::BC6H_RGB_SFLOAT; // BPTC_SIGNED_FLOAT
  case 0x8E8C:
    return Format::BC7_RGBA_UNORM; // BPTC_UNORM
  case 0x8E8D:
    return Format::BC7_RGBA_SRGB; // SRGB_ALPHA_BPTC_UNORM
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

Result parse_ktx(Reader &r) {
  KtxHeader h{};
  if (!r.read(h))
    return nx::err(ImageError::Corrupt);

  const bool swapped = h.endianness == KTX_ENDIAN_REVERSED;
  if (swapped) {
    h.gl_internal_format = swap32(h.gl_internal_format);
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

  const bool cube = h.faces > 1;
  const bool is_3d = h.pixel_depth > 1;
  u32 layers = h.array_elements == 0 ? 1 : h.array_elements;
  if (cube)
    layers *= 6;

  img.width = h.pixel_width;
  img.height = h.pixel_height == 0 ? 1 : h.pixel_height;
  img.depth = is_3d ? h.pixel_depth : 1;
  img.mip_levels = h.mip_levels == 0 ? 1 : h.mip_levels;
  img.array_layers = layers;
  img.dimension = pick_dimension(cube, is_3d, layers);

  if (!r.skip(h.key_value_bytes))
    return nx::err(ImageError::Corrupt);

  const u64 total = total_size(img.format, img.width, img.height, img.depth,
                               img.mip_levels, layers, is_3d);
  img.pixels = nx::blob<u8>(total);
  img.subresources.reserve(nx::cast<usize>(layers) * img.mip_levels);

  // KTX stores mip-major: [imageSize u32][all layers of the level], each level
  // padded to a 4-byte boundary.
  u64 offset = 0;
  for (u32 mip = 0; mip < img.mip_levels; ++mip) {
    u32 image_size = 0;
    if (!r.read(image_size))
      return nx::err(ImageError::Corrupt);
    if (swapped)
      image_size = swap32(image_size);

    const u32 w = dim_at(img.width, mip);
    const u32 hgt = dim_at(img.height, mip);
    const u32 d = is_3d ? dim_at(img.depth, mip) : 1u;
    const u64 per_layer = format_subresource_size(img.format, w, hgt, d);
    const u64 level_bytes = per_layer * layers;
    if (!r.ok(level_bytes))
      return nx::err(ImageError::Corrupt);

    for (u32 layer = 0; layer < layers; ++layer) {
      const u64 dst = offset + per_layer * layer;
      std::memcpy(img.pixels.data() + dst, r.data + r.pos + per_layer * layer,
                  per_layer);
      push_subresource(img, dst, mip, layer, w, hgt, d);
    }
    r.pos += level_bytes;
    r.align4();
    offset += level_bytes;
  }
  return img;
}

// --- stb (PNG / JPG / BMP / TGA / HDR / ...) --------------------------------

Result parse_stb(std::span<const u8> bytes) {
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

  img.width = nx::cast<u32>(w);
  img.height = nx::cast<u32>(h);
  const u64 size = nx::cast<u64>(w) * nx::cast<u64>(h) * stride;
  img.pixels = nx::blob<u8>(size);
  std::memcpy(img.pixels.data(), pixels, size);
  stbi_image_free(pixels);

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
