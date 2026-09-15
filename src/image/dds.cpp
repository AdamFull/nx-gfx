#include "rendering/rhi/image/dds.h"

#include "rendering/rhi/image/image_decode.h"

#include <cstring>
#include <limits>

namespace nxe::rhi {
namespace {

using namespace detail;

constexpr u8 DDS_MAGIC[4] = {'D', 'D', 'S', ' '};

constexpr u32 make_fourcc(const char a, const char b, const char c,
                          const char d) noexcept {
  return nx::cast<u32>(nx::cast<u8>(a)) |
         (nx::cast<u32>(nx::cast<u8>(b)) << 8) |
         (nx::cast<u32>(nx::cast<u8>(c)) << 16) |
         (nx::cast<u32>(nx::cast<u8>(d)) << 24);
}

enum : u32 {
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

[[nodiscard]] Format format_from_dxgi(const u32 dxgi) noexcept {
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

[[nodiscard]] Format
format_from_legacy_header(const DdsPixelFormat &format) noexcept {
  if (format.flags & DDPF_FOURCC) {
    if (format.fourcc == make_fourcc('D', 'X', 'T', '1'))
      return Format::BC1_RGBA_UNORM;
    if (format.fourcc == make_fourcc('D', 'X', 'T', '3'))
      return Format::BC2_RGBA_UNORM;
    if (format.fourcc == make_fourcc('D', 'X', 'T', '5'))
      return Format::BC3_RGBA_UNORM;
    if (format.fourcc == make_fourcc('A', 'T', 'I', '1') ||
        format.fourcc == make_fourcc('B', 'C', '4', 'U'))
      return Format::BC4_R_UNORM;
    if (format.fourcc == make_fourcc('A', 'T', 'I', '2') ||
        format.fourcc == make_fourcc('B', 'C', '5', 'U'))
      return Format::BC5_RG_UNORM;
    return Format::Unknown;
  }
  if (format.flags & DDPF_RGB && format.rgb_bit_count == 32) {
    if (format.r_mask == 0x000000FF)
      return Format::RGBA8_UNORM;
    if (format.r_mask == 0x00FF0000)
      return Format::BGRA8_UNORM;
  }
  return Format::Unknown;
}

} // namespace

bool is_dds(const std::span<const u8> bytes) noexcept {
  return bytes.size() >= sizeof(DDS_MAGIC) &&
         std::memcmp(bytes.data(), DDS_MAGIC, sizeof(DDS_MAGIC)) == 0;
}

ImageResult decode_dds(const std::span<const u8> bytes) {
  if (!is_dds(bytes))
    return nx::err(ImageError::UnknownContainer);

  ImageReader reader{bytes.data(), bytes.size(), sizeof(DDS_MAGIC)};
  DdsHeader header{};
  if (!reader.read(header) || header.size != sizeof(DdsHeader) ||
      header.ddspf.size != sizeof(DdsPixelFormat))
    return nx::err(ImageError::Corrupt);

  ImageData image;
  bool cube = false;
  bool is_3d = false;
  u32 layers = 1;

  const bool dx10 = (header.ddspf.flags & DDPF_FOURCC) &&
                    header.ddspf.fourcc == make_fourcc('D', 'X', '1', '0');
  if (dx10) {
    DdsHeaderDX10 extension{};
    if (!reader.read(extension))
      return nx::err(ImageError::Corrupt);
    if (extension.array_size == 0 ||
        (extension.resource_dimension != 3 &&
         extension.resource_dimension != DDS_DIM_TEXTURE3D))
      return nx::err(ImageError::Corrupt);
    image.format = format_from_dxgi(extension.dxgi_format);
    layers = extension.array_size;
    cube = (extension.misc_flag & DDS_MISC_TEXTURECUBE) != 0;
    is_3d = extension.resource_dimension == DDS_DIM_TEXTURE3D;
    if (is_3d && (cube || layers != 1))
      return nx::err(ImageError::Corrupt);
  } else {
    image.format = format_from_legacy_header(header.ddspf);
    cube = (header.caps2 & DDSCAPS2_CUBEMAP) != 0;
    is_3d = (header.caps2 & DDSCAPS2_VOLUME) != 0 || header.depth > 1;
  }

  if (image.format == Format::Unknown)
    return nx::err(ImageError::UnsupportedFormat);

  if (cube && layers > std::numeric_limits<u32>::max() / 6u)
    return nx::err(ImageError::Corrupt);
  if (cube)
    layers *= 6;
  if (layers > MAX_IMAGE_ARRAY_LAYERS)
    return nx::err(ImageError::Corrupt);

  image.width = header.width;
  image.height = header.height;
  image.depth = is_3d ? header.depth : 1;
  image.mip_levels = header.mip_count == 0 ? 1 : header.mip_count;
  image.array_layers = layers;
  image.dimension = image_dimension(cube, is_3d, layers);

  if (cube && image.width != image.height)
    return nx::err(ImageError::Corrupt);

  u64 total = 0;
  if (!image_total_size(image.format, image.width, image.height, image.depth,
                        image.mip_levels, layers, is_3d, total) ||
      total > reader.remaining())
    return nx::err(ImageError::Corrupt);

  image.pixels = nx::blob<u8>(nx::cast<usize>(total));
  image.subresources.reserve(
      nx::cast<usize>(nx::cast<u64>(layers) * image.mip_levels));

  u64 offset = 0;
  for (u32 layer = 0; layer < layers; ++layer) {
    for (u32 mip = 0; mip < image.mip_levels; ++mip) {
      const u32 width = mip_dimension(image.width, mip);
      const u32 height = mip_dimension(image.height, mip);
      const u32 depth = is_3d ? mip_dimension(image.depth, mip) : 1u;
      const u64 size =
          format_subresource_size(image.format, width, height, depth);
      std::memcpy(image.pixels.data() + nx::cast<usize>(offset),
                  reader.data + reader.pos, nx::cast<usize>(size));
      reader.pos += nx::cast<usize>(size);
      append_subresource(image, offset, mip, layer, width, height, depth);
      offset += size;
    }
  }
  return image;
}

} // namespace nxe::rhi
