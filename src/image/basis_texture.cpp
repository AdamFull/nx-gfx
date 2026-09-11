#include "core/rendering/rhi/image/basis_texture.h"

#include "core/foundation/threading/sync.h"
#include "core/rendering/rhi/image/ktx2.h"

#include <transcoder/basisu_transcoder.h>

#include <cstring>
#include <limits>

namespace nxe::rhi {
namespace {

using BasisFormat = basist::transcoder_texture_format;

void initialize_transcoder() noexcept {
  static nx::once_flag once;
  nx::call_once(once, [] { basist::basisu_transcoder_init(); });
}

[[nodiscard]] bool target_format(const Format format,
                                 BasisFormat &out) noexcept {
  switch (format) {
  case Format::BC1_RGB_UNORM:
  case Format::BC1_RGB_SRGB:
  case Format::BC1_RGBA_UNORM:
  case Format::BC1_RGBA_SRGB:
    out = BasisFormat::cTFBC1_RGB;
    return true;
  case Format::BC3_RGBA_UNORM:
  case Format::BC3_RGBA_SRGB:
    out = BasisFormat::cTFBC3_RGBA;
    return true;
  case Format::BC7_RGBA_UNORM:
  case Format::BC7_RGBA_SRGB:
    out = BasisFormat::cTFBC7_RGBA;
    return true;
  case Format::ASTC_4x4_UNORM:
  case Format::ASTC_4x4_SRGB:
    out = BasisFormat::cTFASTC_LDR_4x4_RGBA;
    return true;
  case Format::RGBA8_UNORM:
  case Format::RGBA8_SRGB:
    out = BasisFormat::cTFRGBA32;
    return true;
  default:
    return false;
  }
}

[[nodiscard]] bool read_info(basist::ktx2_transcoder &transcoder,
                             const std::span<const u8> bytes,
                             BasisTextureInfo &out) noexcept {
  if (bytes.empty() || bytes.size() > std::numeric_limits<u32>::max() ||
      !transcoder.init(bytes.data(), nx::cast<u32>(bytes.size())))
    return false;

  const u32 faces = transcoder.get_faces();
  const u32 layers = nx::max(1u, transcoder.get_layers());
  if ((!transcoder.is_etc1s() && !transcoder.is_uastc()) ||
      (faces != 1 && faces != 6) || layers > MAX_IMAGE_ARRAY_LAYERS / faces)
    return false;

  BasisTextureInfo info;
  info.encoding = transcoder.is_uastc() ? BasisTextureEncoding::Uastc
                                        : BasisTextureEncoding::Etc1s;
  info.dimension =
      faces == 6 ? TextureDimension::TexCube
                 : (transcoder.get_layers() != 0 ? TextureDimension::Tex2DArray
                                                 : TextureDimension::Tex2D);
  info.width = transcoder.get_width();
  info.height = transcoder.get_height();
  info.mip_levels = transcoder.get_levels();
  info.array_layers = layers * faces;
  info.has_alpha = transcoder.get_has_alpha() != 0;
  info.srgb = transcoder.is_srgb();
  if (!info.valid() || (info.dimension == TextureDimension::TexCube &&
                        info.width != info.height))
    return false;
  out = info;
  return true;
}

} // namespace

bool inspect_basis_texture(const std::span<const u8> bytes,
                           BasisTextureInfo &out) noexcept {
  const auto container = inspect_ktx2(bytes);
  if (!container || container->payload == Ktx2PayloadKind::Native)
    return false;

  initialize_transcoder();
  basist::ktx2_transcoder transcoder;
  BasisTextureInfo info;
  const BasisTextureEncoding expected =
      container->payload == Ktx2PayloadKind::BasisUastc
          ? BasisTextureEncoding::Uastc
          : BasisTextureEncoding::Etc1s;
  if (!read_info(transcoder, bytes, info) || info.encoding != expected ||
      info.dimension != container->dimension ||
      info.width != container->width || info.height != container->height ||
      info.mip_levels != container->mip_levels ||
      info.array_layers != container->array_layers ||
      info.srgb != container->srgb)
    return false;
  out = info;
  return true;
}

bool basis_transcode_target_supported(const Format format) noexcept {
  BasisFormat ignored{};
  return target_format(format, ignored);
}

bool transcode_basis_texture(const std::span<const u8> bytes,
                             const Format target, ImageData &out,
                             const nx::cancellation_token &cancellation) {
  cancellation.throw_if_requested();
  BasisFormat basis_format{};
  if (!target_format(target, basis_format))
    return false;

  initialize_transcoder();
  basist::ktx2_transcoder transcoder;
  BasisTextureInfo info;
  if (!read_info(transcoder, bytes, info) || !transcoder.start_transcoding())
    return false;

  const u32 faces = transcoder.get_faces();
  const u32 layers = nx::max(1u, transcoder.get_layers());
  const u64 subresource_count = nx::cast<u64>(layers) * faces * info.mip_levels;
  if (subresource_count > std::numeric_limits<usize>::max())
    return false;

  ImageData decoded;
  decoded.format = target;
  decoded.dimension = info.dimension;
  decoded.width = info.width;
  decoded.height = info.height;
  decoded.depth = 1;
  decoded.mip_levels = info.mip_levels;
  decoded.array_layers = info.array_layers;
  decoded.subresources.reserve(nx::cast<usize>(subresource_count));

  u64 total_bytes = 0;
  for (u32 layer = 0; layer < layers; ++layer) {
    for (u32 face = 0; face < faces; ++face) {
      for (u32 mip = 0; mip < info.mip_levels; ++mip) {
        cancellation.throw_if_requested();
        basist::ktx2_image_level_info level{};
        if (!transcoder.get_image_level_info(level, mip, layer, face))
          return false;
        const u64 size = format_subresource_size(target, level.m_orig_width,
                                                 level.m_orig_height, 1);
        const u64 row_pitch = format_row_pitch(target, level.m_orig_width);
        if (size == 0 || row_pitch > std::numeric_limits<u32>::max() ||
            total_bytes > std::numeric_limits<u64>::max() - size)
          return false;
        decoded.subresources.push_back(
            {total_bytes, size, mip, layer * faces + face, level.m_orig_width,
             level.m_orig_height, 1, nx::cast<u32>(row_pitch)});
        total_bytes += size;
      }
    }
  }
  if (total_bytes == 0 || total_bytes > std::numeric_limits<usize>::max())
    return false;
  decoded.pixels = nx::blob<u8>(nx::cast<usize>(total_bytes));

  const u32 unit_size =
      basist::basis_get_bytes_per_block_or_pixel(basis_format);
  if (unit_size == 0)
    return false;

  // KTX2 supercompression is level-oriented. Transcoding every image in one
  // level before moving to the next avoids repeatedly inflating the same Zstd
  // payload for array and cube textures while preserving ImageData's canonical
  // layer-major subresource order.
  for (u32 mip = 0; mip < info.mip_levels; ++mip) {
    cancellation.throw_if_requested();
    for (u32 layer = 0; layer < layers; ++layer) {
      for (u32 face = 0; face < faces; ++face) {
        cancellation.throw_if_requested();
        const usize index =
            (nx::cast<usize>(layer) * faces + face) * info.mip_levels + mip;
        const ImageSubresource &subresource = decoded.subresources[index];
        if (subresource.size % unit_size != 0 ||
            subresource.size / unit_size > std::numeric_limits<u32>::max())
          return false;
        void *destination = decoded.pixels.data() + subresource.offset;
        if (!transcoder.transcode_image_level(
                mip, layer, face, destination,
                nx::cast<u32>(subresource.size / unit_size), basis_format))
          return false;
      }
    }
  }

  if (!decoded.valid())
    return false;
  out = std::move(decoded);
  return true;
}

} // namespace nxe::rhi
