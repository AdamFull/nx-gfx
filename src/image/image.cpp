#include "rendering/rhi/image/image.h"

#include "rendering/rhi/image/basis_texture.h"
#include "rendering/rhi/image/dds.h"
#include "rendering/rhi/image/image_decode.h"
#include "rendering/rhi/image/ktx.h"
#include "rendering/rhi/image/ktx2.h"

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

constexpr u8 WEBP_RIFF[4] = {'R', 'I', 'F', 'F'};
constexpr u8 WEBP_FORM[4] = {'W', 'E', 'B', 'P'};

[[nodiscard]] bool is_webp(const std::span<const u8> bytes) noexcept {
  return bytes.size() >= 12 &&
         std::memcmp(bytes.data(), WEBP_RIFF, sizeof(WEBP_RIFF)) == 0 &&
         std::memcmp(bytes.data() + 8, WEBP_FORM, sizeof(WEBP_FORM)) == 0;
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
  detail::SubresourceLayout layout;
  if (!detail::subresource_layout(Format::RGBA8_UNORM, width, height, 1,
                                  layout) ||
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

  detail::append_subresource(img, 0, 0, 0, width, height, 1);
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

  detail::append_subresource(img, 0, 0, 0, img.width, img.height, 1);
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
    if (!detail::checked_mul(sub.row_pitch, rows, bytes_per_slice) ||
        !detail::checked_mul(bytes_per_slice, sub.depth, required) ||
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

  if (is_ktx2(bytes)) {
    auto ktx2 = inspect_ktx2(bytes);
    if (!ktx2)
      return nx::err(ktx2.error());
    if (ktx2->payload == Ktx2PayloadKind::Native)
      return decode_native_ktx2(bytes);

    BasisTextureInfo info;
    if (!inspect_basis_texture(bytes, info))
      return nx::err(ImageError::Corrupt);
    ImageData decoded;
    const Format format = info.srgb ? Format::RGBA8_SRGB : Format::RGBA8_UNORM;
    if (!transcode_basis_texture(bytes, format, decoded))
      return nx::err(ImageError::Corrupt);
    return decoded;
  }

  if (is_dds(bytes))
    return decode_dds(bytes);
  if (is_ktx(bytes))
    return decode_ktx(bytes);
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
