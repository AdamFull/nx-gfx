#pragma once

#include "rendering/rhi/image/image.h"

#include <span>

namespace nxe::rhi {

enum class Ktx2PayloadKind : u8 { Native = 0, BasisEtc1s, BasisUastc, Count };

enum class Ktx2Supercompression : u32 {
  None = 0,
  BasisLz = 1,
  Zstd = 2,
  Zlib = 3,
};

/// Container metadata that can be inspected without inflating or transcoding
/// any mip level. Array layers are face-expanded to match ImageData and RHI
/// semantics.
struct Ktx2Info {
  Ktx2PayloadKind payload = Ktx2PayloadKind::Native;
  Ktx2Supercompression supercompression = Ktx2Supercompression::None;
  Format format = Format::Unknown; ///< Unknown for universal Basis payloads.
  TextureDimension dimension = TextureDimension::Tex2D;
  u32 width = 0;
  u32 height = 0;
  u32 depth = 0;
  u32 mip_levels = 0;
  u32 array_layers = 0; ///< Face-expanded.
  bool srgb = false;

  [[nodiscard]] bool valid() const noexcept {
    return payload < Ktx2PayloadKind::Count && width != 0 && height != 0 &&
           depth != 0 && mip_levels != 0 && array_layers != 0 &&
           (payload != Ktx2PayloadKind::Native || format != Format::Unknown);
  }
};

[[nodiscard]] bool is_ktx2(std::span<const u8> bytes) noexcept;

/// Validates the KTX2 container, level index, supported metadata and texture
/// shape without decoding level data. Unsupported native vkFormats and
/// non-canonical orientation/swizzle metadata fail explicitly.
[[nodiscard]] nx::result<Ktx2Info, ImageError>
inspect_ktx2(std::span<const u8> bytes) noexcept;

/// Decodes a native vkFormat KTX2 payload. NONE and per-level Zstandard
/// supercompression are supported. Basis payloads deliberately return
/// UnsupportedFormat and remain the responsibility of basis_texture.
[[nodiscard]] nx::result<ImageData, ImageError>
decode_native_ktx2(std::span<const u8> bytes);

} // namespace nxe::rhi
