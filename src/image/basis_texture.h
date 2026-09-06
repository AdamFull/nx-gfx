#pragma once

#include "core/rendering/rhi/image/image.h"

#include <span>

namespace nxe::rhi {

enum class BasisTextureEncoding : u8 { Etc1s = 0, Uastc, Count };

/// Container metadata that can be read without transcoding the texture.
struct BasisTextureInfo {
  BasisTextureEncoding encoding = BasisTextureEncoding::Etc1s;
  TextureDimension dimension = TextureDimension::Tex2D;
  u32 width = 0;
  u32 height = 0;
  u32 mip_levels = 0;
  u32 array_layers = 0; ///< Face-expanded, matching ImageData/RHI semantics.
  bool has_alpha = false;
  bool srgb = false;

  [[nodiscard]] bool valid() const noexcept {
    return encoding < BasisTextureEncoding::Count && width != 0 &&
           height != 0 && mip_levels != 0 && array_layers != 0;
  }
};

/// Validates a Basis Universal KTX2 payload and returns its GPU-independent
/// layout. KTX2 files with ordinary Vulkan payloads are intentionally rejected.
[[nodiscard]] bool inspect_basis_texture(std::span<const u8> bytes,
                                         BasisTextureInfo &out) noexcept;

/// CPU-transcodes every subresource of a Basis Universal KTX2 payload into a
/// GPU-ready ImageData chain. The destination format is selected by the caller
/// after querying the active RHI device.
[[nodiscard]] bool transcode_basis_texture(std::span<const u8> bytes,
                                           Format target,
                                           ImageData &out) noexcept;

/// Whether the runtime Basis transcoder can produce @p format.
[[nodiscard]] bool basis_transcode_target_supported(Format format) noexcept;

} // namespace nxe::rhi
