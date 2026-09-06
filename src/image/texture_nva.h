#pragma once

#include "core/foundation/strings/utf8_string.h"
#include "core/rendering/rhi/image/basis_texture.h"

#include <span>

namespace nxe::rhi {

enum class TexturePayloadKind : u8 { Native = 0, BasisKtx2, Count };

/// Decoded TEX container. Native payloads are already GPU-ready; BasisKtx2
/// payloads retain their compact bytes until a runtime target has been chosen.
struct TextureAsset {
  TexturePayloadKind kind = TexturePayloadKind::Native;
  ImageData native;
  BasisTextureInfo basis_info;
  nx::blob<u8> basis_ktx2;

  [[nodiscard]] bool valid() const noexcept {
    return kind == TexturePayloadKind::Native
               ? native.valid()
               : kind == TexturePayloadKind::BasisKtx2 && basis_info.valid() &&
                     !basis_ktx2.empty();
  }
};

/// A decoded image as an nva container: its layout in an INFO chunk, its pixels
/// in a DATA chunk. Deterministic, so a cook is reproducible.
[[nodiscard]] nx::vector<u8> encode_texture(const ImageData &image);

/// Wraps an ETC1S/UASTC KTX2 stream in the canonical TEX asset container.
/// Returns an empty vector when the KTX2 payload is invalid or unsupported.
[[nodiscard]] nx::vector<u8>
encode_basis_texture(std::span<const u8> basis_ktx2);

/// Reads either the legacy TEX v1 native payload or a TEX v2 payload.
[[nodiscard]] bool decode_texture_asset(std::span<const u8> bytes,
                                        TextureAsset &out);

/// Compatibility helper for consumers that accept native payloads only.
[[nodiscard]] bool decode_texture(std::span<const u8> bytes, ImageData &out);

[[nodiscard]] nx::string cooked_texture_path(nx::string_view source);

} // namespace nxe::rhi
