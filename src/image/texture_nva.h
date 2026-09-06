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

/// Sequential reader for a native TEX v3 mip chain. The DATA payload is laid
/// out from the coarsest mip to the finest, allowing a compressed NXP entry to
/// be inflated progressively without retaining the complete chain in RAM.
class TextureMipStream {
public:
  TextureMipStream() noexcept = default;
  TextureMipStream(TextureMipStream &&) noexcept = default;
  TextureMipStream &operator=(TextureMipStream &&) noexcept = default;

  TextureMipStream(const TextureMipStream &) = delete;
  TextureMipStream &operator=(const TextureMipStream &) = delete;

  [[nodiscard]] bool valid() const noexcept {
    return m_stream.valid() && m_next_mip != ~u32{0};
  }
  [[nodiscard]] const ImageData &layout() const noexcept { return m_layout; }
  [[nodiscard]] u32 next_mip() const noexcept { return m_next_mip; }

  /// Reads every still-unread level down through @p finest_mip. Returned
  /// subresource offsets address the compact output buffer, not the TEX file.
  [[nodiscard]] bool read_through(u32 finest_mip, ImageData &out);

private:
  friend bool open_texture_mip_stream(nx::fs::stream_ptr, TextureMipStream &);

  nx::fs::stream_ptr m_stream;
  ImageData m_layout;
  u64 m_data_offset = 0;
  u32 m_next_mip = ~u32{0};
};

/// A decoded image as an nva container: its layout in an INFO chunk, its pixels
/// in a DATA chunk. Deterministic, so a cook is reproducible.
[[nodiscard]] nx::vector<u8> encode_texture(const ImageData &image);

/// Wraps an ETC1S/UASTC KTX2 stream in the canonical TEX asset container.
/// Returns an empty vector when the KTX2 payload is invalid or unsupported.
[[nodiscard]] nx::vector<u8>
encode_basis_texture(std::span<const u8> basis_ktx2);

/// Reads legacy TEX v1 native payloads, v2 Basis payloads, and v3 native
/// coarse-first payloads.
[[nodiscard]] bool decode_texture_asset(std::span<const u8> bytes,
                                        TextureAsset &out);

/// Compatibility helper for consumers that accept native payloads only.
[[nodiscard]] bool decode_texture(std::span<const u8> bytes, ImageData &out);

/// Opens a progressively readable native TEX v3 2D mip chain. Other TEX
/// versions and payload kinds return false so callers can use the ordinary
/// whole-asset decoder as a compatibility fallback.
[[nodiscard]] bool open_texture_mip_stream(nx::fs::stream_ptr stream,
                                           TextureMipStream &out);

[[nodiscard]] nx::string cooked_texture_path(nx::string_view source);

} // namespace nxe::rhi
