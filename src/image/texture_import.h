#pragma once

#include "core/foundation/strings/utf8_string.h"
#include "core/rendering/rhi/types.h"

#include <optional>

namespace nxe::rhi {

/// Intermediate container used by the offline encoder before assetc writes the
/// canonical TEX .nxb. Nxb stores ImageData directly and needs no intermediate.
enum class TextureCookContainer : u8 { Nxb, Dds, Ktx, Count };

enum class TextureCookQuality : u8 { Fast, Balanced, High, Exhaustive, Count };
enum class TextureStorage : u8 { Native, BasisEtc1s, BasisUastc, Count };

struct TexturePreset {
  nx::string name;
  TextureStorage storage = TextureStorage::Native;
  /// Unknown means preserve the format decoded from the source/intermediate.
  Format format = Format::Unknown;
  bool mipmaps = true;
  TextureCookQuality quality = TextureCookQuality::Balanced;
  i32 zstd_level = 10;
};

/// One selected, platform-specific profile. A project may point Windows and
/// Linux at one file and Android/iOS at another.
struct TextureProfile {
  nx::string name;
  TextureCookContainer container = TextureCookContainer::Nxb;
  nx::string default_preset;
  nx::vector<TexturePreset> presets;

  [[nodiscard]] const TexturePreset *
  find(nx::string_view preset) const noexcept;
};

/// Optional per-texture overrides read from the `texture` block in .meta.
/// Missing values inherit from the selected profile and an absent block is a
/// completely valid default import.
struct TextureImport {
  nx::string preset;
  std::optional<TextureStorage> storage;
  std::optional<Format> format;
  std::optional<bool> mipmaps;
  std::optional<TextureCookQuality> quality;
  std::optional<i32> zstd_level;
};

struct TextureCookSettings {
  TextureCookContainer container = TextureCookContainer::Nxb;
  TextureStorage storage = TextureStorage::Native;
  Format format = Format::Unknown;
  bool mipmaps = true;
  TextureCookQuality quality = TextureCookQuality::Balanced;
  i32 zstd_level = 10;
  nx::string preset;
};

[[nodiscard]] nx::string_view
to_string(TextureCookContainer container) noexcept;
[[nodiscard]] nx::string_view to_string(TextureCookQuality quality) noexcept;
[[nodiscard]] nx::string_view to_string(TextureStorage storage) noexcept;

/// Strict, versioned readers for authoring configuration. On failure, `error`
/// contains a human-readable field-level diagnostic.
[[nodiscard]] bool read_texture_profile(nx::string_view json,
                                        TextureProfile &out, nx::string &error);
[[nodiscard]] bool read_texture_import(nx::string_view asset_meta_json,
                                       TextureImport &out, nx::string &error);

[[nodiscard]] bool resolve_texture_settings(const TextureProfile &profile,
                                            const TextureImport &import,
                                            TextureCookSettings &out,
                                            nx::string &error);

[[nodiscard]] bool texture_profile_format_supported(Format format) noexcept;
[[nodiscard]] bool texture_container_supports(TextureCookContainer container,
                                              Format format) noexcept;

} // namespace nxe::rhi
