#include "core/rendering/rhi/image/texture_import.h"

#include "core/foundation/serialization/json.h"
#include "core/foundation/strings/format.h"

namespace nxe::rhi {
namespace {

constexpr u64 PROFILE_VERSION = 1;
constexpr u64 IMPORT_VERSION = 1;

[[nodiscard]] bool portable_name(const nx::string_view name) noexcept {
  if (name.empty() || name.size() > 64 || name.front() < 'a' ||
      name.front() > 'z')
    return false;
  for (const char c : name)
    if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' ||
          c == '-'))
      return false;
  return true;
}

[[nodiscard]] bool known_fields(const nx::json::value &object,
                                const std::span<const nx::string_view> known,
                                const nx::string_view owner,
                                nx::string &error) {
  for (const nx::json::member &member : object.as_object()) {
    bool found = false;
    for (const nx::string_view field : known)
      if (member.key.view() == field) {
        found = true;
        break;
      }
    if (!found) {
      error = nx::format("{} has unknown field '{}'", owner, member.key.view());
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool read_version(const nx::json::value &object,
                                const u64 expected, const nx::string_view owner,
                                nx::string &error) {
  const nx::json::value *const version = object.find("version");
  if (version == nullptr || !version->fits<u64>() ||
      version->get_u64() != expected) {
    error = nx::format("{} version must be {}", owner, expected);
    return false;
  }
  return true;
}

[[nodiscard]] std::optional<TextureCookContainer>
container_from_string(const nx::string_view name) noexcept {
  if (name == "nxb")
    return TextureCookContainer::Nxb;
  if (name == "dds")
    return TextureCookContainer::Dds;
  if (name == "ktx")
    return TextureCookContainer::Ktx;
  return std::nullopt;
}

[[nodiscard]] std::optional<TextureCookQuality>
quality_from_string(const nx::string_view name) noexcept {
  if (name == "fast")
    return TextureCookQuality::Fast;
  if (name == "balanced")
    return TextureCookQuality::Balanced;
  if (name == "high")
    return TextureCookQuality::High;
  if (name == "exhaustive")
    return TextureCookQuality::Exhaustive;
  return std::nullopt;
}

[[nodiscard]] bool read_target_format(const nx::json::value &value, Format &out,
                                      const nx::string_view owner,
                                      nx::string &error) {
  if (!value.is_string()) {
    error = nx::format("{}.format must be a string", owner);
    return false;
  }
  const nx::string_view name = value.get_string();
  out = name == "source" ? Format::Unknown : format_from_string(name);
  if (name != "source" && !texture_profile_format_supported(out)) {
    error = nx::format("{}.format '{}' is not a cookable texture format", owner,
                       name);
    return false;
  }
  return true;
}

[[nodiscard]] bool read_quality(const nx::json::value &value,
                                TextureCookQuality &out,
                                const nx::string_view owner,
                                nx::string &error) {
  if (!value.is_string()) {
    error = nx::format("{}.quality must be a string", owner);
    return false;
  }
  const auto parsed = quality_from_string(value.get_string());
  if (!parsed) {
    error = nx::format("{}.quality must be fast, balanced, high, or exhaustive",
                       owner);
    return false;
  }
  out = *parsed;
  return true;
}

[[nodiscard]] bool is_astc(const Format format) noexcept {
  return format >= Format::ASTC_4x4_UNORM && format <= Format::ASTC_8x8_SRGB;
}

} // namespace

const TexturePreset *
TextureProfile::find(const nx::string_view preset) const noexcept {
  for (const TexturePreset &one : presets)
    if (one.name.view() == preset)
      return &one;
  return nullptr;
}

nx::string_view to_string(const TextureCookContainer container) noexcept {
  switch (container) {
  case TextureCookContainer::Nxb:
    return "nxb";
  case TextureCookContainer::Dds:
    return "dds";
  case TextureCookContainer::Ktx:
    return "ktx";
  default:
    return "invalid";
  }
}

nx::string_view to_string(const TextureCookQuality quality) noexcept {
  switch (quality) {
  case TextureCookQuality::Fast:
    return "fast";
  case TextureCookQuality::Balanced:
    return "balanced";
  case TextureCookQuality::High:
    return "high";
  case TextureCookQuality::Exhaustive:
    return "exhaustive";
  default:
    return "invalid";
  }
}

bool texture_profile_format_supported(const Format format) noexcept {
  return format == Format::R8_UNORM || format == Format::RG8_UNORM ||
         format == Format::RGBA8_UNORM || format == Format::RGBA8_SRGB ||
         format == Format::RGBA16_FLOAT || format_is_compressed(format);
}

bool texture_container_supports(const TextureCookContainer container,
                                const Format format) noexcept {
  if (format == Format::Unknown)
    return container == TextureCookContainer::Nxb;
  if (!texture_profile_format_supported(format))
    return false;
  return container != TextureCookContainer::Dds || !is_astc(format);
}

bool read_texture_profile(const nx::string_view json, TextureProfile &out,
                          nx::string &error) {
  static constexpr nx::string_view ROOT_FIELDS[] = {
      "version", "name", "container", "defaultPreset", "presets"};
  static constexpr nx::string_view PRESET_FIELDS[] = {"format", "mipmaps",
                                                      "quality"};
  error.clear();

  const auto parsed = nx::json::parse(
      json, {.reject_duplicate_keys = true, .validate_utf8 = true});
  if (!parsed || !parsed->is_object()) {
    error = "texture profile must be a JSON object";
    return false;
  }
  const nx::json::value &root = *parsed;
  if (!known_fields(root, ROOT_FIELDS, "texture profile", error) ||
      !read_version(root, PROFILE_VERSION, "texture profile", error))
    return false;

  const nx::json::value *const name = root.find("name");
  if (name == nullptr || !name->is_string() ||
      !portable_name(name->get_string())) {
    error = "texture profile.name must be a portable lower-case name";
    return false;
  }
  const nx::json::value *const container = root.find("container");
  const auto parsed_container =
      container != nullptr && container->is_string()
          ? container_from_string(container->get_string())
          : std::nullopt;
  if (!parsed_container) {
    error = "texture profile.container must be nxb, dds, or ktx";
    return false;
  }
  const nx::json::value *const fallback = root.find("defaultPreset");
  if (fallback == nullptr || !fallback->is_string() ||
      !portable_name(fallback->get_string())) {
    error = "texture profile.defaultPreset must be a portable preset name";
    return false;
  }
  const nx::json::value *const presets = root.find("presets");
  if (presets == nullptr || !presets->is_object() || presets->empty()) {
    error = "texture profile.presets must be a non-empty object";
    return false;
  }

  TextureProfile built;
  built.name = nx::string(name->get_string());
  built.container = *parsed_container;
  built.default_preset = nx::string(fallback->get_string());
  built.presets.reserve(presets->size());
  for (const nx::json::member &member : presets->as_object()) {
    const nx::string_view preset_name = member.key.view();
    const nx::string owner = nx::format("texture preset '{}'", preset_name);
    if (!portable_name(preset_name)) {
      error = nx::format("{} is not a portable lower-case name", owner.view());
      return false;
    }
    if (!member.val.is_object() ||
        !known_fields(member.val, PRESET_FIELDS, owner.view(), error)) {
      if (error.empty())
        error = nx::format("{} must be an object", owner.view());
      return false;
    }
    const nx::json::value *const format = member.val.find("format");
    const nx::json::value *const mipmaps = member.val.find("mipmaps");
    const nx::json::value *const quality = member.val.find("quality");
    TexturePreset preset;
    preset.name = nx::string(preset_name);
    if (format == nullptr ||
        !read_target_format(*format, preset.format, owner.view(), error))
      return false;
    if (mipmaps == nullptr || !mipmaps->is_bool()) {
      error = nx::format("{}.mipmaps must be true or false", owner.view());
      return false;
    }
    preset.mipmaps = mipmaps->as_bool();
    if (quality == nullptr ||
        !read_quality(*quality, preset.quality, owner.view(), error))
      return false;
    if (!texture_container_supports(built.container, preset.format)) {
      error = nx::format(
          "{} format {} cannot be stored through a {} encoder", owner.view(),
          preset.format == Format::Unknown ? nx::string_view("source")
                                           : to_string(preset.format),
          to_string(built.container));
      return false;
    }
    built.presets.push_back(std::move(preset));
  }
  if (built.find(built.default_preset.view()) == nullptr) {
    error = nx::format("texture profile.defaultPreset '{}' does not exist",
                       built.default_preset.view());
    return false;
  }
  out = std::move(built);
  error.clear();
  return true;
}

bool read_texture_import(const nx::string_view asset_meta_json,
                         TextureImport &out, nx::string &error) {
  static constexpr nx::string_view FIELDS[] = {"version", "preset", "format",
                                               "mipmaps", "quality"};
  error.clear();
  const auto parsed = nx::json::parse(
      asset_meta_json, {.reject_duplicate_keys = true, .validate_utf8 = true});
  if (!parsed || !parsed->is_object()) {
    error = "asset metadata must be a JSON object";
    return false;
  }
  const nx::json::value *const texture = parsed->find("texture");
  if (texture == nullptr) {
    out = {};
    error.clear();
    return true;
  }
  if (!texture->is_object()) {
    error = "asset metadata.texture must be an object";
    return false;
  }
  if (!known_fields(*texture, FIELDS, "asset metadata.texture", error) ||
      !read_version(*texture, IMPORT_VERSION, "asset metadata.texture", error))
    return false;

  TextureImport built;
  if (const nx::json::value *preset = texture->find("preset")) {
    if (!preset->is_string() || !portable_name(preset->get_string())) {
      error = "asset metadata.texture.preset must be a portable preset name";
      return false;
    }
    built.preset = nx::string(preset->get_string());
  }
  if (const nx::json::value *format = texture->find("format")) {
    Format parsed_format = Format::Unknown;
    if (!read_target_format(*format, parsed_format, "asset metadata.texture",
                            error))
      return false;
    built.format = parsed_format;
  }
  if (const nx::json::value *mipmaps = texture->find("mipmaps")) {
    if (!mipmaps->is_bool()) {
      error = "asset metadata.texture.mipmaps must be true or false";
      return false;
    }
    built.mipmaps = mipmaps->as_bool();
  }
  if (const nx::json::value *quality = texture->find("quality")) {
    TextureCookQuality parsed_quality{};
    if (!read_quality(*quality, parsed_quality, "asset metadata.texture",
                      error))
      return false;
    built.quality = parsed_quality;
  }
  out = std::move(built);
  error.clear();
  return true;
}

bool resolve_texture_settings(const TextureProfile &profile,
                              const TextureImport &import,
                              TextureCookSettings &out, nx::string &error) {
  error.clear();
  const nx::string_view name = import.preset.empty()
                                   ? profile.default_preset.view()
                                   : import.preset.view();
  const TexturePreset *const preset = profile.find(name);
  if (preset == nullptr) {
    error = nx::format("texture preset '{}' does not exist in profile '{}'",
                       name, profile.name.view());
    return false;
  }
  TextureCookSettings built{.container = profile.container,
                            .format = import.format.value_or(preset->format),
                            .mipmaps = import.mipmaps.value_or(preset->mipmaps),
                            .quality = import.quality.value_or(preset->quality),
                            .preset = nx::string(name)};
  if (!texture_container_supports(built.container, built.format)) {
    error =
        nx::format("texture override format {} cannot be stored through a "
                   "{} encoder",
                   built.format == Format::Unknown ? nx::string_view("source")
                                                   : to_string(built.format),
                   to_string(built.container));
    return false;
  }
  out = std::move(built);
  error.clear();
  return true;
}

} // namespace nxe::rhi
