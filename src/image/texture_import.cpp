#include "rendering/rhi/image/texture_import.h"

#include "core/foundation/serialization/json.h"
#include "core/foundation/strings/format.h"

namespace nxe::rhi {
namespace {

constexpr u64 LEGACY_VERSION = 1;
/// Version 3 adds maxSize, and to `.meta` blocks per-profile overrides; 2
/// stays readable.
constexpr u64 PROFILE_VERSION = 3;
constexpr u64 IMPORT_VERSION = 3;
constexpr u32 MAX_TEXTURE_SIZE = 16384;

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

[[nodiscard]] std::optional<TextureStorage>
storage_from_string(const nx::string_view name) noexcept {
  if (name == "native")
    return TextureStorage::Native;
  if (name == "basis_etc1s")
    return TextureStorage::BasisEtc1s;
  if (name == "basis_uastc")
    return TextureStorage::BasisUastc;
  return std::nullopt;
}

[[nodiscard]] bool read_storage(const nx::json::value &value,
                                TextureStorage &out,
                                const nx::string_view owner,
                                nx::string &error) {
  if (!value.is_string()) {
    error = nx::format("{}.storage must be a string", owner);
    return false;
  }
  const auto parsed = storage_from_string(value.get_string());
  if (!parsed) {
    error = nx::format("{}.storage must be native, basis_etc1s, or basis_uastc",
                       owner);
    return false;
  }
  out = *parsed;
  return true;
}

[[nodiscard]] bool read_zstd_level(const nx::json::value &value, i32 &out,
                                   const nx::string_view owner,
                                   nx::string &error) {
  if (!value.fits<i64>() || value.get_i64() < 1 || value.get_i64() > 22) {
    error = nx::format("{}.zstdLevel must be an integer from 1 to 22", owner);
    return false;
  }
  out = nx::cast<i32>(value.get_i64());
  return true;
}

[[nodiscard]] bool read_max_size(const nx::json::value &value, u32 &out,
                                 const nx::string_view owner,
                                 nx::string &error) {
  if (!value.fits<i64>() || value.get_i64() < 1 ||
      value.get_i64() > MAX_TEXTURE_SIZE) {
    error = nx::format("{}.maxSize must be an integer from 1 to {}", owner,
                       MAX_TEXTURE_SIZE);
    return false;
  }
  out = nx::cast<u32>(value.get_i64());
  return true;
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

nx::string_view to_string(const TextureStorage storage) noexcept {
  switch (storage) {
  case TextureStorage::Native:
    return "native";
  case TextureStorage::BasisEtc1s:
    return "basis_etc1s";
  case TextureStorage::BasisUastc:
    return "basis_uastc";
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
  static constexpr nx::string_view ROOT_FIELDS_V1[] = {
      "version", "name", "container", "defaultPreset", "presets"};
  static constexpr nx::string_view ROOT_FIELDS_V2[] = {
      "version", "name", "defaultPreset", "presets"};
  static constexpr nx::string_view PRESET_FIELDS_V1[] = {"format", "mipmaps",
                                                         "quality"};
  static constexpr nx::string_view PRESET_FIELDS_V2[] = {
      "storage", "format", "mipmaps", "quality", "zstdLevel"};
  static constexpr nx::string_view PRESET_FIELDS_V3[] = {
      "storage", "format", "mipmaps", "quality", "zstdLevel", "maxSize"};
  error.clear();

  const auto parsed = nx::json::parse(
      json, {.reject_duplicate_keys = true, .validate_utf8 = true});
  if (!parsed || !parsed->is_object()) {
    error = "texture profile must be a JSON object";
    return false;
  }
  const nx::json::value &root = *parsed;
  const nx::json::value *const version_value = root.find("version");
  if (version_value == nullptr || !version_value->fits<u64>() ||
      version_value->get_u64() < LEGACY_VERSION ||
      version_value->get_u64() > PROFILE_VERSION) {
    error = nx::format("texture profile version must be from {} to {}",
                       LEGACY_VERSION, PROFILE_VERSION);
    return false;
  }
  const u64 version = version_value->get_u64();
  if (!known_fields(root,
                    version == LEGACY_VERSION
                        ? std::span<const nx::string_view>(ROOT_FIELDS_V1)
                        : std::span<const nx::string_view>(ROOT_FIELDS_V2),
                    "texture profile", error))
    return false;

  const nx::json::value *const name = root.find("name");
  if (name == nullptr || !name->is_string() ||
      !portable_name(name->get_string())) {
    error = "texture profile.name must be a portable lower-case name";
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
  if (version == LEGACY_VERSION) {
    const nx::json::value *const container = root.find("container");
    const auto parsed_container =
        container != nullptr && container->is_string()
            ? container_from_string(container->get_string())
            : std::nullopt;
    if (!parsed_container) {
      error = "texture profile.container must be nxb, dds, or ktx";
      return false;
    }
    built.container = *parsed_container;
  }
  built.default_preset = nx::string(fallback->get_string());
  built.presets.reserve(presets->size());
  for (const nx::json::member &member : presets->as_object()) {
    const nx::string_view preset_name = member.key.view();
    const nx::string owner = nx::format("texture preset '{}'", preset_name);
    if (!portable_name(preset_name)) {
      error = nx::format("{} is not a portable lower-case name", owner.view());
      return false;
    }
    const std::span<const nx::string_view> preset_fields =
        version == LEGACY_VERSION
            ? std::span<const nx::string_view>(PRESET_FIELDS_V1)
        : version == 2 ? std::span<const nx::string_view>(PRESET_FIELDS_V2)
                       : std::span<const nx::string_view>(PRESET_FIELDS_V3);
    if (!member.val.is_object() ||
        !known_fields(member.val, preset_fields, owner.view(), error)) {
      if (error.empty())
        error = nx::format("{} must be an object", owner.view());
      return false;
    }
    const nx::json::value *const format = member.val.find("format");
    const nx::json::value *const mipmaps = member.val.find("mipmaps");
    const nx::json::value *const quality = member.val.find("quality");
    const nx::json::value *const storage = member.val.find("storage");
    const nx::json::value *const zstd_level = member.val.find("zstdLevel");
    const nx::json::value *const max_size = member.val.find("maxSize");
    TexturePreset preset;
    preset.name = nx::string(preset_name);
    if (max_size != nullptr &&
        !read_max_size(*max_size, preset.max_size, owner.view(), error))
      return false;
    if (version != LEGACY_VERSION) {
      if (storage == nullptr ||
          !read_storage(*storage, preset.storage, owner.view(), error))
        return false;
      if (preset.storage == TextureStorage::Native) {
        if (format == nullptr ||
            !read_target_format(*format, preset.format, owner.view(), error))
          return false;
      } else if (format != nullptr) {
        error = nx::format("{}.format is selected at runtime for Basis storage",
                           owner.view());
        return false;
      }
      if (zstd_level != nullptr) {
        if (preset.storage != TextureStorage::BasisUastc) {
          error =
              nx::format("{}.zstdLevel is only valid for basis_uastc storage",
                         owner.view());
          return false;
        }
        if (!read_zstd_level(*zstd_level, preset.zstd_level, owner.view(),
                             error))
          return false;
      }
    } else if (format == nullptr || !read_target_format(*format, preset.format,
                                                        owner.view(), error)) {
      return false;
    }
    if (mipmaps == nullptr || !mipmaps->is_bool()) {
      error = nx::format("{}.mipmaps must be true or false", owner.view());
      return false;
    }
    preset.mipmaps = mipmaps->as_bool();
    if (quality == nullptr ||
        !read_quality(*quality, preset.quality, owner.view(), error))
      return false;
    if (preset.storage == TextureStorage::Native &&
        !texture_container_supports(built.container, preset.format)) {
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

namespace {

/// Reads the fields @p object sets into @p out; the caller has checked that
/// it holds no others.
[[nodiscard]] bool read_import_fields(const nx::json::value &object,
                                      const nx::string_view owner,
                                      TextureImportFields &out,
                                      nx::string &error) {
  if (const nx::json::value *preset = object.find("preset")) {
    if (!preset->is_string() || !portable_name(preset->get_string())) {
      error = nx::format("{}.preset must be a portable preset name", owner);
      return false;
    }
    out.preset = nx::string(preset->get_string());
  }
  if (const nx::json::value *storage = object.find("storage")) {
    TextureStorage parsed_storage{};
    if (!read_storage(*storage, parsed_storage, owner, error))
      return false;
    out.storage = parsed_storage;
  }
  if (const nx::json::value *format = object.find("format")) {
    Format parsed_format = Format::Unknown;
    if (!read_target_format(*format, parsed_format, owner, error))
      return false;
    out.format = parsed_format;
  }
  if (const nx::json::value *mipmaps = object.find("mipmaps")) {
    if (!mipmaps->is_bool()) {
      error = nx::format("{}.mipmaps must be true or false", owner);
      return false;
    }
    out.mipmaps = mipmaps->as_bool();
  }
  if (const nx::json::value *quality = object.find("quality")) {
    TextureCookQuality parsed_quality{};
    if (!read_quality(*quality, parsed_quality, owner, error))
      return false;
    out.quality = parsed_quality;
  }
  if (const nx::json::value *zstd_level = object.find("zstdLevel")) {
    i32 parsed_level = 0;
    if (!read_zstd_level(*zstd_level, parsed_level, owner, error))
      return false;
    out.zstd_level = parsed_level;
  }
  if (const nx::json::value *max_size = object.find("maxSize")) {
    u32 parsed_size = 0;
    if (!read_max_size(*max_size, parsed_size, owner, error))
      return false;
    out.max_size = parsed_size;
  }
  return true;
}

template <class T>
void overlay(std::optional<T> &field, const std::optional<T> &over) {
  if (over.has_value())
    field = over;
}

} // namespace

bool read_texture_import(const nx::string_view asset_meta_json,
                         TextureImport &out, nx::string &error) {
  static constexpr nx::string_view FIELDS_V1[] = {"version", "preset", "format",
                                                  "mipmaps", "quality"};
  static constexpr nx::string_view FIELDS_V2[] = {
      "version", "preset",  "storage",  "format",
      "mipmaps", "quality", "zstdLevel"};
  static constexpr nx::string_view FIELDS_V3[] = {
      "version", "preset",    "storage", "format",  "mipmaps",
      "quality", "zstdLevel", "maxSize", "profiles"};
  static constexpr nx::string_view PROFILE_FIELDS[] = {
      "preset",  "storage",   "format", "mipmaps",
      "quality", "zstdLevel", "maxSize"};
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
  const nx::json::value *const version_value = texture->find("version");
  if (version_value == nullptr || !version_value->fits<u64>() ||
      version_value->get_u64() < LEGACY_VERSION ||
      version_value->get_u64() > IMPORT_VERSION) {
    error = nx::format("asset metadata.texture version must be from {} to {}",
                       LEGACY_VERSION, IMPORT_VERSION);
    return false;
  }
  const u64 version = version_value->get_u64();
  if (!known_fields(*texture,
                    version == LEGACY_VERSION
                        ? std::span<const nx::string_view>(FIELDS_V1)
                    : version == 2
                        ? std::span<const nx::string_view>(FIELDS_V2)
                        : std::span<const nx::string_view>(FIELDS_V3),
                    "asset metadata.texture", error))
    return false;

  TextureImport built;
  if (!read_import_fields(*texture, "asset metadata.texture", built, error))
    return false;
  if (const nx::json::value *profiles = texture->find("profiles")) {
    if (!profiles->is_object()) {
      error = "asset metadata.texture.profiles must be an object of profiles";
      return false;
    }
    built.profiles.reserve(profiles->size());
    for (const nx::json::member &member : profiles->as_object()) {
      const nx::string owner =
          nx::format("asset metadata.texture.profiles.{}", member.key.view());
      if (!portable_name(member.key.view())) {
        error = nx::format("{} is not a portable profile name", owner.view());
        return false;
      }
      if (!member.val.is_object()) {
        error = nx::format("{} must be an object", owner.view());
        return false;
      }
      TextureProfileOverride entry{.profile = nx::string(member.key.view())};
      if (!known_fields(member.val, PROFILE_FIELDS, owner.view(), error) ||
          !read_import_fields(member.val, owner.view(), entry.fields, error))
        return false;
      built.profiles.push_back(std::move(entry));
    }
  }
  out = std::move(built);
  error.clear();
  return true;
}

bool resolve_texture_settings(const TextureProfile &profile,
                              const TextureImport &import,
                              TextureCookSettings &out, nx::string &error) {
  error.clear();
  TextureImportFields fields = import;
  for (const TextureProfileOverride &entry : import.profiles) {
    if (entry.profile != profile.name)
      continue;
    if (!entry.fields.preset.empty())
      fields.preset = entry.fields.preset;
    overlay(fields.storage, entry.fields.storage);
    overlay(fields.format, entry.fields.format);
    overlay(fields.mipmaps, entry.fields.mipmaps);
    overlay(fields.quality, entry.fields.quality);
    overlay(fields.zstd_level, entry.fields.zstd_level);
    overlay(fields.max_size, entry.fields.max_size);
  }
  const nx::string_view name = fields.preset.empty()
                                   ? profile.default_preset.view()
                                   : fields.preset.view();
  const TexturePreset *const preset = profile.find(name);
  if (preset == nullptr) {
    error = nx::format("texture preset '{}' does not exist in profile '{}'",
                       name, profile.name.view());
    return false;
  }
  TextureCookSettings built{
      .container = profile.container,
      .storage = fields.storage.value_or(preset->storage),
      .format = fields.format.value_or(preset->format),
      .mipmaps = fields.mipmaps.value_or(preset->mipmaps),
      .quality = fields.quality.value_or(preset->quality),
      .zstd_level = fields.zstd_level.value_or(preset->zstd_level),
      .max_size = fields.max_size.value_or(preset->max_size),
      .preset = nx::string(name)};
  if (built.storage != TextureStorage::Native &&
      built.format != Format::Unknown) {
    error = "Basis storage selects its GPU format at runtime; remove the "
            "format override";
    return false;
  }
  if (built.storage != TextureStorage::BasisUastc &&
      fields.zstd_level.has_value()) {
    error = "zstdLevel is only valid for basis_uastc storage";
    return false;
  }
  if (built.storage == TextureStorage::Native &&
      !texture_container_supports(built.container, built.format)) {
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
