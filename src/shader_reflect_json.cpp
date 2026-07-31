/**
 * @file shader_reflect_json.cpp
 * @brief Populates ShaderReflection from slangc's -reflection-json sidecar.
 *
 * One parser for both backends. Names, field offsets and threadGroupSize are
 * the same whichever target the JSON was generated for; binding topology is
 * not, and is not meant to be - Metal packs the bindless arrays into a single
 * argument buffer where Vulkan has three descriptor slots. Each build reads
 * the sidecar generated for its own target, so that difference is correct.
 */

#include "core/rendering/rhi/shader_reflect.h"

#include "core/foundation/diagnostics/log.h"
#include "core/foundation/serialization/json.h"
#include "core/foundation/serialization/json_util.h"

namespace nxe::rhi {
namespace {

const nx::log::Category log_rhi = nx::log::category("rhi");

/// Slang names the same concept differently per target: a push block is
/// "pushConstantBuffer" for SPIR-V but "constantBuffer" for Metal.
[[nodiscard]] ReflectKind to_kind(const nx::string_view kind,
                                  const nx::string_view name) noexcept {
  if (kind == "pushConstantBuffer")
    return ReflectKind::PushConstant;
  if (kind == "constantBuffer")
    // Metal has no push constants; the block Slang calls "pc" is still the one
    // the engine pushes, so it keeps the same kind on both backends.
    return name == "pc" ? ReflectKind::PushConstant
                        : ReflectKind::UniformBuffer;
  if (kind == "shaderResource" || kind == "descriptorTableSlot")
    return ReflectKind::SampledTexture;
  if (kind == "unorderedAccess")
    return ReflectKind::StorageTexture;
  if (kind == "samplerState")
    return ReflectKind::Sampler;
  return ReflectKind::StorageBuffer;
}

void read_fields(const nx::json::value &element_type, ReflectBlock &out) {
  const nx::json::value &fields = element_type["fields"];
  if (!fields.is_array())
    return;

  for (usize i = 0; i < fields.size(); ++i) {
    const nx::json::value &field = fields[i];
    const nx::string_view name = nx::json::string_or(field, "name", {});
    if (name.empty())
      continue;
    const nx::json::value &binding = field["binding"];
    out.members.push_back(ReflectMember{
        .name = nx::id_string(name),
        .offset = nx::json::u32_or(binding, "offset", 0),
        .size = nx::json::u32_or(binding, "size", 0),
        .array_count = 1,
    });
  }
}

} // namespace

bool reflect_from_json(const nx::string_view text, ShaderReflection &out) {
  out.blocks.clear();
  out.compute_entries.clear();

  auto doc = nx::json::parse(text);
  if (!doc) {
    nx::loge(log_rhi, "shader reflection: {}", doc.error().message());
    return false;
  }
  const nx::json::value &root = doc.value();

  const nx::json::value &params = root["parameters"];
  if (params.is_array()) {
    for (usize i = 0; i < params.size(); ++i) {
      const nx::json::value &param = params[i];
      const nx::string_view name =
          nx::json::string_or(param, "name", "parameter");
      const nx::json::value &binding = param["binding"];

      ReflectBlock block;
      block.name = nx::id_string(name);
      block.kind = to_kind(nx::json::string_or(binding, "kind", {}), name);
      block.binding = nx::json::u32_or(binding, "index", 0);
      block.set = nx::json::u32_or(binding, "space", 0);

      const nx::json::value &element = param["type"]["elementType"];
      block.size = nx::json::u32_or(element, "size", 0);
      read_fields(element, block);
      // Slang states a block's size on its fields rather than on the block for
      // a push constant buffer, so derive it when it is not given.
      if (block.size == 0 && !block.members.empty()) {
        const ReflectMember &last = block.members.back();
        block.size = last.offset + last.size;
      }

      out.blocks.push_back(std::move(block));
    }
  }

  const nx::json::value &entries = root["entryPoints"];
  if (entries.is_array()) {
    for (usize i = 0; i < entries.size(); ++i) {
      const nx::json::value &entry = entries[i];
      if (nx::json::string_or(entry, "stage", {}) != "compute")
        continue;

      ReflectComputeEntry out_entry;
      out_entry.name = nx::id_string(nx::json::string_or(entry, "name", {}));
      const nx::json::value &tgs = entry["threadGroupSize"];
      if (tgs.is_array() && tgs.size() == 3)
        for (usize axis = 0; axis < 3; ++axis)
          out_entry.local_size[axis] = nx::cast<u32>(tgs[axis].get_u64(0));
      out.compute_entries.push_back(std::move(out_entry));
    }
  }

  return true;
}

} // namespace nxe::rhi
