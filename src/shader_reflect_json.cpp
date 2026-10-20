
#include "nx/gfx/reflection.h"

#include "core/foundation/diagnostics/log.h"
#include "core/foundation/serialization/json.h"
#include "core/foundation/serialization/json_util.h"

namespace nx::gfx {
namespace {

/// Slang names the same concept differently per target: a push block is
/// "pushConstantBuffer" for SPIR-V but "constantBuffer" for Metal.
[[nodiscard]] reflect_kind to_kind(const nx::string_view kind,
                                   const nx::string_view name) noexcept {
  if (kind == "pushConstantBuffer")
    return reflect_kind::PushConstant;
  if (kind == "constantBuffer")
    return name == "pc" ? reflect_kind::PushConstant
                        : reflect_kind::UniformBuffer;
  if (kind == "shaderResource" || kind == "descriptorTableSlot")
    return reflect_kind::SampledTexture;
  if (kind == "unorderedAccess")
    return reflect_kind::StorageTexture;
  if (kind == "samplerState")
    return reflect_kind::Sampler;
  nx::logw(nx::log::rhi,
           "shader reflection: parameter '{}' has unrecognized kind '{}'; "
           "treating it as a storage buffer",
           name, kind);
  return reflect_kind::StorageBuffer;
}

void read_fields(const nx::json::value &element_type, reflect_block &out) {
  const nx::json::value &fields = element_type["fields"];
  if (!fields.is_array())
    return;

  for (usize i = 0; i < fields.size(); ++i) {
    const nx::json::value &field = fields[i];
    const nx::string_view name = nx::json::string_or(field, "name", {});
    if (name.empty())
      continue;
    const nx::json::value &binding = field["binding"];
    out.members.push_back(reflect_member{
        .name = nx::id_string(name),
        .offset = nx::json::u32_or(binding, "offset", 0),
        .size = nx::json::u32_or(binding, "size", 0),
        .array_count = 1,
    });
  }
}
}

bool reflect_from_json(const nx::string_view text, shader_reflection &out) {
  out.blocks.clear();
  out.compute_entries.clear();

  auto doc = nx::json::parse(text);
  if (!doc) {
    nx::loge(nx::log::rhi, "shader reflection: {}", doc.error().message());
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

      reflect_block block;
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
        const reflect_member &last = block.members.back();
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

      reflect_compute_entry out_entry;
      out_entry.name = nx::id_string(nx::json::string_or(entry, "name", {}));
      const nx::json::value &tgs = entry["threadGroupSize"];
      if (tgs.is_array() && tgs.size() == 3)
        for (usize axis = 0; axis < 3; ++axis)
          out_entry.local_size[axis] = nx::cast<u32>(tgs[axis].get_u64(0));
      out.compute_entries.push_back(out_entry);
    }
  }

  return true;
}

} // namespace nx::gfx
