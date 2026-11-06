#include "framework/nxtest.h"

#include "core/foundation/platform/filesystem.h"
#include "core/foundation/serialization/json.h"
#include "core/foundation/serialization/json_util.h"
#include "core/foundation/strings/format.h"
#include "gfx_interop_probe.h"
#include "nx/gfx/shaders/bvh_layout.h"
#include "nx/gfx/shaders/ray_layout.h"

#include <cstddef>
#include <cstdio>
#include <span>

namespace {

struct Field {
  nx::string_view name;
  usize offset = 0;
  usize size = 0;
};

struct Shared {
  nx::string_view name;
  std::span<const Field> fields;
};

#define NX_FIELD(type, field)                                                  \
  Field { #field, offsetof(type, field), sizeof(type::field) }

constexpr Field GFX_INTEROP_PROBE_FIELDS[] = {
    NX_FIELD(GfxInteropProbe, words),   NX_FIELD(GfxInteropProbe, texture),
    NX_FIELD(GfxInteropProbe, storage), NX_FIELD(GfxInteropProbe, vec2),
    NX_FIELD(GfxInteropProbe, uvec2),   NX_FIELD(GfxInteropProbe, ivec2),
    NX_FIELD(GfxInteropProbe, pad0),    NX_FIELD(GfxInteropProbe, pad1),
    NX_FIELD(GfxInteropProbe, vec4),    NX_FIELD(GfxInteropProbe, uvec4),
    NX_FIELD(GfxInteropProbe, ivec4),   NX_FIELD(GfxInteropProbe, matrix)};

constexpr Field RAY_INSTANCE_FIELDS[] = {
    NX_FIELD(NxRayInstance, row0),
    NX_FIELD(NxRayInstance, row1),
    NX_FIELD(NxRayInstance, row2),
    NX_FIELD(NxRayInstance, custom_index_and_mask),
    NX_FIELD(NxRayInstance, flags),
    NX_FIELD(NxRayInstance, bottom_level)};

constexpr Field BVH_HEADER_FIELDS[] = {NX_FIELD(NxBvhHeader, magic),
                                       NX_FIELD(NxBvhHeader, version),
                                       NX_FIELD(NxBvhHeader, node_count),
                                       NX_FIELD(NxBvhHeader, primitive_count),
                                       NX_FIELD(NxBvhHeader, level_count),
                                       NX_FIELD(NxBvhHeader, flags),
                                       NX_FIELD(NxBvhHeader, nodes_offset),
                                       NX_FIELD(NxBvhHeader, primitives_offset),
                                       NX_FIELD(NxBvhHeader, levels_offset),
                                       NX_FIELD(NxBvhHeader, bounds_offset),
                                       NX_FIELD(NxBvhHeader, lo_x),
                                       NX_FIELD(NxBvhHeader, lo_y),
                                       NX_FIELD(NxBvhHeader, lo_z),
                                       NX_FIELD(NxBvhHeader, hi_x),
                                       NX_FIELD(NxBvhHeader, hi_y),
                                       NX_FIELD(NxBvhHeader, hi_z)};

constexpr Field BVH_NODE_FIELDS[] = {
    NX_FIELD(NxBvhNode, origin_x), NX_FIELD(NxBvhNode, origin_y),
    NX_FIELD(NxBvhNode, origin_z), NX_FIELD(NxBvhNode, meta),
    NX_FIELD(NxBvhNode, lo_x),     NX_FIELD(NxBvhNode, lo_y),
    NX_FIELD(NxBvhNode, lo_z),     NX_FIELD(NxBvhNode, hi_x),
    NX_FIELD(NxBvhNode, hi_y),     NX_FIELD(NxBvhNode, hi_z),
    NX_FIELD(NxBvhNode, parent),   NX_FIELD(NxBvhNode, reserved),
    NX_FIELD(NxBvhNode, children)};

constexpr Field BVH_TRIANGLE_FIELDS[] = {NX_FIELD(NxBvhTriangle, v0),
                                         NX_FIELD(NxBvhTriangle, v1),
                                         NX_FIELD(NxBvhTriangle, v2)};

constexpr Field BVH_INSTANCE_FIELDS[] = {
    NX_FIELD(NxBvhInstance, row0),
    NX_FIELD(NxBvhInstance, row1),
    NX_FIELD(NxBvhInstance, row2),
    NX_FIELD(NxBvhInstance, bottom_level),
    NX_FIELD(NxBvhInstance, custom_index_and_mask),
    NX_FIELD(NxBvhInstance, flags)};

constexpr Field BVH_BOUNDS_FIELDS[] = {
    NX_FIELD(NxBvhBounds, lo_x), NX_FIELD(NxBvhBounds, lo_y),
    NX_FIELD(NxBvhBounds, lo_z), NX_FIELD(NxBvhBounds, hi_x),
    NX_FIELD(NxBvhBounds, hi_y), NX_FIELD(NxBvhBounds, hi_z),
    NX_FIELD(NxBvhBounds, pad0), NX_FIELD(NxBvhBounds, pad1)};

#undef NX_FIELD

constexpr Shared SHARED[] = {
    {"GfxInteropProbe", GFX_INTEROP_PROBE_FIELDS},
    {"NxRayInstance", RAY_INSTANCE_FIELDS},
    {"NxBvhHeader", BVH_HEADER_FIELDS},
    {"NxBvhNode", BVH_NODE_FIELDS},
    {"NxBvhTriangle", BVH_TRIANGLE_FIELDS},
    {"NxBvhBounds", BVH_BOUNDS_FIELDS},
    {"NxBvhInstance", BVH_INSTANCE_FIELDS},
};

/// The struct a buffer parameter holds, or null.
[[nodiscard]] const nx::json::value *element_of(const nx::json::value &param) {
  const nx::json::value &type = param["type"];
  for (const nx::string_view key : {"resultType", "elementType"}) {
    const nx::json::value &element = type[key];
    if (nx::json::string_or(element, "kind", {}) == "struct")
      return &element;
  }
  return nullptr;
}

[[nodiscard]] const nx::json::value *reflected(const nx::json::value &root,
                                               const nx::string_view name) {
  const nx::json::value &params = root["parameters"];
  for (usize i = 0; i < params.size(); ++i) {
    const nx::json::value *const element = element_of(params[i]);
    if (element != nullptr && nx::json::string_or(*element, "name", {}) == name)
      return element;
  }
  return nullptr;
}

} // namespace

TEST_CASE("gfx interop: the shader library's shared types have the same "
          "fields at the same offsets in C++ and Slang") {
  const nx::string path =
      nx::string(NX_GFX_TEST_SHADER_DIR) + "/gfx_interop_layout.refl.json";
  auto text = nx::fs::file_read_text(nx::fs::path_view(path));
  if (!text.has_value())
    SKIP("test shaders are not built in this configuration");
  auto doc = nx::json::parse(text->view());
  REQUIRE(doc.has_value());
  const nx::json::value &root = doc.value();

  for (const Shared &shared : SHARED) {
    const nx::json::value *const slang = reflected(root, shared.name);
    REQUIRE(slang != nullptr);
    const nx::json::value &fields = (*slang)["fields"];
    nx::string mismatches;
    if (fields.size() != shared.fields.size())
      mismatches += nx::format(" {} fields in Slang, {} in C++;", fields.size(),
                               shared.fields.size());
    for (usize i = 0; i < fields.size(); ++i) {
      const nx::string_view name = nx::json::string_or(fields[i], "name", {});
      const nx::json::value &binding = fields[i]["binding"];
      const usize offset = nx::json::u32_or(binding, "offset", ~0u);
      const usize size = nx::json::u32_or(binding, "size", ~0u);
      const Field *cpp = nullptr;
      for (const Field &field : shared.fields)
        if (field.name == name)
          cpp = &field;
      if (cpp == nullptr)
        mismatches += nx::format(" '{}' is not in C++'s list;", name);
      else if (cpp->offset != offset || cpp->size != size)
        mismatches += nx::format(
            " '{}' is {} bytes at {} in Slang, {} bytes at {} in C++;", name,
            size, offset, cpp->size, cpp->offset);
    }
    if (!mismatches.empty())
      std::fprintf(stderr, "%s:%s\n", shared.name.data(), mismatches.c_str());
    CHECK(mismatches.empty());
  }
}
