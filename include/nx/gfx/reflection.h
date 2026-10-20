#pragma once

#include "core/foundation/containers/small_vector.h"
#include "core/foundation/strings/id_string.h"
#include "nx/gfx/types.h"

namespace nx::gfx {

enum class reflect_kind : u8 {
  PushConstant,
  UniformBuffer,
  StorageBuffer,
  SampledTexture,
  StorageTexture,
  Sampler,
};

struct reflect_member {
  nx::id_string name;
  u32 offset = 0;
  u32 size = 0;
  u32 array_count = 1;
};

struct reflect_block {
  nx::id_string name;
  reflect_kind kind = reflect_kind::PushConstant;
  u32 set = 0;
  u32 binding = 0;
  u32 size = 0;
  nx::small_vector<reflect_member, 12> members;

  [[nodiscard]] const reflect_member *
  member(nx::id_string field) const noexcept;
};

struct reflect_compute_entry {
  nx::id_string name;
  u32 local_size[3] = {};
};

/// A shader's interface. Owned by the RHI, one per shader_handle.
struct shader_reflection {
  nx::small_vector<reflect_block, 8> blocks;
  nx::small_vector<reflect_compute_entry, 4> compute_entries;

  [[nodiscard]] const reflect_block *block(nx::id_string name) const noexcept;
  [[nodiscard]] const reflect_block *push_block() const noexcept;
  [[nodiscard]] bool local_size(nx::id_string entry_point,
                                u32 out_size[3]) const noexcept;
};

[[nodiscard]] bool reflect_from_json(nx::string_view text,
                                     shader_reflection &out);

} // namespace nx::gfx
