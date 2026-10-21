#include "nx/gfx/reflection.h"

namespace nx::gfx {

const reflect_member *
reflect_block::member(const nx::id_string field) const noexcept {
  for (const reflect_member &m : members)
    if (m.name == field)
      return &m;
  return nullptr;
}

const reflect_block *
shader_reflection::block(const nx::id_string name) const noexcept {
  for (const reflect_block &b : blocks)
    if (b.name == name)
      return &b;
  return nullptr;
}

const reflect_block *shader_reflection::push_block() const noexcept {
  for (const reflect_block &b : blocks)
    if (b.kind == reflect_kind::PushConstant)
      return &b;
  return nullptr;
}

bool shader_reflection::local_size(const nx::id_string entry_point,
                                   u32 out_size[3]) const noexcept {
  for (const reflect_compute_entry &e : compute_entries) {
    if (e.name != entry_point)
      continue;
    if (e.local_size[0] == 0)
      return false;
    out_size[0] = e.local_size[0];
    out_size[1] = e.local_size[1];
    out_size[2] = e.local_size[2];
    return true;
  }
  return false;
}

} // namespace nx::gfx
