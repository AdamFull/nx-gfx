#include "rendering/rhi/shader_reflect.h"

#include "core/foundation/diagnostics/log.h"

#include <cstring>

namespace nxe::rhi {

const ReflectMember *
ReflectBlock::member(const nx::id_string field) const noexcept {
  for (const ReflectMember &m : members)
    if (m.name == field)
      return &m;
  return nullptr;
}

const ReflectBlock *
ShaderReflection::block(const nx::id_string name) const noexcept {
  for (const ReflectBlock &b : blocks)
    if (b.name == name)
      return &b;
  return nullptr;
}

const ReflectBlock *ShaderReflection::push_block() const noexcept {
  for (const ReflectBlock &b : blocks)
    if (b.kind == ReflectKind::PushConstant)
      return &b;
  return nullptr;
}

bool ShaderReflection::local_size(const nx::id_string entry_point,
                                  u32 out_size[3]) const noexcept {
  for (const ReflectComputeEntry &e : compute_entries) {
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

BlockWriter::BlockWriter(const ReflectBlock &block) : m_block(&block) {
  m_data.resize(block.size);
  clear();
}

void BlockWriter::clear() noexcept {
  if (!m_data.empty())
    std::memset(m_data.data(), 0, m_data.size());
}

bool BlockWriter::set(const nx::id_string field, const void *data,
                      const u32 size) noexcept {
  if (m_block == nullptr || data == nullptr)
    return false;

  const ReflectMember *member = m_block->member(field);
  if (member == nullptr) {
    nx::loge(nx::log::rhi, "block '{}' has no field '{}'", m_block->name.view(),
             field.view());
    return false;
  }
  if (size != member->size) {
    nx::loge(nx::log::rhi, "'{}.{}': writing {} bytes into a {}-byte field",
             m_block->name.view(), field.view(), size, member->size);
    return false;
  }
  // Reflection is the only source of the offset, but a corrupt or unexpected
  // module must not turn into a heap write.
  if (member->offset + size > m_data.size())
    return false;

  std::memcpy(m_data.data() + member->offset, data, size);
  return true;
}

bool BlockWriter::set_all(const void *data, const u32 size) noexcept {
  if (m_block == nullptr || data == nullptr)
    return false;
  if (size != m_data.size()) {
    nx::loge(nx::log::rhi, "block '{}' is {} bytes, not {}", m_block->name.view(),
             m_data.size(), size);
    return false;
  }
  std::memcpy(m_data.data(), data, size);
  return true;
}

}
