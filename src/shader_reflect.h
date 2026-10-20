#pragma once

#include "core/foundation/containers/small_vector.h"
#include "nx/gfx/reflection.h"

#include <span>

namespace nxe::rhi {

// The names before nx::gfx, until every caller has moved to nx/gfx (G6).
using ReflectKind = nx::gfx::reflect_kind;
using ReflectMember = nx::gfx::reflect_member;
using ReflectBlock = nx::gfx::reflect_block;
using ReflectComputeEntry = nx::gfx::reflect_compute_entry;
using ShaderReflection = nx::gfx::shader_reflection;

using nx::gfx::reflect_from_json;

class BlockWriter {
public:
  BlockWriter() = default;
  explicit BlockWriter(const ReflectBlock &block);

  [[nodiscard]] bool valid() const noexcept { return m_block != nullptr; }
  [[nodiscard]] const ReflectBlock *block() const noexcept { return m_block; }

  bool set(nx::id_string field, const void *data, u32 size) noexcept;

  template <typename T>
  bool set(const nx::id_string field, const T &value) noexcept {
    return set(field, &value, nx::cast<u32>(sizeof(T)));
  }

  /// Replaces the whole block. @p size must equal the block's size.
  bool set_all(const void *data, u32 size) noexcept;

  template <typename T> bool set_all(const T &value) noexcept {
    return set_all(&value, nx::cast<u32>(sizeof(T)));
  }

  [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
    return {m_data.data(), m_data.size()};
  }

  void clear() noexcept;

private:
  const ReflectBlock *m_block = nullptr;
  nx::small_vector<std::byte, 128> m_data;
};

} // namespace nxe::rhi
