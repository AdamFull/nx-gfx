#pragma once

#include "core/foundation/containers/small_vector.h"
#include "core/foundation/strings/id_string.h"
#include "core/rendering/rhi/types.h"

#include <span>

namespace nxe::rhi {

enum class ReflectKind : u8 {
  PushConstant,
  UniformBuffer,
  StorageBuffer,
  SampledTexture,
  StorageTexture,
  Sampler,
};

struct ReflectMember {
  nx::id_string name;
  u32 offset = 0;
  u32 size = 0;
  u32 array_count = 1;
};

struct ReflectBlock {
  nx::id_string name;
  ReflectKind kind = ReflectKind::PushConstant;
  u32 set = 0;
  u32 binding = 0;
  u32 size = 0;
  nx::small_vector<ReflectMember, 12> members;

  [[nodiscard]] const ReflectMember *member(nx::id_string field) const noexcept;
};

struct ReflectComputeEntry {
  nx::id_string name;
  u32 local_size[3] = {};
};

/// A shader's interface. Owned by the RHI, one per ShaderHandle.
struct ShaderReflection {
  nx::small_vector<ReflectBlock, 8> blocks;
  nx::small_vector<ReflectComputeEntry, 4> compute_entries;

  [[nodiscard]] const ReflectBlock *block(nx::id_string name) const noexcept;
  [[nodiscard]] const ReflectBlock *push_block() const noexcept;
  [[nodiscard]] bool local_size(nx::id_string entry_point,
                                u32 out_size[3]) const noexcept;
};

[[nodiscard]] bool reflect_from_json(nx::string_view text,
                                     ShaderReflection &out);

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

}
