#include "rendering/rhi/vulkan/vk_reflection.h"

#include <spirv_reflect.h>

#include <cstring>

namespace nxe::rhi {
namespace {

[[nodiscard]] u32 word_at(const std::span<const u8> code,
                          const usize index) noexcept {
  u32 word = 0;
  std::memcpy(&word, code.data() + index * sizeof(u32), sizeof(u32));
  return word;
}

/// Bounds-checks the instruction stream.
///
/// SPIRV-Reflect assumes it is handed a module that already parsed: on a
/// truncated or corrupt one it trips an assert in debug and reads past the end
/// in release. Shader blobs come from a mounted archive, so they get walked
/// once here before it ever sees them.
[[nodiscard]] bool well_formed(const std::span<const u8> code) noexcept {
  constexpr usize HEADER_WORDS = 5;
  if (code.size() < HEADER_WORDS * sizeof(u32) || (code.size() % 4) != 0)
    return false;

  const usize count = code.size() / sizeof(u32);
  if (word_at(code, 0) != 0x07230203u)
    return false;

  usize i = HEADER_WORDS;
  while (i < count) {
    const u32 first = word_at(code, i);
    const u32 length = first >> 16;
    // A zero length would not advance, and one past the end would over-read.
    if (length == 0 || i + length > count)
      return false;

    // Word offset at which each string-bearing instruction's literal starts.
    // A literal that never terminates inside its own instruction sends
    // SPIRV-Reflect walking off the end, and the length check above does not
    // catch it - the instruction is the size it claims to be.
    u32 string_start = 0;
    switch (first & 0xFFFFu) {
    case 5:  // OpName
    case 7:  // OpString
    case 11: // OpExtInstImport
      string_start = 2;
      break;
    case 6: // OpMemberName
    case 15: // OpEntryPoint
      string_start = 3;
      break;
    case 4:   // OpSourceExtension
    case 10:  // OpExtension
    case 330: // OpModuleProcessed
      string_start = 1;
      break;
    default:
      break;
    }

    if (string_start != 0) {
      if (length <= string_start)
        return false;
      bool terminated = false;
      for (usize w = i + string_start; w < i + length && !terminated; ++w) {
        const u32 word = word_at(code, w);
        for (u32 b = 0; b < 4; ++b)
          if (((word >> (b * 8)) & 0xFFu) == 0) {
            terminated = true;
            break;
          }
      }
      if (!terminated)
        return false;
    }

    i += length;
  }
  return i == count;
}

} // namespace

usize spirv_compute_entries(const std::span<const u8> code,
                            const std::span<SpirvComputeEntry> out) noexcept {
  if (out.empty() || !well_formed(code))
    return 0;

  // NO_COPY leaves entry point names pointing into `code` rather than into the
  // module's own storage, which is the lifetime this header promises.
  SpvReflectShaderModule module{};
  if (spvReflectCreateShaderModule2(SPV_REFLECT_MODULE_FLAG_NO_COPY,
                                    code.size(), code.data(),
                                    &module) != SPV_REFLECT_RESULT_SUCCESS)
    return 0;

  usize written = 0;
  for (u32 i = 0; i < module.entry_point_count && written < out.size(); ++i) {
    const SpvReflectEntryPoint &entry = module.entry_points[i];
    if (entry.spirv_execution_model != SpvExecutionModelGLCompute)
      continue;

    out[written].name = nx::string_view(entry.name);
    out[written].local_size[0] = entry.local_size.x;
    out[written].local_size[1] = entry.local_size.y;
    out[written].local_size[2] = entry.local_size.z;
    ++written;
  }

  spvReflectDestroyShaderModule(&module);
  return written;
}

namespace {

[[nodiscard]] ReflectKind to_kind(const SpvReflectDescriptorType type) noexcept {
  switch (type) {
  case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER:
    return ReflectKind::Sampler;
  case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE:
    return ReflectKind::StorageTexture;
  case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
  case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
    return ReflectKind::StorageBuffer;
  case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
  case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
    return ReflectKind::UniformBuffer;
  default:
    return ReflectKind::SampledTexture;
  }
}

/// Flattens one level of a block's members. Nested structs are not expanded:
/// their offsets are absolute already, so a caller can still write the outer
/// struct whole.
void fill_members(const SpvReflectBlockVariable &block, ReflectBlock &out) {
  for (u32 i = 0; i < block.member_count; ++i) {
    const SpvReflectBlockVariable &m = block.members[i];
    if (m.name == nullptr)
      continue;
    out.members.push_back(ReflectMember{
        .name = nx::id_string(nx::string_view(m.name)),
        .offset = m.offset,
        .size = m.size,
        .array_count = m.array.dims_count > 0 ? m.array.dims[0] : 1u,
    });
  }
}

} // namespace

bool spirv_reflect_module(const std::span<const u8> code,
                          ShaderReflection &out) noexcept {
  out.blocks.clear();
  out.compute_entries.clear();

  if (!well_formed(code))
    return false;

  SpvReflectShaderModule module{};
  if (spvReflectCreateShaderModule2(SPV_REFLECT_MODULE_FLAG_NO_COPY,
                                    code.size(), code.data(),
                                    &module) != SPV_REFLECT_RESULT_SUCCESS)
    return false;

  for (u32 i = 0; i < module.entry_point_count; ++i) {
    const SpvReflectEntryPoint &entry = module.entry_points[i];
    if (entry.spirv_execution_model != SpvExecutionModelGLCompute)
      continue;
    out.compute_entries.push_back(ReflectComputeEntry{
        .name = nx::id_string(nx::string_view(entry.name)),
        .local_size = {entry.local_size.x, entry.local_size.y,
                       entry.local_size.z},
    });
  }

  for (u32 i = 0; i < module.push_constant_block_count; ++i) {
    const SpvReflectBlockVariable &pc = module.push_constant_blocks[i];
    ReflectBlock block;
    block.name = nx::id_string(
        nx::string_view(pc.name != nullptr ? pc.name : "push_constants"));
    block.kind = ReflectKind::PushConstant;
    block.size = pc.size;
    fill_members(pc, block);
    out.blocks.push_back(std::move(block));
  }

  for (u32 i = 0; i < module.descriptor_binding_count; ++i) {
    const SpvReflectDescriptorBinding &binding = module.descriptor_bindings[i];
    ReflectBlock block;
    block.name = nx::id_string(
        nx::string_view(binding.name != nullptr && binding.name[0] != '\0'
                            ? binding.name
                            : binding.type_description != nullptr &&
                                      binding.type_description->type_name !=
                                          nullptr
                                  ? binding.type_description->type_name
                                  : "binding"));
    block.kind = to_kind(binding.descriptor_type);
    block.set = binding.set;
    block.binding = binding.binding;
    block.size = binding.block.size;
    if (block.kind == ReflectKind::UniformBuffer ||
        block.kind == ReflectKind::StorageBuffer)
      fill_members(binding.block, block);
    out.blocks.push_back(std::move(block));
  }

  spvReflectDestroyShaderModule(&module);
  return true;
}

bool spirv_local_size(const std::span<const u8> code,
                      const nx::string_view entry_point,
                      u32 out_size[3]) noexcept {
  SpirvComputeEntry entries[16];
  const usize count = spirv_compute_entries(code, entries);

  for (usize i = 0; i < count; ++i) {
    if (entries[i].name != entry_point)
      continue;
    if (entries[i].local_size[0] == 0)
      return false; // Present, but the size is not stated as literals.
    out_size[0] = entries[i].local_size[0];
    out_size[1] = entries[i].local_size[1];
    out_size[2] = entries[i].local_size[2];
    return true;
  }
  return false;
}

} // namespace nxe::rhi
