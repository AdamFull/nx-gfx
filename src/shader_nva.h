#pragma once

/**
 * @file shader_nva.h
 * @brief The cooked binary shader: an nva container (SHDR) in a `.nxb` file
 * (namespace nxe::rhi).
 */

#include "core/foundation/core/foundation.h"
#include "core/foundation/strings/utf8_string.h"

#include <span>

namespace nxe::rhi {

enum class ShaderTarget : u32 { Spirv = 0, Metallib = 1 };

/// A compiled shader: its binary, the reflection JSON slangc emitted beside it,
/// and which backend the binary targets.
struct ShaderBlob {
  nx::vector<u8> code;
  nx::string reflection_json;
  ShaderTarget target = ShaderTarget::Spirv;
};

/// Packs a compiled shader into an nva container.
[[nodiscard]] nx::vector<u8> encode_shader(std::span<const u8> code,
                                           nx::string_view reflection_json,
                                           ShaderTarget target);

/// Reads a container built by encode_shader into @p out. False when the bytes
/// are not a valid shader container.
[[nodiscard]] bool decode_shader(std::span<const u8> bytes, ShaderBlob &out);

} // namespace nxe::rhi
