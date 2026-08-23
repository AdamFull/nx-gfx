#pragma once

#include "core/foundation/core/foundation.h"
#include "core/foundation/strings/utf8_string.h"

#include <span>

namespace nxe::rhi {

enum class ShaderTarget : u32 { Spirv = 0, Metallib = 1 };

struct ShaderBlob {
  nx::vector<u8> code;
  nx::string reflection_json;
  ShaderTarget target = ShaderTarget::Spirv;
};

[[nodiscard]] nx::vector<u8> encode_shader(std::span<const u8> code,
                                           nx::string_view reflection_json,
                                           ShaderTarget target);

/// Reads a container built by encode_shader into @p out. False when the bytes
/// are not a valid shader container.
[[nodiscard]] bool decode_shader(std::span<const u8> bytes, ShaderBlob &out);

}
