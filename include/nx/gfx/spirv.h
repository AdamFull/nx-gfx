#pragma once

#include "core/foundation/core/foundation.h"
#include "core/foundation/core/result.h"
#include "core/foundation/strings/utf8_string_view.h"

#include <span>

namespace nx::gfx {

enum class spirv_error : u8 {
  /// Not a whole number of words, or shorter than the header.
  Truncated,
  /// Not little-endian SPIR-V.
  BadMagic,
  /// An instruction of zero words, or one running past the end.
  BadInstruction,
};

/// What every SPIR-V module is put through after it is compiled and before
/// a device loads it. Today one change: slangc declares SPV_KHR_ray_tracing
/// for a ray query made from an address (OpConvertUToAccelerationStructureKHR,
/// which SPV_KHR_ray_query also provides), and a device without
/// VK_KHR_ray_tracing_pipeline refuses the module for it, so a module that
/// holds no RayTracingKHR capability loses that declaration. Works in place on
/// @p module; answers its new size in bytes, the same for a module it leaves
/// as it was, byte for byte.
[[nodiscard]] nx::result<usize, spirv_error>
normalize_spirv(std::span<u8> module) noexcept;

[[nodiscard]] nx::string_view spirv_error_name(spirv_error error) noexcept;

} // namespace nx::gfx
