#include "nx/gfx/spirv.h"

#include <cstring>

namespace nx::gfx {
namespace {

constexpr u32 MAGIC = 0x07230203u;
constexpr u32 HEADER_WORDS = 5;
constexpr u32 OP_EXTENSION = 10;
constexpr u32 OP_CAPABILITY = 17;
constexpr u32 CAPABILITY_RAY_TRACING = 4479;
constexpr nx::string_view RAY_TRACING_EXTENSION = "SPV_KHR_ray_tracing";

[[nodiscard]] u32 word(const std::span<const u8> bytes,
                       const usize i) noexcept {
  u32 value;
  std::memcpy(&value, bytes.data() + i * 4, 4);
  return value;
}

/// The literal an OpExtension of @p count words starting at word @p at names,
/// up to its terminating zero or the instruction's end.
[[nodiscard]] nx::string_view extension_name(const std::span<const u8> bytes,
                                             const usize at,
                                             const u32 count) noexcept {
  const char *const first =
      reinterpret_cast<const char *>(bytes.data() + (at + 1) * 4);
  const usize most = (nx::cast<usize>(count) - 1) * 4;
  usize length = 0;
  while (length < most && first[length] != '\0')
    ++length;
  return {first, length};
}

} // namespace

nx::result<usize, spirv_error>
normalize_spirv(const std::span<u8> module) noexcept {
  if (module.size() % 4 != 0 || module.size() < HEADER_WORDS * 4)
    return nx::err(spirv_error::Truncated);
  const usize words = module.size() / 4;
  if (word(module, 0) != MAGIC)
    return nx::err(spirv_error::BadMagic);

  bool ray_tracing = false;
  for (usize at = HEADER_WORDS; at < words;) {
    const u32 head = word(module, at);
    const u32 count = head >> 16;
    if (count == 0 || count > words - at)
      return nx::err(spirv_error::BadInstruction);
    if ((head & 0xFFFFu) == OP_CAPABILITY && count >= 2 &&
        word(module, at + 1) == CAPABILITY_RAY_TRACING)
      ray_tracing = true;
    at += count;
  }
  if (ray_tracing)
    return module.size();

  usize kept = HEADER_WORDS;
  for (usize at = HEADER_WORDS; at < words;) {
    const u32 count = word(module, at) >> 16;
    const bool drop =
        (word(module, at) & 0xFFFFu) == OP_EXTENSION && count >= 2 &&
        extension_name(module, at, count) == RAY_TRACING_EXTENSION;
    if (!drop) {
      if (kept != at)
        std::memmove(module.data() + kept * 4, module.data() + at * 4,
                     nx::cast<usize>(count) * 4);
      kept += count;
    }
    at += count;
  }
  return kept * 4;
}

nx::string_view spirv_error_name(const spirv_error error) noexcept {
  switch (error) {
  case spirv_error::Truncated:
    return "truncated";
  case spirv_error::BadMagic:
    return "not little-endian SPIR-V";
  case spirv_error::BadInstruction:
    return "malformed instruction";
  }
  return "unknown";
}

} // namespace nx::gfx
