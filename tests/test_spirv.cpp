#include "framework/nxtest.h"

#include "core/foundation/core/foundation.h"
#include "nx/gfx/spirv.h"

#include <cstring>

namespace gfx = nx::gfx;

namespace {

constexpr u32 OP_EXTENSION = 10;
constexpr u32 OP_MEMORY_MODEL = 14;
constexpr u32 OP_CAPABILITY = 17;
constexpr u32 CAPABILITY_SHADER = 1;
constexpr u32 CAPABILITY_RAY_QUERY = 4472;
constexpr u32 CAPABILITY_RAY_TRACING = 4479;

/// A module's words, built an instruction at a time.
struct Module {
  nx::vector<u32> words{0x07230203u, 0x00010400u, 0u, 100u, 0u};

  Module &capability(const u32 capability) {
    words.push_back(2u << 16 | OP_CAPABILITY);
    words.push_back(capability);
    return *this;
  }
  Module &extension(const char *const name) {
    const usize length = std::strlen(name) + 1;
    const u32 literal = nx::cast<u32>((length + 3) / 4);
    words.push_back((literal + 1) << 16 | OP_EXTENSION);
    const usize first = words.size();
    words.resize(first + literal, 0u);
    std::memcpy(words.data() + first, name, length - 1);
    return *this;
  }
  Module &memory_model() {
    words.push_back(3u << 16 | OP_MEMORY_MODEL);
    words.push_back(0u);
    words.push_back(1u);
    return *this;
  }

  [[nodiscard]] nx::vector<u8> bytes() const {
    nx::vector<u8> out(words.size() * 4);
    std::memcpy(out.data(), words.data(), out.size());
    return out;
  }
};

/// What normalize_spirv makes of @p in: its bytes, cut to the size it answers.
[[nodiscard]] nx::vector<u8> normalized(nx::vector<u8> in) {
  const auto size = gfx::normalize_spirv({in.data(), in.size()});
  REQUIRE(size.has_value());
  in.resize(size.value());
  return in;
}

[[nodiscard]] Module ray_query_module() {
  Module m;
  m.capability(CAPABILITY_SHADER).capability(CAPABILITY_RAY_QUERY);
  return m;
}

} // namespace

TEST_CASE("gfx spirv: a ray-query module loses SPV_KHR_ray_tracing and only "
          "that") {
  Module with = ray_query_module();
  with.extension("SPV_KHR_ray_query")
      .extension("SPV_KHR_ray_tracing")
      .extension("SPV_KHR_physical_storage_buffer")
      .memory_model();
  Module without = ray_query_module();
  without.extension("SPV_KHR_ray_query")
      .extension("SPV_KHR_physical_storage_buffer")
      .memory_model();
  CHECK(normalized(with.bytes()) == without.bytes());
}

TEST_CASE("gfx spirv: a module with the RayTracingKHR capability keeps the "
          "extension") {
  Module m = ray_query_module();
  m.capability(CAPABILITY_RAY_TRACING)
      .extension("SPV_KHR_ray_query")
      .extension("SPV_KHR_ray_tracing")
      .memory_model();
  CHECK(normalized(m.bytes()) == m.bytes());
}

TEST_CASE("gfx spirv: a module with nothing to change comes back byte for "
          "byte") {
  Module plain;
  plain.capability(CAPABILITY_SHADER)
      .extension("SPV_KHR_physical_storage_buffer")
      .memory_model();
  CHECK(normalized(plain.bytes()) == plain.bytes());

  Module longer = ray_query_module();
  longer.extension("SPV_KHR_ray_tracing_position_fetch").memory_model();
  CHECK(normalized(longer.bytes()) == longer.bytes());
}

TEST_CASE("gfx spirv: malformed modules are refused") {
  const nx::vector<u8> good = ray_query_module().memory_model().bytes();

  nx::vector<u8> uneven = good;
  uneven.push_back(0);
  auto result = gfx::normalize_spirv({uneven.data(), uneven.size()});
  REQUIRE(!result.has_value());
  CHECK(result.error() == gfx::spirv_error::Truncated);

  nx::vector<u8> header_only(16, 0);
  result = gfx::normalize_spirv({header_only.data(), header_only.size()});
  REQUIRE(!result.has_value());
  CHECK(result.error() == gfx::spirv_error::Truncated);

  nx::vector<u8> swapped = good;
  std::swap(swapped[0], swapped[3]);
  std::swap(swapped[1], swapped[2]);
  result = gfx::normalize_spirv({swapped.data(), swapped.size()});
  REQUIRE(!result.has_value());
  CHECK(result.error() == gfx::spirv_error::BadMagic);

  Module zero = ray_query_module();
  zero.words.push_back(0u);
  nx::vector<u8> zero_bytes = zero.bytes();
  result = gfx::normalize_spirv({zero_bytes.data(), zero_bytes.size()});
  REQUIRE(!result.has_value());
  CHECK(result.error() == gfx::spirv_error::BadInstruction);

  Module overrun = ray_query_module();
  overrun.words.push_back(9u << 16 | OP_CAPABILITY);
  overrun.words.push_back(CAPABILITY_SHADER);
  nx::vector<u8> overrun_bytes = overrun.bytes();
  result = gfx::normalize_spirv({overrun_bytes.data(), overrun_bytes.size()});
  REQUIRE(!result.has_value());
  CHECK(result.error() == gfx::spirv_error::BadInstruction);
}
