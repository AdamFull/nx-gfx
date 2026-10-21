#pragma once

// NxPtr<T>: a device address shaders read and write T through, the C++ half
// of the pair whose Slang half is nxg_ptr.slang. A shared struct declares a
// field NxPtr<T> once for both; C++ fills it with a buffer's address:
// `push.instances = {gfx::buffer_address(dev, buffer)}`.

#include <cstdint>

template <typename T> struct NxPtr {
  std::uint64_t value;

  [[nodiscard]] constexpr bool is_valid() const noexcept { return value != 0; }
};

static_assert(sizeof(NxPtr<std::uint32_t>) == 8 &&
                  alignof(NxPtr<std::uint32_t>) == 8,
              "NxPtr is one 64-bit address on both sides");
