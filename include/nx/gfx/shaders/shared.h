#pragma once

// The macros a header shared by C++ and Slang (see interop.h) writes its
// constants, functions and size checks with.

#ifdef __cplusplus
#include <bit>
#include <cmath>
#include <cstdint>

using uint = std::uint32_t;
using uint64_t = std::uint64_t;

#define NX_CONST(type, name, value) inline constexpr type name = value
#define NX_SHARED_FN [[nodiscard]] inline
#define NX_SHARED_SIZE(type, bytes)                                            \
  static_assert(sizeof(type) == (bytes), #type " is shared with Slang")
#define NX_SQRT(value) std::sqrt(value)
#define NX_ASUINT(value) std::bit_cast<uint>(value)
#define NX_ASFLOAT(value) std::bit_cast<float>(value)
#else
#define NX_CONST(type, name, value) static const type name = value
#define NX_SHARED_FN
#define NX_SHARED_SIZE(type, bytes)
#define NX_SQRT(value) sqrt(value)
#define NX_ASUINT(value) asuint(value)
#define NX_ASFLOAT(value) asfloat(value)
#endif
