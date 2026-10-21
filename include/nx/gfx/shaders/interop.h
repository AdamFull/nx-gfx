#pragma once

// What every header shared by C++ and Slang starts from. Such a header is
// written once for both languages: structs whose fields use the types here
// (NxPtr<T>, NxTexture2D<T> and the other texture handles, float4 and the
// rest), constants made with NX_CONST and functions marked NX_SHARED_FN
// (shared.h), with no #ifdef and no visibility keywords.
//
// Slang reads it inside a module with neither a `module` declaration nor any
// visibility keyword, where every declaration is public, fields included; a
// module with an API of its own re-exports that one with `__exported import`.
// Headers read into one module include each other; one read into another
// module includes them for C++ only and its shader imports the module that
// has them, since a second copy of a struct makes every use of it ambiguous.
//
// A shared struct has the same offsets in C++, in Slang's natural layout
// (what NxPtr loads through) and in std430 (push constants): fields are 4, 8
// or 16 byte types placed where all three agree.
//
// C++'s float4 and the rest are plain structs here. A consumer that wants its
// own math types on the fields names a header supplying them in
// NX_GFX_INTEROP_MATH_TYPES, which the build defines for every translation
// unit alike.

#include "nx/gfx/shaders/shared.h"

#ifdef __cplusplus
#include "nx/gfx/shaders/ptr.hpp"
#include "nx/gfx/shaders/texture.hpp"

#ifdef NX_GFX_INTEROP_MATH_TYPES
#include NX_GFX_INTEROP_MATH_TYPES
#else
struct float2 {
  float x, y;
};
struct float4 {
  float x, y, z, w;
};
struct int2 {
  std::int32_t x, y;
};
struct int4 {
  std::int32_t x, y, z, w;
};
struct uint2 {
  std::uint32_t x, y;
};
struct uint4 {
  std::uint32_t x, y, z, w;
};
struct float4x4 {
  float4 columns[4];
};
#endif

static_assert(sizeof(float2) == 8 && sizeof(int2) == 8 && sizeof(uint2) == 8 &&
                  sizeof(float4) == 16 && sizeof(int4) == 16 &&
                  sizeof(uint4) == 16 && sizeof(float4x4) == 64,
              "the math types keep Slang's sizes");
#endif
