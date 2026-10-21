#pragma once

// A struct with a field of each type nx/gfx/shaders/interop.h gives shared
// structs, which test_gfx_interop_layout.cpp lays out on both sides.

#include "nx/gfx/shaders/interop.h"

struct GfxInteropProbe {
  NxPtr<uint> words;
  NxTexture2D<float4> texture;
  NxRWTexture2D<float4> storage;
  float2 vec2;
  uint2 uvec2;
  int2 ivec2;
  uint pad0;
  uint pad1;
  float4 vec4;
  uint4 uvec4;
  int4 ivec4;
  float4x4 matrix;
};
NX_SHARED_SIZE(GfxInteropProbe, 160);
