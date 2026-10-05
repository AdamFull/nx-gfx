#pragma once

// What every header shared by C++ and Slang starts from. Such a header is
// written once for both languages: structs whose fields use the types here
// (NxPtr<T>, NxTexture2D<T> and the other texture handles, float4 and the
// rest), constants made with NX_CONST and functions marked NX_SHARED_FN
// (nx_shared.h), with no #ifdef and no visibility keywords.
//
// Slang reads it inside a module with neither a `module` declaration nor any
// visibility keyword, where every declaration is public, fields included; a
// module with an API of its own re-exports that one with `__exported import`.
//
// A shared struct has the same offsets in C++, in Slang's natural layout
// (what NxPtr loads through) and in std430 (push constants): fields are 4, 8
// or 16 byte types placed where all three agree, which the interop layout
// test checks against the shaders' reflection.

#include "rendering/rhi/shaders/nx_shared.h"

#ifdef __cplusplus
#include "core/foundation/core/foundation.h"
#include "rendering/rhi/shaders/nx_ptr.hpp"
#include "rendering/rhi/shaders/nx_texture.hpp"

#include <glm/glm.hpp>

using float2 = glm::vec2;
using float4 = glm::vec4;
using float4x4 = glm::mat4;
using int4 = glm::ivec4;
using uint2 = glm::uvec2;
using uint4 = glm::uvec4;
#endif
