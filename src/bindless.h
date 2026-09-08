#pragma once

#include "core/foundation/core/foundation.h"
#include "core/rendering/rhi/shaders/bindless_layout.h"

namespace nxe::rhi {

inline constexpr u32 BINDLESS_SET = 0;

inline constexpr u32 BINDLESS_BINDING_SAMPLERS = 0;
inline constexpr u32 BINDLESS_BINDING_TEXTURES = 1;
inline constexpr u32 BINDLESS_BINDING_RW_TEXTURES = 2;

/// Upper bounds on each array; the actual size is min(this, device limit).
/// TEXTURES must exceed the peak count of simultaneously live textures because
/// a texture's SRV slot is its handle index.
inline constexpr u32 MAX_BINDLESS_SAMPLERS = NX_BINDLESS_MAX_SAMPLERS;
inline constexpr u32 MAX_BINDLESS_TEXTURES = NX_BINDLESS_MAX_TEXTURES;
inline constexpr u32 MAX_BINDLESS_RW_TEXTURES = NX_BINDLESS_MAX_RW_TEXTURES;

inline constexpr u32 PUSH_CONSTANT_SIZE = 128;

inline constexpr u32 BINDLESS_INVALID_INDEX = ~0u;

}
