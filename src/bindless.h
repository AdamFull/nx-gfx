#pragma once

#include "core/foundation/core/foundation.h"
#include "rendering/rhi/shaders/core_layout.h"
#include "rendering/rhi/shaders/nx_texture.hpp"

namespace nxe::rhi {

inline constexpr u32 BINDLESS_SET = NX_BINDLESS_SET;

inline constexpr u32 BINDLESS_BINDING_SAMPLERS = NX_BINDLESS_BINDING_SAMPLERS;
inline constexpr u32 BINDLESS_BINDING_TEXTURES = NX_BINDLESS_BINDING_TEXTURES;
inline constexpr u32 BINDLESS_BINDING_RW_TEXTURES =
    NX_BINDLESS_BINDING_RW_TEXTURES;
inline constexpr u32 BINDLESS_BINDING_GLOBALS = NX_BINDLESS_BINDING_GLOBALS;

inline constexpr u32 NULL_PAGE_BYTES = NX_NULL_PAGE_BYTES;

/// Upper bounds on each array; the actual size is min(this, device limit).
/// TEXTURES must exceed the peak count of simultaneously live textures because
/// a texture's SRV slot is its handle index.
inline constexpr u32 MAX_BINDLESS_SAMPLERS = NX_BINDLESS_MAX_SAMPLERS;
inline constexpr u32 MAX_BINDLESS_TEXTURES = NX_BINDLESS_MAX_TEXTURES;
inline constexpr u32 MAX_BINDLESS_RW_TEXTURES = NX_BINDLESS_MAX_RW_TEXTURES;

inline constexpr u32 PUSH_CONSTANT_SIZE = 128;

inline constexpr u32 BINDLESS_INVALID_INDEX = ~0u;

} // namespace nxe::rhi
