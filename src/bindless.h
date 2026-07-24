#pragma once

/**
 * @file bindless.h
 * @brief The bindless binding contract shared by C++ and shaders.
 *
 * One descriptor set, bound once per frame and never rebound. Buffers are not
 * in it - they are reached by GPU device address carried in the push block.
 * Shaders must mirror these set/binding numbers. See docs/design.md section 5.
 */

#include "foundation/core/foundation.h"

namespace nxe::rhi {

inline constexpr u32 BINDLESS_SET = 0;

inline constexpr u32 BINDLESS_BINDING_SAMPLERS = 0;    ///< sampler[]
inline constexpr u32 BINDLESS_BINDING_TEXTURES = 1;    ///< sampled image[]
inline constexpr u32 BINDLESS_BINDING_RW_TEXTURES = 2; ///< storage image[]

/// Upper bounds on each array; the actual size is min(this, device limit).
/// TEXTURES must exceed the peak count of simultaneously live textures because
/// a texture's SRV slot is its handle index.
inline constexpr u32 MAX_BINDLESS_SAMPLERS = 1024;
inline constexpr u32 MAX_BINDLESS_TEXTURES = 65536;
inline constexpr u32 MAX_BINDLESS_RW_TEXTURES = 8192;

/// Guaranteed on both Vulkan and Metal.
inline constexpr u32 PUSH_CONSTANT_SIZE = 128;

/// Value meaning "no resource" in a push block index field.
inline constexpr u32 BINDLESS_INVALID_INDEX = ~0u;

} // namespace nxe::rhi
