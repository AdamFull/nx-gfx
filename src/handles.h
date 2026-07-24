#pragma once

/**
 * @file handles.h
 * @brief Typed, generation-checked resource handles (namespace nxe::rhi).
 *
 * Resources are named by handle, never by pointer: creation is free-threaded
 * and destruction is deferred to the end of the frame, so a raw pointer would
 * be a dangling-reference generator. Handles carry a generation, so using one
 * after its resource is destroyed is detectable rather than undefined.
 *
 * The handle's index doubles as its bindless descriptor slot - handle.index()
 * is exactly the value a shader puts in its push block. That is why no separate
 * "bindless slot" concept appears anywhere in this API.
 *
 * nx::basic_handle's Tag parameter keeps the types distinct, so a BufferHandle
 * will not silently pass where a TextureHandle is wanted.
 */

#include "foundation/core/handle.h"

namespace nxe::rhi {

// Textures and buffers use 32 index bits: a streaming atlas set churns slots
// fast enough that handle32's 8 version bits would wrap in a long session.
using TextureHandle = nx::basic_handle<u64, 32, struct TextureTag>;
using BufferHandle = nx::basic_handle<u64, 32, struct BufferTag>;

// These are created once and live for the process, so 24 index / 8 version bits
// are ample.
using SamplerHandle = nx::basic_handle<u32, 24, struct SamplerTag>;
using PipelineHandle = nx::basic_handle<u32, 24, struct PipelineTag>;
using ShaderHandle = nx::basic_handle<u32, 24, struct ShaderTag>;

/// Bindless descriptor slot for a resource, for putting in a push block.
/// Only meaningful for classes that live in a bindless heap (sampled images,
/// storage images, samplers).
template <typename Repr, usize IndexBits, typename Tag>
[[nodiscard]] constexpr u32
bindless_index(const nx::basic_handle<Repr, IndexBits, Tag> handle) noexcept {
  return nx::cast<u32>(handle.index());
}

} // namespace nxe::rhi
