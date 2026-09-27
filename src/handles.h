#pragma once

#include "core/foundation/core/handle.h"

namespace nxe::rhi {

// Textures and buffers use 32 index bits: a streaming atlas set churns slots
// fast enough that handle32's 8 version bits would wrap in a long session.
using TextureHandle = nx::basic_handle<u64, 32, struct TextureTag>;
using BufferHandle = nx::basic_handle<u64, 32, struct BufferTag>;

// These are created once and live for the process, so 24 index / 8 version bits
// are ample.

using HeapHandle = nx::basic_handle<u32, 24, struct HeapTag>;
using SamplerHandle = nx::basic_handle<u32, 24, struct SamplerTag>;
using PipelineHandle = nx::basic_handle<u32, 24, struct PipelineTag>;
using ShaderHandle = nx::basic_handle<u32, 24, struct ShaderTag>;
using QueryPoolHandle = nx::basic_handle<u32, 24, struct QueryPoolTag>;
using SwapchainHandle = nx::basic_handle<u32, 24, struct SwapchainTag>;

template <typename Repr, usize IndexBits, typename Tag>
[[nodiscard]] constexpr u32
bindless_index(const nx::basic_handle<Repr, IndexBits, Tag> handle) noexcept {
  return nx::cast<u32>(handle.index());
}

}
