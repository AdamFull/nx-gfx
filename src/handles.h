#pragma once

#include "nx/gfx/types.h"

namespace nxe::rhi {

// The names before nx::gfx, for projects made before it (through
// app/engine.h); drop them once those have moved.
using TextureHandle = nx::gfx::texture_handle;
using BufferHandle = nx::gfx::buffer_handle;
using HeapHandle = nx::gfx::heap_handle;
using SamplerHandle = nx::gfx::sampler_handle;
using PipelineHandle = nx::gfx::pipeline_handle;
using ShaderHandle = nx::gfx::shader_handle;
using QueryPoolHandle = nx::gfx::query_pool_handle;
using SwapchainHandle = nx::gfx::swapchain_handle;

using nx::gfx::bindless_index;

} // namespace nxe::rhi
