#pragma once

#include "nx/gfx/commands.h"
#include "nx/gfx/types.h"

namespace nxe::rhi {

// The names before nx::gfx. Only the backend's own sources and projects made
// before nx::gfx (through app/engine.h) use them.
using Format = nx::gfx::format;
using StateTracking = nx::gfx::state_tracking;
using ResourceState = nx::gfx::resource_state;
using TextureUsage = nx::gfx::texture_usage;
using TextureUsageFlags = nx::gfx::texture_usage_flags;
using BufferUsage = nx::gfx::buffer_usage;
using BufferUsageFlags = nx::gfx::buffer_usage_flags;
using SubgroupOp = nx::gfx::subgroup_op;
using SubgroupOpFlags = nx::gfx::subgroup_op_flags;
using MemoryUsage = nx::gfx::memory_usage;
using TextureDimension = nx::gfx::texture_dimension;
using LoadOp = nx::gfx::load_op;
using StoreOp = nx::gfx::store_op;
using FilterMode = nx::gfx::filter_mode;
using AddressMode = nx::gfx::address_mode;
using CompareOp = nx::gfx::compare_op;
using ShaderStage = nx::gfx::shader_stage;
using PrimitiveTopology = nx::gfx::primitive_topology;
using IndexType = nx::gfx::index_type;
using CullMode = nx::gfx::cull_mode;
using FrontFace = nx::gfx::front_face;
using BlendMode = nx::gfx::blend_mode;
using QueueType = nx::gfx::queue_type;
using PresentMode = nx::gfx::present_mode;
using SurfaceAlpha = nx::gfx::surface_alpha;
using AcquireResult = nx::gfx::acquire_result;

using nx::gfx::buffer_usage_from_string;
using nx::gfx::format_block_extent;
using nx::gfx::format_channels;
using nx::gfx::format_from_string;
using nx::gfx::format_is_compressed;
using nx::gfx::format_is_depth;
using nx::gfx::format_is_float;
using nx::gfx::format_is_srgb;
using nx::gfx::format_is_stencil;
using nx::gfx::format_row_pitch;
using nx::gfx::format_subresource_size;
using nx::gfx::format_texel_size;
using nx::gfx::is_write_state;
using nx::gfx::texture_usage_from_string;
using nx::gfx::to_string;

} // namespace nxe::rhi
