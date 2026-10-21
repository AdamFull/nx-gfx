#pragma once

#include "nx/gfx/commands.h"
#include "nx/gfx/device.h"
#include "nx/gfx/frame.h"
#include "nx/gfx/pipelines.h"
#include "nx/gfx/resources.h"
#include "nx/gfx/types.h"
#include "rendering/rhi/handles.h"
#include "rendering/rhi/types.h"

namespace nxe::rhi {

// The names before nx::gfx. Only the backend's own sources and projects made
// before nx::gfx (through app/engine.h) use them.
using Extent2D = nx::gfx::extent2d;
using Extent3D = nx::gfx::extent3d;
using Offset2D = nx::gfx::offset2d;
using Rect2D = nx::gfx::rect2d;
using Viewport = nx::gfx::viewport;
using ClearValue = nx::gfx::clear_value;
using TextureDesc = nx::gfx::texture_desc;
using MemoryRequirements = nx::gfx::memory_requirements;
using HeapDesc = nx::gfx::heap_desc;
using BufferDesc = nx::gfx::buffer_desc;
using SamplerDesc = nx::gfx::sampler_desc;
using RenderAttachment = nx::gfx::render_attachment;
using RenderPassDesc = nx::gfx::render_pass_desc;
using SwapchainDesc = nx::gfx::swapchain_desc;
using ShaderReflection = nx::gfx::shader_reflection;
using ShaderDesc = nx::gfx::shader_desc;
using ShaderEntry = nx::gfx::shader_entry;
using BlendState = nx::gfx::blend_state;
using GraphicsPipelineDesc = nx::gfx::graphics_pipeline_desc;
using ComputePipelineDesc = nx::gfx::compute_pipeline_desc;
using DrawIndirectCommand = nx::gfx::draw_indirect_command;
using DrawMeshTasksIndirectCommand = nx::gfx::draw_mesh_tasks_indirect_command;
using DrawIndexedIndirectCommand = nx::gfx::draw_indexed_indirect_command;
using DispatchIndirectCommand = nx::gfx::dispatch_indirect_command;
using TextureBarrier = nx::gfx::texture_barrier;
using BufferBarrier = nx::gfx::buffer_barrier;
using RecordingSplit = nx::gfx::recording_split;
using DeviceCaps = nx::gfx::device_caps;
using DeviceInitFailure = nx::gfx::device_init_failure;
using DeviceInitReport = nx::gfx::device_init_report;
using DeviceDesc = nx::gfx::device_desc;

using nx::gfx::clear_color;
using nx::gfx::clear_depth;
using nx::gfx::DEFAULT_SAMPLER_DESC;
using nx::gfx::make_version;
using nx::gfx::MAX_COLOR_ATTACHMENTS;
using nx::gfx::max_samples;
using nx::gfx::samples_at_most;
using nx::gfx::SHADER_BINARY_EXT;

} // namespace nxe::rhi
