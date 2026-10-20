#pragma once

#include "nx/gfx/types.h"

#include <cstddef>
#include <span>

namespace nx::gfx {

/// Whether barrier() advances the RHI's tracked texture states. Skip it when
/// the caller owns the timeline and may be recording on several threads.
enum class state_tracking : u8 { Update, Skip };

struct render_attachment {
  texture_handle texture;
  u32 mip_level = 0;
  u32 base_layer = 0;
  load_op load = load_op::Clear;
  store_op store = store_op::Store;
  clear_value clear = {};
  texture_handle resolve_target;
  /// A depth attachment the pass only tests, in its read-only state
  /// (resource_state::DepthStencilRead) and storing nothing; the pass's
  /// pipelines must not write depth.
  bool read_only = false;
};

struct render_pass_desc {
  nx::string_view name = "pass";
  render_attachment color[MAX_COLOR_ATTACHMENTS] = {};
  u32 color_count = 0;
  render_attachment depth_stencil = {};
  bool has_depth_stencil = false;
  rect2d render_area = {};
};

struct draw_indirect_command {
  u32 vertex_count = 0;
  u32 instance_count = 0;
  u32 first_vertex = 0;
  u32 first_instance = 0;
};

struct draw_mesh_tasks_indirect_command {
  u32 groups_x = 0;
  u32 groups_y = 0;
  u32 groups_z = 0;
};

struct draw_indexed_indirect_command {
  u32 index_count = 0;
  u32 instance_count = 0;
  u32 first_index = 0;
  i32 vertex_offset = 0;
  u32 first_instance = 0;
};

struct dispatch_indirect_command {
  u32 groups_x = 0;
  u32 groups_y = 0;
  u32 groups_z = 0;
};

/// Barrier describing a texture's state transition. The render graph is the
/// intended producer of these; hand-written barriers are expert-mode. From a
/// written state to itself, it orders the accesses on either side of it.
struct texture_barrier {
  texture_handle texture;
  resource_state from = resource_state::Undefined;
  resource_state to = resource_state::Undefined;
  bool discard = false;
  u32 base_mip = 0;
  u32 mip_count = ~0u;
  u32 base_layer = 0;
  u32 layer_count = ~0u;
};

struct buffer_barrier {
  buffer_handle buffer;
  resource_state from = resource_state::Undefined;
  resource_state to = resource_state::Undefined;
  u64 offset = 0;
  u64 size = ~0ull;
};

void begin_render_pass(command_list &cmd, const render_pass_desc &desc);
void end_render_pass(command_list &cmd);
void begin_compute_pass(command_list &cmd, nx::string_view name = "compute");
void end_compute_pass(command_list &cmd);

void set_viewport(command_list &cmd, const viewport &vp);
void set_scissor(command_list &cmd, const rect2d &rect);
/// Overrides the bound graphics pipeline's own until the next bind.
void set_cull_mode(command_list &cmd, cull_mode cull);
void set_front_face(command_list &cmd, front_face face);

/// Clears @p count slots. Every slot written must be reset first, outside a
/// render pass.
void reset_queries(command_list &cmd, query_pool_handle pool, u32 first,
                   u32 count);
void write_timestamp(command_list &cmd, query_pool_handle pool, u32 index);

void bind_pipeline(command_list &cmd, pipeline_handle pipeline);
/// Sets the whole 128-byte push block, or part of it. Offset and size in
/// bytes.
void push_constants(command_list &cmd, const void *data, u32 size,
                    u32 offset = 0);
void push_constants(command_list &cmd, std::span<const std::byte> bytes,
                    u32 offset = 0);
/// The only buffer binding: everything else a shader reads is reached
/// bindlessly. Index fetch is fixed-function on every backend.
void bind_index_buffer(command_list &cmd, buffer_handle buffer, index_type type,
                       u64 offset = 0);

void draw(command_list &cmd, u32 vertex_count, u32 instance_count = 1,
          u32 first_vertex = 0, u32 first_instance = 0);
void draw_indexed(command_list &cmd, u32 index_count, u32 instance_count = 1,
                  u32 first_index = 0, i32 vertex_offset = 0,
                  u32 first_instance = 0);
void dispatch(command_list &cmd, u32 groups_x, u32 groups_y = 1,
              u32 groups_z = 1);
void draw_indirect(command_list &cmd, buffer_handle buffer, u64 offset = 0,
                   u32 draw_count = 1, u32 stride = 0);
void draw_indexed_indirect(command_list &cmd, buffer_handle buffer,
                           u64 offset = 0, u32 draw_count = 1, u32 stride = 0);
/// draw_mesh_tasks_indirect_command records, for a mesh pipeline; nothing
/// where the device has no mesh shaders.
void draw_mesh_tasks_indirect(command_list &cmd, buffer_handle buffer,
                              u64 offset = 0, u32 draw_count = 1,
                              u32 stride = 0);
void dispatch_indirect(command_list &cmd, buffer_handle buffer, u64 offset = 0);

void barrier(command_list &cmd, const texture_barrier *textures,
             u32 texture_count, const buffer_barrier *buffers = nullptr,
             u32 buffer_count = 0,
             state_tracking tracking = state_tracking::Update);
void barrier(command_list &cmd, const texture_barrier &texture);

/// Copies one tightly packed colour mip into a host-readable buffer, outside
/// a pass, from CopySrc. @p row_pitch may pad the rows.
[[nodiscard]] bool copy_texture_to_buffer(command_list &cmd,
                                          texture_handle texture,
                                          buffer_handle buffer, u64 row_pitch,
                                          u32 mip = 0, u32 layer = 0);
/// Copies @p size bytes outside a pass and makes them visible to the host.
/// The source is CopySrc and the destination CopyDst, in usage and state.
[[nodiscard]] bool copy_buffer(command_list &cmd, buffer_handle source,
                               u64 source_offset, buffer_handle destination,
                               u64 destination_offset, u64 size);

void push_debug_group(command_list &cmd, nx::string_view name, f32 r = 1.f,
                      f32 g = 1.f, f32 b = 1.f);
void pop_debug_group(command_list &cmd);

} // namespace nx::gfx
