#pragma once

#include "nx/gfx/types.h"

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

} // namespace nx::gfx
