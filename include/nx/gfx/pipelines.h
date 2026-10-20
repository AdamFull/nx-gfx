#pragma once

#include "nx/gfx/pipeline_job.h"
#include "nx/gfx/reflection.h"
#include "nx/gfx/types.h"

namespace nx::gfx {

#if defined(NX_RHI_METAL)
inline constexpr nx::string_view SHADER_BINARY_EXT = ".metallib";
#else
inline constexpr nx::string_view SHADER_BINARY_EXT = ".spv";
#endif

/// A compiled shader blob (SPIR-V on Vulkan, metallib on Metal). One blob may
/// hold several entry points; a pipeline names the one it wants.
struct shader_desc {
  nx::string_view name = "shader";
  const void *code = nullptr;
  usize code_size = 0;
  /// Its interface, already parsed (reflect_from_json, a cooked container);
  /// copied. Null for a shader with none.
  const shader_reflection *reflection = nullptr;
};

struct shader_entry {
  shader_handle shader;
  nx::string_view entry_point = "main";
};

struct blend_state {
  bool enabled = false;
  blend_mode mode = blend_mode::Opaque;
  /// Per-channel write mask, bit 0..3 = R,G,B,A.
  u8 write_mask = 0xF;
};

struct graphics_pipeline_desc {
  nx::string_view name = "pipeline";
  /// Exactly one of vertex and mesh has a shader; a description without
  /// shaders yet (GpuPipelineRuntime's) names a mesh pipeline by an empty
  /// vertex entry. A mesh pipeline has no vertex input or topology of its
  /// own, needs device_caps::mesh_shaders and draws only through
  /// draw_mesh_tasks_indirect.
  shader_entry vertex;
  shader_entry mesh;
  shader_entry fragment;

  /// Attachment formats, for dynamic rendering. Must match the render pass the
  /// pipeline is used in.
  format color_formats[MAX_COLOR_ATTACHMENTS] = {};
  u32 color_count = 0;
  format depth_format = format::Unknown;
  // No "= {}" here (unlike color_formats above): GCC has a real bug binding
  // a temporary aggregate-initialized from a braced-init-list directly to a
  // reference function parameter when one of its members is an array of
  // class type with its own default member initializer - every
  // compile_pipeline(device, {...}) call site failed with "no known
  // conversion" on Linux, even the ones that never touch .blend. Safe to
  // drop: blend_state's own fields all have default member initializers, so
  // omitting .blend from an aggregate-init list already value-initializes
  // each element identically via [dcl.init.aggr]'s "empty initializer list"
  // rule, and plain `graphics_pipeline_desc d;` default-init still runs
  // blend_state's implicit default constructor per element either way.
  blend_state blend[MAX_COLOR_ATTACHMENTS];

  primitive_topology topology = primitive_topology::TriangleList;
  cull_mode cull = cull_mode::None;
  gfx::front_face front_face = gfx::front_face::CounterClockwise;

  bool depth_test = false;
  bool depth_write = false;
  compare_op depth_compare = compare_op::LessOrEqual;

  u32 sample_count = 1;
  /// Runs the fragment shader once for each covered sample, at that sample,
  /// where the device can (device_caps::sample_shading); once a pixel
  /// otherwise.
  bool sample_shading = false;
};

struct compute_pipeline_desc {
  nx::string_view name = "pipeline";
  shader_entry compute;
  u32 group_size[3] = {0, 0, 0};
};

[[nodiscard]] shader_handle create_shader(device &dev, const shader_desc &desc);
void destroy_shader(device &dev, shader_handle shader) noexcept;
[[nodiscard]] const shader_reflection &
reflection(const device &dev, shader_handle shader) noexcept;

/// On the device's owner thread; compile the job anywhere, then publish it
/// on the owner thread.
[[nodiscard]] pipeline_job prepare_pipeline(device &dev,
                                            const graphics_pipeline_desc &desc);
[[nodiscard]] pipeline_job prepare_pipeline(device &dev,
                                            const compute_pipeline_desc &desc);
[[nodiscard]] pipeline_handle publish_pipeline(device &dev, pipeline_job &&job);
void destroy_pipeline(device &dev, pipeline_handle pipeline) noexcept;
[[nodiscard]] nx::workload_statistics
pipeline_compile_statistics(const device &dev) noexcept;

} // namespace nx::gfx
