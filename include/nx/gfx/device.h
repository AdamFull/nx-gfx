#pragma once

#include "nx/gfx/types.h"

namespace nx::gfx {

struct device_caps {
  nx::string_view device_name;
  u32 api_version = 0;
  u32 driver_version = 0;
  u32 vendor_id = 0;
  bool discrete = false;

  bool descriptor_indexing = false;
  bool buffer_device_address = false;

  /// True on every backend in practice, for different reasons: Vulkan
  /// requires all three at physical-device selection (device creation fails
  /// otherwise, so this is a fact, not a probe), and Metal has never had a
  /// render-pass-object or pipeline-barrier API to opt into in the first
  /// place - its baseline behaviour already is what these name. Present for
  /// diagnostics and ModuleCapability completeness; no code branches on
  /// these being false because on a successfully-initialized device they
  /// never are.
  bool dynamic_rendering = false;
  bool synchronization2 = false;
  bool timeline_semaphores = false;

  bool async_compute = false;
  bool dedicated_transfer = false;
  bool draw_indirect_count = false;
  bool memoryless_attachments = false;
  bool timestamps = false;
  /// One frame's commands can be recorded on several threads at once, through
  /// split_frame and begin_recording.
  bool parallel_recording = false;

  /// The sample counts a multisampled target may have in every attachment
  /// format, colour, integer and depth alike: bit N set where N samples
  /// work. A device may skip one between its largest and 1 (Mali has no 2).
  u32 msaa_sample_counts = 1;

  /// What 3D may use where the device has it, each enabled at device
  /// creation when present. Metal reports none of them until its 3D track.
  bool mesh_shaders = false;
  bool buffer_int64_atomics = false;
  bool image_int64_atomics = false;
  /// Half-precision arithmetic in shaders (SPIR-V Float16).
  bool shader_float16 = false;
  bool fragment_barycentrics = false;
  /// graphics_pipeline_desc::sample_shading.
  bool sample_shading = false;
  /// A fragment shader may store to and atomically update buffers.
  bool fragment_stores = false;
  bool acceleration_structures = false;
  /// Implies acceleration_structures.
  bool ray_queries = false;
  /// In compute shaders. subgroup_size is what a dispatch gets unless its
  /// pipeline asks for another in [min_subgroup_size, max_subgroup_size],
  /// which needs subgroup_size_control.
  subgroup_op_flags subgroup_ops;
  u32 subgroup_size = 0;
  u32 min_subgroup_size = 0;
  u32 max_subgroup_size = 0;
  bool subgroup_size_control = false;

  u32 max_texture_2d = 0;
  u64 max_buffer_size = 0;
  extent3d max_compute_work_groups{0, 0, 0};
  u32 max_bindless_sampled_images = 0;
  u32 max_bindless_storage_images = 0;
  u32 max_bindless_samplers = 0;
  u32 max_push_constant_size = 0;
  u64 device_local_memory = 0;
  f32 timestamp_period_ns = 0.f;
};

/// The most samples a target may have in @p counts (device_caps::
/// msaa_sample_counts); 1 at least.
[[nodiscard]] constexpr u32 max_samples(const u32 counts) noexcept {
  for (u32 samples = 64; samples > 1; samples >>= 1)
    if ((counts & samples) != 0)
      return samples;
  return 1;
}

/// The most samples in @p counts that are at most @p wanted; 1 at least.
[[nodiscard]] constexpr u32 samples_at_most(const u32 counts,
                                            const u32 wanted) noexcept {
  for (u32 samples = 64; samples > 1; samples >>= 1)
    if (samples <= wanted && (counts & samples) != 0)
      return samples;
  return 1;
}

/// Classifies device startup without leaking a backend-specific error model to
/// the application. Unsupported means that the available system graphics
/// stack cannot satisfy the RHI's required capability floor; Failed means a
/// compatible stack was found but initialization did not complete.
enum class device_init_failure : u8 {
  None,
  Unsupported,
  Failed,
};

struct device_init_report {
  device_init_failure failure = device_init_failure::None;
  u32 vendor_id = 0;
  u32 device_id = 0;
};

} // namespace nx::gfx
