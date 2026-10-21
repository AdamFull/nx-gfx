#pragma once

#include "core/foundation/core/result.h"
#include "nx/gfx/crash.h"
#include "nx/gfx/native.h"
#include "nx/gfx/types.h"

#include <span>

namespace nx::gfx {

/// Vulkan feature routes a device takes where it has the feature. Disabling
/// one makes the device take the fallback, so the fallback is tested on any
/// machine; Shipping builds ignore it.
enum class route : u8 {
  None = 0,
  /// Dynamic rendering; the fallback is render passes over imageless
  /// framebuffers.
  DynamicRendering = 1 << 0,
};
NX_ENUM_FLAGS_DEFINE(route, route_flags)

struct device_desc {
  nx::string_view application_name;
  u32 application_version = 0;
  /// What drivers key application profiles on; make_version() packs it.
  nx::string_view engine_name;
  u32 engine_version = 0;
#if NX_DEBUG
  bool enable_validation = true;
  bool enable_sync_validation = true;
#else
  bool enable_validation = false;
  bool enable_sync_validation = false;
#endif
  bool break_on_validation_error = false;
  gfx::gpu_crash_mode gpu_crash_mode = gfx::gpu_crash_mode::Basic;
  nx::string_view pipeline_cache_path;
  native_driver_view native_driver;
  u32 frames_in_flight = 2;
  /// Asked what to enable before the device is made, and only then.
  std::span<native_device_client *const> native_clients{};
  /// Outside Shipping, NX_GFX_DISABLE_ROUTES adds to it ("dynamic_rendering").
  route_flags disable_routes = route::None;
};

struct device_caps {
  nx::string_view device_name;
  u32 api_version = 0;
  u32 driver_version = 0;
  u32 vendor_id = 0;
  bool discrete = false;

  bool descriptor_indexing = false;
  bool buffer_device_address = false;

  /// Whether passes record through dynamic rendering. A Vulkan device without
  /// it (or with route::DynamicRendering disabled) records through render
  /// passes instead; Metal always has it.
  bool dynamic_rendering = false;
  /// True on every device that initialized: Vulkan requires both, and Metal's
  /// baseline already is what they name.
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

struct device_init_error {
  device_init_report report;
  /// What is missing, or what failed.
  nx::string message;
};

/// Makes a device on the calling thread, which owns it from then on.
[[nodiscard]] nx::result<device *, device_init_error>
create_device(const device_desc &desc);
/// Waits for the GPU, then frees the device and whatever it still holds.
/// Nothing may be recording.
void destroy_device(device *dev) noexcept;

[[nodiscard]] const device_caps &caps(const device &dev) noexcept;
[[nodiscard]] gpu_device_status device_status(const device &dev) noexcept;
[[nodiscard]] u32 frames_in_flight(const device &dev) noexcept;
[[nodiscard]] bool supports(const device &dev, format fmt,
                            texture_usage_flags usage);

/// Pushes queued uploads to the GPU now, for a device that runs no frames.
void flush_uploads(device &dev) noexcept;
/// Waits for the GPU, then frees every retired resource. Never while another
/// thread records or submits a frame: that frame may hold what this frees.
void wait_idle(device &dev) noexcept;
/// Waits for the frames already submitted; frees nothing.
void wait_submitted(device &dev) noexcept;
/// Writes the driver's pipeline cache to device_desc::pipeline_cache_path,
/// from any thread, while compiles continue.
bool save_pipeline_cache(device &dev) noexcept;

/// A producer that runs ahead of rendering holds the serial of work that may
/// still name resources; their destruction waits until it is released.
void hold_retirements(device &dev, u64 serial) noexcept;
void release_retirements(device &dev, u64 serial) noexcept;

} // namespace nx::gfx
