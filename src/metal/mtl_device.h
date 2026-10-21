#pragma once

#include "core/foundation/strings/utf8_string.h"
#include "core/foundation/threading/thread_pool.h"
#include "core/foundation/threading/workload.h"
#include "nx/gfx/bindless.h"
#include "nx/gfx/gfx.h"
#include "nx/gfx/crash.h"
#include "nx/gfx/types.h"
#include "nx/gfx/native.h"
#include "nx/gfx/pipeline_job.h"
#include "nx/gfx/reflection.h"
#include "nx/gfx/upload.h"

#include <span>

namespace nx::gfx::detail {

class MetalDevice;

struct MetalDeviceState;

/// Asynchronous GPU upload and readback. Uploads are
/// batched: a call records a blit and returns a ticket, but nothing is
/// committed until the batch flushes (once per frame, or on flush()/wait()).
class MetalUploader {
public:
  MetalUploader() = default;

  MetalUploader(const MetalUploader &) = delete;
  MetalUploader &operator=(const MetalUploader &) = delete;

  [[nodiscard]] gfx::upload_ticket
  upload_texture(gfx::texture_handle handle, std::span<const u8> src,
                 std::span<const gfx::image_subresource> layout,
                 gfx::resource_state final_state = gfx::resource_state::ShaderReadOnly);
  [[nodiscard]] gfx::upload_ticket
  upload_texture(gfx::texture_handle handle, const gfx::texture_data &data,
                 gfx::resource_state final_state = gfx::resource_state::ShaderReadOnly);
  [[nodiscard]] gfx::upload_ticket
  upload_buffer(gfx::buffer_handle handle, u64 dst_offset, std::span<const u8> src,
                gfx::resource_state final_state = gfx::resource_state::ShaderReadOnly);
  [[nodiscard]] gfx::upload_ticket
  clear_buffer(gfx::buffer_handle handle,
               gfx::resource_state final_state = gfx::resource_state::ShaderReadOnly);

  // Readbacks (render thread). The source must not be in concurrent GPU use.
  // A readback flushes any open upload batch and submits on its own.

  gfx::readback_result read_buffer(gfx::buffer_handle handle, u64 offset, u64 size);
  gfx::readback_result read_texture(gfx::texture_handle handle, u32 mip = 0, u32 layer = 0);

  [[nodiscard]] bool is_complete(gfx::upload_ticket ticket) const noexcept;
  void wait(gfx::upload_ticket ticket) noexcept;
  void flush() noexcept;
  /// Submits any open batch without waiting for it, matching
  /// VulkanUploader::flush_batch() - a caller that needs completion still
  /// calls wait() on the ticket.
  void flush_batch() noexcept;

  [[nodiscard]] gfx::uploader_stats stats() const noexcept;

private:
  friend class MetalDevice;

  MetalDeviceState *m_state = nullptr;
};

/// Records one frame's work. A context owns at most one encoder at a time:
/// begin_render_pass opens it, end_render_pass closes it.
class MetalCommandContext {
public:
  MetalCommandContext() = default;

  [[nodiscard]] bool valid() const noexcept { return m_state != nullptr; }

  void begin_render_pass(const gfx::render_pass_desc &desc);
  void end_render_pass();

  void begin_compute_pass(nx::string_view name = "compute");
  void end_compute_pass();

  void set_viewport(const gfx::viewport &viewport);
  void set_scissor(const gfx::rect2d &rect);
  /// Override the bound render pipeline's own until the next bind.
  void set_cull_mode(gfx::cull_mode cull);
  void set_front_face(gfx::front_face face);

  void reset_queries(gfx::query_pool_handle, u32, u32) {}
  void write_timestamp(gfx::query_pool_handle, u32) {}

  void bind_pipeline(gfx::pipeline_handle pipeline);
  /// Sets the whole 128-byte push block. Offset/size in bytes.
  void push_constants(const void *data, u32 size, u32 offset = 0);

  void bind_index_buffer(gfx::buffer_handle buffer, gfx::index_type type, u64 offset = 0);

  void draw(u32 vertex_count, u32 instance_count = 1, u32 first_vertex = 0,
            u32 first_instance = 0);
  void draw_indexed(u32 index_count, u32 instance_count = 1,
                    u32 first_index = 0, i32 vertex_offset = 0,
                    u32 first_instance = 0);
  void dispatch(u32 groups_x, u32 groups_y = 1, u32 groups_z = 1);

  void draw_indirect(gfx::buffer_handle buffer, u64 offset = 0, u32 draw_count = 1,
                     u32 stride = 0);
  void draw_indexed_indirect(gfx::buffer_handle buffer, u64 offset = 0,
                             u32 draw_count = 1, u32 stride = 0);
  /// gfx::draw_mesh_tasks_indirect_command records, for a mesh pipeline; nothing where
  /// the device has no mesh shaders.
  void draw_mesh_tasks_indirect(gfx::buffer_handle buffer, u64 offset = 0,
                                u32 draw_count = 1, u32 stride = 0);
  void dispatch_indirect(gfx::buffer_handle buffer, u64 offset = 0);

  void barrier(const gfx::texture_barrier *textures, u32 texture_count,
               const gfx::buffer_barrier *buffers = nullptr, u32 buffer_count = 0,
               gfx::state_tracking tracking = gfx::state_tracking::Update);
  void barrier(const gfx::texture_barrier &texture) { barrier(&texture, 1); }

  /// Copies one tightly packed colour mip into a host-readable buffer. The
  /// source must already be in CopySrc and this must be called outside a
  /// render/compute pass. row_pitch may include padding between rows.
  [[nodiscard]] bool copy_texture_to_buffer(gfx::texture_handle texture,
                                            gfx::buffer_handle buffer, u64 row_pitch,
                                            u32 mip = 0, u32 layer = 0);
  /// Copies @p size bytes of @p source from @p source_offset into
  /// @p destination at @p destination_offset, outside a render/compute pass,
  /// and makes them visible to the host. The source must be CopySrc, the
  /// destination CopyDst, both in usage and state.
  [[nodiscard]] bool copy_buffer(gfx::buffer_handle source, u64 source_offset,
                                 gfx::buffer_handle destination,
                                 u64 destination_offset, u64 size);

  void push_debug_group(nx::string_view name, f32 r = 1.f, f32 g = 1.f,
                        f32 b = 1.f);
  void pop_debug_group();

private:
  friend class MetalDevice;

  [[nodiscard]] struct MetalContextSlot &ctx_slot() const noexcept;

  MetalDeviceState *m_state = nullptr;
  u32 m_slot = 0;
  bool m_in_render_pass = false;
};

class MetalDevice {
public:
  MetalDevice() = default;
  ~MetalDevice();

  MetalDevice(const MetalDevice &) = delete;
  MetalDevice &operator=(const MetalDevice &) = delete;

  [[nodiscard]] bool init(const gfx::device_desc &desc);
  void shutdown() noexcept;

  [[nodiscard]] nx::string_view init_error() const noexcept {
    return m_init_error;
  }
  [[nodiscard]] const gfx::device_init_report &init_report() const noexcept {
    return m_init_report;
  }

  [[nodiscard]] const gfx::device_caps &caps() const noexcept { return m_caps; }
  [[nodiscard]] gfx::gpu_device_status device_status() const noexcept;

  [[nodiscard]] u32 frames_in_flight() const noexcept;

  [[nodiscard]] MetalUploader &uploader() noexcept { return m_uploader; }

  void flush_uploads() noexcept;

  /// See VulkanDevice::create_swapchain.
  [[nodiscard]] gfx::swapchain_handle create_swapchain(const gfx::swapchain_desc &desc);
  /// A swapchain with no layer behind it: a texture of its own that nothing
  /// presents. Everything that reads swapchain_format(), swapchain_extent() or
  /// current_backbuffer() sees a normal swapchain, so a renderer draws a whole
  /// frame without a window; acquire() and end_frame() drop only the drawable
  /// and the present.
  [[nodiscard]] gfx::swapchain_handle
  create_offscreen_swapchain(u32 width, u32 height, gfx::format format);
  [[nodiscard]] bool resize_swapchain(gfx::swapchain_handle swapchain, u32 width,
                                      u32 height);
  void destroy_swapchain(gfx::swapchain_handle swapchain) noexcept;

  /// Whether @p swapchain is one with no layer behind it.
  [[nodiscard]] bool offscreen_swapchain(gfx::swapchain_handle swapchain) const;

  [[nodiscard]] bool supports(gfx::format format, gfx::texture_usage_flags usage) const;

  /// See VulkanDevice::swapchain_format.
  [[nodiscard]] gfx::format swapchain_format() const noexcept;
  [[nodiscard]] gfx::extent2d swapchain_extent(gfx::swapchain_handle swapchain) const;
  [[nodiscard]] gfx::surface_alpha swapchain_alpha(gfx::swapchain_handle swapchain) const;
  /// A layer resizes with its drawableSize, so nothing goes stale here.
  [[nodiscard]] bool swapchain_stale(gfx::swapchain_handle) const { return false; }

  /// See VulkanDevice::begin_frame, acquire and end_frame.
  [[nodiscard]] bool begin_frame(MetalCommandContext &out_context);
  [[nodiscard]] gfx::acquire_result acquire(gfx::swapchain_handle swapchain);
  [[nodiscard]] bool end_frame();

  // A frame records into one command buffer here, so caps() reports no
  // parallel_recording and nothing is ever split off it.
  [[nodiscard]] gfx::recording_split split_frame(MetalCommandContext &, u32) {
    return {};
  }
  [[nodiscard]] bool begin_recording(gfx::recording_split, u32,
                                     MetalCommandContext &) {
    return false;
  }
  void end_recording(MetalCommandContext &) {}

  [[nodiscard]] bool begin_headless_frame(MetalCommandContext &out_context);

  [[nodiscard]] bool end_headless_frame();

  [[nodiscard]] gfx::texture_handle
  current_backbuffer(gfx::swapchain_handle swapchain) const noexcept;

  /// Waits for the GPU, then frees every retired resource. Never while
  /// another thread records or submits a frame: that frame may hold what this
  /// frees (an engine waits for its render job first, Engine::wait_gpu_idle).
  void wait_idle() noexcept;
  /// See VulkanDevice::wait_submitted.
  void wait_submitted() noexcept;
  /// See VulkanDevice::save_pipeline_cache; writes the binary archive.
  bool save_pipeline_cache() noexcept;
  /// Metal asks no client before the device exists, so it makes no queue of
  /// a client's.
  [[nodiscard]] gfx::native_queue
  native_queue(const gfx::native_device_client &) const noexcept {
    return {};
  }

  /// See VulkanDevice::hold_retirements.
  void hold_retirements(u64 serial) noexcept;
  void release_retirements(u64 serial) noexcept;

  [[nodiscard]] gfx::texture_handle create_texture(const gfx::texture_desc &desc);
  void destroy_texture(gfx::texture_handle handle) noexcept;
  /// See VulkanDevice::replace_texture.
  [[nodiscard]] bool replace_texture(gfx::texture_handle target,
                                     gfx::texture_handle replacement);
  /// Mip-residency streaming is not yet wired on Metal; returns false so the
  /// engine uploads the full chain instead (see the Vulkan implementation).
  [[nodiscard]] bool set_texture_view_base_mip(gfx::texture_handle handle,
                                               u32 base_mip);
  [[nodiscard]] bool supports_mip_streaming() const noexcept { return false; }
  [[nodiscard]] gfx::texture_usage_flags
  texture_usage(gfx::texture_handle handle) const noexcept;

  [[nodiscard]] gfx::buffer_handle create_buffer(const gfx::buffer_desc &desc);
  void destroy_buffer(gfx::buffer_handle handle) noexcept;

  [[nodiscard]] gfx::memory_requirements
  texture_requirements(const gfx::texture_desc &desc) const;
  [[nodiscard]] gfx::heap_handle create_heap(const gfx::heap_desc &desc);
  void destroy_heap(gfx::heap_handle handle) noexcept;
  [[nodiscard]] gfx::texture_handle
  create_texture_in_heap(const gfx::texture_desc &desc, gfx::heap_handle heap, u64 offset);

  [[nodiscard]] gfx::sampler_handle create_sampler(const gfx::sampler_desc &desc);
  void destroy_sampler(gfx::sampler_handle handle) noexcept;

  [[nodiscard]] gfx::query_pool_handle create_timestamp_pool(u32) { return {}; }
  void destroy_query_pool(gfx::query_pool_handle) noexcept {}
  [[nodiscard]] bool read_timestamps(gfx::query_pool_handle, std::span<u64>) const {
    return false;
  }

  [[nodiscard]] gfx::shader_handle create_shader(const gfx::shader_desc &desc);
  void destroy_shader(gfx::shader_handle handle) noexcept;

  /// See VulkanDevice::prepare_pipeline.
  [[nodiscard]] gfx::pipeline_job prepare_pipeline(const gfx::graphics_pipeline_desc &desc);
  [[nodiscard]] gfx::pipeline_job prepare_pipeline(const gfx::compute_pipeline_desc &desc);
  [[nodiscard]] gfx::pipeline_handle publish_pipeline(gfx::pipeline_job &&job);
  [[nodiscard]] nx::workload_statistics
  pipeline_compile_statistics() const noexcept {
    return m_pipeline_compile_workload.statistics();
  }
  void destroy_pipeline(gfx::pipeline_handle handle) noexcept;

  [[nodiscard]] const gfx::shader_reflection &
  reflection(gfx::shader_handle handle) const noexcept;

  [[nodiscard]] gfx::resource_state
  texture_state(gfx::texture_handle handle) const noexcept;
  void set_texture_state(gfx::texture_handle handle, gfx::resource_state state) noexcept;

  [[nodiscard]] u32 texture_index(gfx::texture_handle handle) const noexcept;
  [[nodiscard]] u32 rw_texture_index(gfx::texture_handle handle,
                                     u32 mip = 0) const noexcept;
  [[nodiscard]] u32 sampler_index(gfx::sampler_handle handle) const noexcept;
  [[nodiscard]] u64 buffer_address(gfx::buffer_handle handle) const noexcept;

  [[nodiscard]] void *buffer_mapped(gfx::buffer_handle handle) const noexcept;
  // Upload buffers use MTLStorageModeShared, whose CPU writes are immediately
  // visible to the GPU on supported Apple platforms.
  void flush_buffer(gfx::buffer_handle, u64 = 0, u64 = ~0ull) noexcept {}
  void invalidate_buffer(gfx::buffer_handle, u64 = 0, u64 = ~0ull) noexcept {}

private:
  gfx::device_caps m_caps;
  nx::string m_device_name;
  nx::string m_init_error;
  gfx::device_init_report m_init_report;
  MetalDeviceState *m_state = nullptr;
  MetalUploader m_uploader;
  nx::workload_tracker m_pipeline_compile_workload{"pipeline::compile"};

  // Neither takes retire_lock on entry; each waits for the GPU first.
  bool swap_texture(gfx::texture_handle target, gfx::texture_handle replacement) noexcept;
  void apply_replacements() noexcept;
};

} // namespace nx::gfx::detail
