#pragma once

#include "core/foundation/strings/utf8_string.h"
#include "core/foundation/threading/thread_pool.h"
#include "core/foundation/threading/workload.h"
#include "rendering/rhi/bindless.h"
#include "rendering/rhi/descs.h"
#include "rendering/rhi/gpu_crash.h"
#include "rendering/rhi/handles.h"
#include "rendering/rhi/image/image.h"
#include "rendering/rhi/native_driver.h"
#include "rendering/rhi/pipeline_compiler.h"
#include "rendering/rhi/shader_reflect.h"
#include "rendering/rhi/types.h"
#include "rendering/rhi/upload.h"

#include <span>

namespace nxe::rhi {

class MetalDevice;

struct MetalDeviceState;

struct DeviceDesc {
  nx::string_view application_name = "nx2d";
  u32 application_version = 0;
#if NX_DEBUG
  bool enable_validation = true;
#else
  bool enable_validation = false;
#endif
  bool break_on_validation_error = false;
  GpuCrashMode gpu_crash_mode = GpuCrashMode::Basic;
  nx::string_view pipeline_cache_path;
  NativeDriverView native_driver;
  u32 frames_in_flight = 2;
};

struct ReadbackResult {
  UploadTicket ticket;
  const u8 *data = nullptr;
  u64 size = 0;

  [[nodiscard]] std::span<const u8> bytes() const noexcept {
    return {data, nx::cast<usize>(size)};
  }
};

struct UploaderStats {
  u64 total_submits = 0; ///< Lifetime batch submissions.
  u64 completed_submits = 0;
  u64 ring_used = 0; ///< Bytes staged into the batch still open.
  u64 ring_capacity = 0;
  usize extra_buffers = 0;
  usize inflight_submits = 0;
};

/// Asynchronous GPU upload and readback. Uploads are
/// batched: a call records a blit and returns a ticket, but nothing is
/// committed until the batch flushes (once per frame, or on flush()/wait()).
class MetalUploader {
public:
  MetalUploader() = default;

  MetalUploader(const MetalUploader &) = delete;
  MetalUploader &operator=(const MetalUploader &) = delete;

  [[nodiscard]] UploadTicket
  upload_texture(TextureHandle handle, std::span<const u8> src,
                 std::span<const ImageSubresource> layout,
                 ResourceState final_state = ResourceState::ShaderReadOnly);
  [[nodiscard]] UploadTicket
  upload_texture(TextureHandle handle, const ImageData &image,
                 ResourceState final_state = ResourceState::ShaderReadOnly);
  [[nodiscard]] UploadTicket
  upload_buffer(BufferHandle handle, u64 dst_offset, std::span<const u8> src,
                ResourceState final_state = ResourceState::ShaderReadOnly);
  [[nodiscard]] UploadTicket
  clear_buffer(BufferHandle handle,
               ResourceState final_state = ResourceState::ShaderReadOnly);

  // Readbacks (render thread). The source must not be in concurrent GPU use.
  // A readback flushes any open upload batch and submits on its own.

  ReadbackResult read_buffer(BufferHandle handle, u64 offset, u64 size);
  ReadbackResult read_texture(TextureHandle handle, u32 mip = 0, u32 layer = 0);

  [[nodiscard]] bool is_complete(UploadTicket ticket) const noexcept;
  void wait(UploadTicket ticket) noexcept;
  void flush() noexcept;
  /// Submits any open batch without waiting for it, matching
  /// VulkanUploader::flush_batch() - a caller that needs completion still
  /// calls wait() on the ticket.
  void flush_batch() noexcept;

  [[nodiscard]] UploaderStats stats() const noexcept;

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

  void begin_render_pass(const RenderPassDesc &desc);
  void end_render_pass();

  void begin_compute_pass(nx::string_view name = "compute");
  void end_compute_pass();

  void set_viewport(const Viewport &viewport);
  void set_scissor(const Rect2D &rect);

  void reset_queries(QueryPoolHandle, u32, u32) {}
  void write_timestamp(QueryPoolHandle, u32) {}

  void bind_pipeline(PipelineHandle pipeline);
  /// Sets the whole 128-byte push block. Offset/size in bytes.
  void push_constants(const void *data, u32 size, u32 offset = 0);

  void push_constants(const BlockWriter &block) {
    push_constants(block.bytes().data(), nx::cast<u32>(block.bytes().size()));
  }

  void bind_index_buffer(BufferHandle buffer, IndexType type, u64 offset = 0);

  void draw(u32 vertex_count, u32 instance_count = 1, u32 first_vertex = 0,
            u32 first_instance = 0);
  void draw_indexed(u32 index_count, u32 instance_count = 1,
                    u32 first_index = 0, i32 vertex_offset = 0,
                    u32 first_instance = 0);
  void dispatch(u32 groups_x, u32 groups_y = 1, u32 groups_z = 1);

  void draw_indirect(BufferHandle buffer, u64 offset = 0, u32 draw_count = 1,
                     u32 stride = 0);
  void draw_indexed_indirect(BufferHandle buffer, u64 offset = 0,
                             u32 draw_count = 1, u32 stride = 0);
  void dispatch_indirect(BufferHandle buffer, u64 offset = 0);

  void barrier(const TextureBarrier *textures, u32 texture_count,
               const BufferBarrier *buffers = nullptr, u32 buffer_count = 0,
               StateTracking tracking = StateTracking::Update);
  void barrier(const TextureBarrier &texture) { barrier(&texture, 1); }

  /// Copies one tightly packed colour mip into a host-readable buffer. The
  /// source must already be in CopySrc and this must be called outside a
  /// render/compute pass. row_pitch may include padding between rows.
  [[nodiscard]] bool copy_texture_to_buffer(TextureHandle texture,
                                            BufferHandle buffer, u64 row_pitch,
                                            u32 mip = 0, u32 layer = 0);

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

  [[nodiscard]] bool init(const DeviceDesc &desc);
  void shutdown() noexcept;

  [[nodiscard]] nx::string_view init_error() const noexcept {
    return m_init_error;
  }
  [[nodiscard]] const DeviceInitReport &init_report() const noexcept {
    return m_init_report;
  }

  [[nodiscard]] const DeviceCaps &caps() const noexcept { return m_caps; }
  [[nodiscard]] GpuDeviceStatus device_status() const noexcept;

  [[nodiscard]] u32 frames_in_flight() const noexcept;

  [[nodiscard]] MetalUploader &uploader() noexcept { return m_uploader; }

  void flush_uploads() noexcept;

  /// See VulkanDevice::create_swapchain.
  [[nodiscard]] SwapchainHandle create_swapchain(const SwapchainDesc &desc);
  /// A swapchain with no layer behind it: a texture of its own that nothing
  /// presents. Everything that reads swapchain_format(), swapchain_extent() or
  /// current_backbuffer() sees a normal swapchain, so a renderer draws a whole
  /// frame without a window; acquire() and end_frame() drop only the drawable
  /// and the present.
  [[nodiscard]] SwapchainHandle
  create_offscreen_swapchain(u32 width, u32 height, Format format);
  [[nodiscard]] bool resize_swapchain(SwapchainHandle swapchain, u32 width,
                                      u32 height);
  void destroy_swapchain(SwapchainHandle swapchain) noexcept;

  /// Whether @p swapchain is one with no layer behind it.
  [[nodiscard]] bool offscreen_swapchain(SwapchainHandle swapchain) const;

  [[nodiscard]] bool supports(Format format, TextureUsageFlags usage) const;

  /// See VulkanDevice::swapchain_format.
  [[nodiscard]] Format swapchain_format() const noexcept;
  [[nodiscard]] Extent2D swapchain_extent(SwapchainHandle swapchain) const;
  [[nodiscard]] SurfaceAlpha swapchain_alpha(SwapchainHandle swapchain) const;
  /// A layer resizes with its drawableSize, so nothing goes stale here.
  [[nodiscard]] bool swapchain_stale(SwapchainHandle) const { return false; }

  /// See VulkanDevice::begin_frame, acquire and end_frame.
  [[nodiscard]] bool begin_frame(MetalCommandContext &out_context);
  [[nodiscard]] AcquireResult acquire(SwapchainHandle swapchain);
  [[nodiscard]] bool end_frame();

  // A frame records into one command buffer here, so caps() reports no
  // parallel_recording and nothing is ever split off it.
  [[nodiscard]] RecordingSplit split_frame(MetalCommandContext &, u32) {
    return {};
  }
  [[nodiscard]] bool begin_recording(RecordingSplit, u32,
                                     MetalCommandContext &) {
    return false;
  }
  void end_recording(MetalCommandContext &) {}

  [[nodiscard]] bool begin_headless_frame(MetalCommandContext &out_context);

  [[nodiscard]] bool end_headless_frame();

  [[nodiscard]] TextureHandle
  current_backbuffer(SwapchainHandle swapchain) const noexcept;

  void wait_idle() noexcept;

  /// See VulkanDevice::hold_retirements.
  void hold_retirements(u64 serial) noexcept;
  void release_retirements(u64 serial) noexcept;

  [[nodiscard]] TextureHandle create_texture(const TextureDesc &desc);
  void destroy_texture(TextureHandle handle) noexcept;
  /// See VulkanDevice::replace_texture.
  [[nodiscard]] bool replace_texture(TextureHandle target,
                                     TextureHandle replacement);
  /// Mip-residency streaming is not yet wired on Metal; returns false so the
  /// engine uploads the full chain instead (see the Vulkan implementation).
  [[nodiscard]] bool set_texture_view_base_mip(TextureHandle handle,
                                               u32 base_mip);
  [[nodiscard]] bool supports_mip_streaming() const noexcept { return false; }
  [[nodiscard]] TextureUsageFlags
  texture_usage(TextureHandle handle) const noexcept;

  [[nodiscard]] BufferHandle create_buffer(const BufferDesc &desc);
  void destroy_buffer(BufferHandle handle) noexcept;

  [[nodiscard]] MemoryRequirements
  texture_requirements(const TextureDesc &desc) const;
  [[nodiscard]] HeapHandle create_heap(const HeapDesc &desc);
  void destroy_heap(HeapHandle handle) noexcept;
  [[nodiscard]] TextureHandle
  create_texture_in_heap(const TextureDesc &desc, HeapHandle heap, u64 offset);

  [[nodiscard]] SamplerHandle create_sampler(const SamplerDesc &desc);
  void destroy_sampler(SamplerHandle handle) noexcept;

  [[nodiscard]] QueryPoolHandle create_timestamp_pool(u32) { return {}; }
  void destroy_query_pool(QueryPoolHandle) noexcept {}
  [[nodiscard]] bool read_timestamps(QueryPoolHandle, std::span<u64>) const {
    return false;
  }

  [[nodiscard]] ShaderHandle create_shader(const ShaderDesc &desc);
  void destroy_shader(ShaderHandle handle) noexcept;

  [[nodiscard]] NX_DEPRECATED_MSG(
      "use create_graphics_pipeline_async and publish_pipeline") PipelineHandle
      create_graphics_pipeline(const GraphicsPipelineDesc &desc);
  [[nodiscard]] NX_DEPRECATED_MSG(
      "use create_compute_pipeline_async and publish_pipeline") PipelineHandle
      create_compute_pipeline(const ComputePipelineDesc &desc);
  /// Copies descriptor inputs on the owner, then performs native compilation
  /// on a worker. Only publication of the opaque result touches handle tables.
  [[nodiscard]] nx::future<CompiledPipeline> create_graphics_pipeline_async(
      nx::thread_pool &workers, const GraphicsPipelineDesc &desc,
      nx::task_priority priority = nx::task_priority::Normal);
  [[nodiscard]] nx::future<CompiledPipeline> create_compute_pipeline_async(
      nx::thread_pool &workers, const ComputePipelineDesc &desc,
      nx::task_priority priority = nx::task_priority::Normal);
  [[nodiscard]] PipelineHandle publish_pipeline(CompiledPipeline &&compiled);
  [[nodiscard]] nx::workload_statistics
  pipeline_compile_statistics() const noexcept {
    return m_pipeline_compile_workload.statistics();
  }
  void destroy_pipeline(PipelineHandle handle) noexcept;

  [[nodiscard]] const ShaderReflection &
  reflection(ShaderHandle handle) const noexcept;

  [[nodiscard]] ResourceState
  texture_state(TextureHandle handle) const noexcept;
  void set_texture_state(TextureHandle handle, ResourceState state) noexcept;

  [[nodiscard]] u32 texture_index(TextureHandle handle) const noexcept;
  [[nodiscard]] u32 rw_texture_index(TextureHandle handle,
                                     u32 mip = 0) const noexcept;
  [[nodiscard]] u32 sampler_index(SamplerHandle handle) const noexcept;
  [[nodiscard]] u64 buffer_address(BufferHandle handle) const noexcept;

  [[nodiscard]] void *buffer_mapped(BufferHandle handle) const noexcept;
  // Upload buffers use MTLStorageModeShared, whose CPU writes are immediately
  // visible to the GPU on supported Apple platforms.
  void flush_buffer(BufferHandle, u64 = 0, u64 = ~0ull) noexcept {}
  void invalidate_buffer(BufferHandle, u64 = 0, u64 = ~0ull) noexcept {}

private:
  DeviceCaps m_caps;
  nx::string m_device_name;
  nx::string m_init_error;
  DeviceInitReport m_init_report;
  MetalDeviceState *m_state = nullptr;
  MetalUploader m_uploader;
  nx::workload_tracker m_pipeline_compile_workload;

  // Neither takes retire_lock on entry; each waits for the GPU first.
  bool swap_texture(TextureHandle target, TextureHandle replacement) noexcept;
  void apply_replacements() noexcept;
};

} // namespace nxe::rhi
