#pragma once

/**
 * @file mtl_device.h
 * @brief Metal device, swapchain and command context.
 *
 * Pure C++ by necessity: device.h pulls this into ordinary .cpp files, and
 * every Metal type is an Objective-C object. All of that lives behind
 * MetalDeviceState, defined only in mtl_internal.h.
 */

#include "foundation/strings/utf8_string.h"
#include "rendering/rhi/bindless.h"
#include "rendering/rhi/descs.h"
#include "rendering/rhi/handles.h"
#include "rendering/rhi/image/image.h"
#include "rendering/rhi/shader_reflect.h"
#include "rendering/rhi/types.h"
#include "rendering/rhi/upload.h"

#include <span>

namespace nxe::rhi {

class MetalDevice;

/// The backend's Objective-C state. Opaque here on purpose; see mtl_internal.h.
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
  /// File the pipeline cache is loaded from and saved to. Empty keeps the cache
  /// in memory only (recompiled every run). Metal's equivalent is an
  /// MTLBinaryArchive; unused until pipelines land.
  nx::string_view pipeline_cache_path;
};

/// In-flight readback. `data` is valid once the ticket completes, and only
/// until a later read_* reuses the staging - copy it out promptly.
struct ReadbackResult {
  UploadTicket ticket;
  const u8 *data = nullptr;
  u64 size = 0;

  [[nodiscard]] std::span<const u8> bytes() const noexcept {
    return {data, nx::cast<usize>(size)};
  }
};

/// A snapshot of the uploader's internal load, for profiling and stress tests.
struct UploaderStats {
  u64 total_submits = 0;     ///< Lifetime batch submissions.
  u64 completed_submits = 0; ///< Of those, finished on the GPU.
  u64 ring_used = 0;         ///< Bytes staged into the batch still open.
  u64 ring_capacity = 0;     ///< Total capacity of every pooled staging buffer.
  usize extra_buffers = 0;   ///< Staging buffers sitting idle in the pool.
  usize inflight_submits = 0; ///< Batches whose GPU work is not done yet.
  u64 loads_ready = 0;        ///< Async loads that reached Ready.
  u64 loads_failed = 0;
};

/// Asynchronous GPU upload, readback and file/decode streaming. Uploads are
/// batched: a call records a blit and returns a ticket, but nothing is
/// committed until the batch flushes (once per frame, or on flush()/wait()).
class MetalUploader {
public:
  MetalUploader() = default;

  MetalUploader(const MetalUploader &) = delete;
  MetalUploader &operator=(const MetalUploader &) = delete;

  static constexpr TextureUsageFlags DEFAULT_LOAD_USAGE =
      TextureUsage::Sampled | TextureUsage::CopyDst;

  UploadTicket
  upload_texture(TextureHandle handle, std::span<const u8> src,
                 std::span<const ImageSubresource> layout,
                 ResourceState final_state = ResourceState::ShaderReadOnly);
  UploadTicket
  upload_texture(TextureHandle handle, const ImageData &image,
                 ResourceState final_state = ResourceState::ShaderReadOnly);
  UploadTicket
  upload_buffer(BufferHandle handle, u64 dst_offset, std::span<const u8> src,
                ResourceState final_state = ResourceState::ShaderReadOnly);

  LoadRequest load_texture(nx::string_view path,
                           TextureUsageFlags usage = DEFAULT_LOAD_USAGE);
  LoadRequest
  load_texture_from_memory(std::span<const u8> bytes,
                           TextureUsageFlags usage = DEFAULT_LOAD_USAGE);

  [[nodiscard]] LoadStatus load_status(LoadRequest req) const noexcept;
  [[nodiscard]] TextureHandle loaded_texture(LoadRequest req) const noexcept;
  void release(LoadRequest req) noexcept;

  // Readbacks (render thread). The source must not be in concurrent GPU use.
  // A readback flushes any open upload batch and submits on its own.

  ReadbackResult read_buffer(BufferHandle handle, u64 offset, u64 size);
  ReadbackResult read_texture(TextureHandle handle, u32 mip = 0, u32 layer = 0);

  [[nodiscard]] bool is_complete(UploadTicket ticket) const noexcept;
  void wait(UploadTicket ticket) noexcept;
  void flush() noexcept;

  [[nodiscard]] UploaderStats stats() const noexcept;

  /// Turns finished decodes into textures and submits their uploads. The
  /// device calls it once per frame; anything driving the uploader without
  /// frames must call it, or loads never leave Uploading.
  void pump();

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

  void bind_pipeline(PipelineHandle pipeline);
  /// Sets the whole 128-byte push block. Offset/size in bytes.
  void push_constants(const void *data, u32 size, u32 offset = 0);

  /// Pushes what a BlockWriter staged, at the size reflection reported.
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

  /// Only bookkeeping: Metal tracks non-heap hazards itself, and heap ones are
  /// handled by the pass fence. The state drives load/store actions.
  void barrier(const TextureBarrier *textures, u32 texture_count,
               const BufferBarrier *buffers = nullptr, u32 buffer_count = 0,
               StateTracking tracking = StateTracking::Update);
  void barrier(const TextureBarrier &texture) { barrier(&texture, 1); }

  void push_debug_group(nx::string_view name, f32 r = 1.f, f32 g = 1.f,
                        f32 b = 1.f);
  void pop_debug_group();

private:
  friend class MetalDevice;

  /// Defined in mtl_device.mm; resolves m_slot against the device.
  [[nodiscard]] struct MetalContextSlot &ctx_slot() const noexcept;

  MetalDeviceState *m_state = nullptr;
  /// Index into the device's recording slots. Contexts record on different
  /// threads, so mutable state lives per slot, not on the device.
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

  [[nodiscard]] const DeviceCaps &caps() const noexcept { return m_caps; }

  [[nodiscard]] MetalUploader &uploader() noexcept { return m_uploader; }

  // Swapchain

  [[nodiscard]] bool create_swapchain(const SwapchainDesc &desc);
  [[nodiscard]] bool resize_swapchain(u32 width, u32 height);
  void destroy_swapchain() noexcept;

  [[nodiscard]] Format swapchain_format() const noexcept;
  [[nodiscard]] Extent2D swapchain_extent() const noexcept;

  // Frame

  [[nodiscard]] AcquireResult begin_frame(MetalCommandContext &out_context);
  [[nodiscard]] bool end_frame();

  [[nodiscard]] bool begin_headless_frame(MetalCommandContext &out_context);

  [[nodiscard]] bool end_headless_frame();

  /// Always fails for now: every MetalCommandContext shares the device's one
  /// command buffer and encoder, so there is nothing to hand out. Parallel
  /// encoding needs a command buffer per context, ordered with -enqueue.
  /// Callers fall back to recording serially.
  [[nodiscard]] bool open_contexts(std::span<MetalCommandContext> out);

  [[nodiscard]] TextureHandle current_backbuffer() const noexcept;

  void wait_idle() noexcept;

  // Resources

  [[nodiscard]] TextureHandle create_texture(const TextureDesc &desc);
  void destroy_texture(TextureHandle handle) noexcept;

  [[nodiscard]] BufferHandle create_buffer(const BufferDesc &desc);
  void destroy_buffer(BufferHandle handle) noexcept;

  // Memory aliasing. The render graph packs transients with disjoint lifetimes
  // into one heap; see the Vulkan header for the contract, which is identical.
  [[nodiscard]] MemoryRequirements
  texture_requirements(const TextureDesc &desc) const;
  [[nodiscard]] HeapHandle create_heap(const HeapDesc &desc);
  void destroy_heap(HeapHandle handle) noexcept;
  [[nodiscard]] TextureHandle
  create_texture_in_heap(const TextureDesc &desc, HeapHandle heap, u64 offset);

  [[nodiscard]] SamplerHandle create_sampler(const SamplerDesc &desc);
  void destroy_sampler(SamplerHandle handle) noexcept;

  // Shaders and pipelines. NOT IMPLEMENTED YET: these return null handles, so
  // callers skip the draw and fall back to the clear-only path.

  [[nodiscard]] ShaderHandle create_shader(const ShaderDesc &desc);
  void destroy_shader(ShaderHandle handle) noexcept;

  [[nodiscard]] PipelineHandle
  create_graphics_pipeline(const GraphicsPipelineDesc &desc);
  [[nodiscard]] PipelineHandle
  create_compute_pipeline(const ComputePipelineDesc &desc);
  void destroy_pipeline(PipelineHandle handle) noexcept;

  // Bindless indices, for the push block. The heap needs argument-buffer
  // Tier 2 and is not built yet, so these report "no slot" for now.

  /// The shader's reflected interface. Empty for an unknown handle.
  [[nodiscard]] const ShaderReflection &
  reflection(ShaderHandle handle) const noexcept;

  /// Tracked state, for the graph to resolve transitions against up front.
  [[nodiscard]] ResourceState
  texture_state(TextureHandle handle) const noexcept;
  /// Pairs with StateTracking::Skip: the caller writes back what it recorded.
  void set_texture_state(TextureHandle handle, ResourceState state) noexcept;

  [[nodiscard]] u32 texture_index(TextureHandle handle) const noexcept;
  [[nodiscard]] u32 rw_texture_index(TextureHandle handle,
                                     u32 mip = 0) const noexcept;
  [[nodiscard]] u32 sampler_index(SamplerHandle handle) const noexcept;
  [[nodiscard]] u64 buffer_address(BufferHandle handle) const noexcept;

  /// Host pointer for a persistently mapped buffer, else null. Per-frame data
  /// is written straight through this rather than staged and copied.
  [[nodiscard]] void *buffer_mapped(BufferHandle handle) const noexcept;

private:
  DeviceCaps m_caps;
  /// Backing store for m_caps.device_name, which is only a view.
  nx::string m_device_name;
  MetalDeviceState *m_state = nullptr;
  MetalUploader m_uploader;
};

} // namespace nxe::rhi
