#pragma once

/**
 * @file mtl_device.h
 * @brief Metal device, swapchain and command context.
 *
 * Unlike the Vulkan backend, whose header can name VkImage and friends because
 * they are plain C handles, every Metal type is an Objective-C object. This
 * header is included from ordinary .cpp translation units (device.h pulls it
 * in for the Device alias), so it must stay pure C++: all Objective-C state
 * lives behind MetalDeviceState, which is only defined in mtl_internal.h and
 * only visible to the .mm files.
 */

#include "foundation/strings/utf8_string.h"
#include "rendering/rhi/bindless.h"
#include "rendering/rhi/descs.h"
#include "rendering/rhi/handles.h"
#include "rendering/rhi/image/image.h"
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

/**
 * @brief Asynchronous GPU upload and streaming service.
 *
 * NOT IMPLEMENTED YET - every entry point here is a stub that fails cleanly, so
 * that code written against the Vulkan backend compiles and runs (drawing
 * nothing) on Metal. See docs: the uploader lands with the resource pass.
 */
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

  [[nodiscard]] bool is_complete(UploadTicket ticket) const noexcept;
  void wait(UploadTicket ticket) noexcept;
  void flush() noexcept;

private:
  friend class MetalDevice;

  MetalDeviceState *m_state = nullptr;
};

/**
 * @brief Records one frame's work into the frame's MTLCommandBuffer.
 *
 * Metal has no standalone command buffer that outlives an encoder, so a context
 * owns at most one encoder at a time: begin_render_pass opens a render command
 * encoder and end_render_pass closes it.
 */
class MetalCommandContext {
public:
  MetalCommandContext() = default;

  [[nodiscard]] bool valid() const noexcept { return m_state != nullptr; }

  void begin_render_pass(const RenderPassDesc &desc);
  void end_render_pass();

  void set_viewport(const Viewport &viewport);
  void set_scissor(const Rect2D &rect);

  void bind_pipeline(PipelineHandle pipeline);
  /// Sets the whole 128-byte push block. Offset/size in bytes.
  void push_constants(const void *data, u32 size, u32 offset = 0);

  void draw(u32 vertex_count, u32 instance_count = 1, u32 first_vertex = 0,
            u32 first_instance = 0);
  void draw_indexed(u32 index_count, u32 instance_count = 1,
                    u32 first_index = 0, i32 vertex_offset = 0,
                    u32 first_instance = 0);
  void dispatch(u32 groups_x, u32 groups_y = 1, u32 groups_z = 1);

  /// Metal tracks hazards between encoders for itself, so these only keep the
  /// RHI's own view of each texture's state up to date - which is what
  /// begin_render_pass and the swapchain read to decide load/store actions.
  void barrier(const TextureBarrier *textures, u32 texture_count,
               const BufferBarrier *buffers = nullptr, u32 buffer_count = 0);
  void barrier(const TextureBarrier &texture) { barrier(&texture, 1); }

  void push_debug_group(nx::string_view name, f32 r = 1.f, f32 g = 1.f,
                        f32 b = 1.f);
  void pop_debug_group();

private:
  friend class MetalDevice;

  MetalDeviceState *m_state = nullptr;
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

  [[nodiscard]] TextureHandle current_backbuffer() const noexcept;

  void wait_idle() noexcept;

  // Resources

  [[nodiscard]] TextureHandle create_texture(const TextureDesc &desc);
  void destroy_texture(TextureHandle handle) noexcept;

  [[nodiscard]] BufferHandle create_buffer(const BufferDesc &desc);
  void destroy_buffer(BufferHandle handle) noexcept;

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

  [[nodiscard]] u32 texture_index(TextureHandle handle) const noexcept;
  [[nodiscard]] u32 rw_texture_index(TextureHandle handle,
                                     u32 mip = 0) const noexcept;
  [[nodiscard]] u32 sampler_index(SamplerHandle handle) const noexcept;
  [[nodiscard]] u64 buffer_address(BufferHandle handle) const noexcept;

private:
  DeviceCaps m_caps;
  /// Backing store for m_caps.device_name, which is only a view.
  nx::string m_device_name;
  MetalDeviceState *m_state = nullptr;
  MetalUploader m_uploader;
};

} // namespace nxe::rhi
