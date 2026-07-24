#pragma once

/**
 * @file vk_device.h
 * @brief Vulkan 1.3 device, swapchain and bindless heap.
 */

#include "foundation/containers/fixed_vector.h"
#include "foundation/containers/slot_map.h"
#include "foundation/containers/small_vector.h"
#include "foundation/strings/utf8_string.h"
#include "rendering/rhi/bindless.h"
#include "rendering/rhi/descs.h"
#include "rendering/rhi/handles.h"
#include "rendering/rhi/types.h"
#include "rendering/rhi/vulkan/vk_common.h"

#include <vk_mem_alloc.h>

namespace nxe::rhi {

class VulkanDevice;

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
  /// in memory only (recompiled every run).
  nx::string_view pipeline_cache_path;
};

/// One VkImage, several views: a sampled view spans the whole mip chain, but an
/// attachment or storage view must address exactly one mip.
///   - srv: sampling; its bindless slot is the texture's handle index.
///   - rtv: colour/depth attachment; mip 0.
///   - uavs[i] / uav_indices[i]: storage view of mip i and its bindless slot.
struct VulkanTexture {
  VkImage image = VK_NULL_HANDLE;
  VkImageView srv = VK_NULL_HANDLE;
  VkImageView rtv = VK_NULL_HANDLE;
  nx::small_vector<VkImageView, 12> uavs;
  nx::small_vector<u32, 12> uav_indices;
  VmaAllocation allocation = VK_NULL_HANDLE;
  Format format = Format::Unknown;
  TextureDimension dimension = TextureDimension::Tex2D;
  u32 width = 0;
  u32 height = 0;
  u32 mip_levels = 1;
  u32 layers = 1;
  TextureUsageFlags usage = TextureUsage::None;
  ResourceState state = ResourceState::Undefined;
  bool owns_image = true;
};

struct VulkanBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  VkDeviceAddress device_address = 0;
  void *mapped = nullptr;
  u64 size = 0;
  BufferUsageFlags usage = BufferUsage::None;
};

struct VulkanSampler {
  VkSampler sampler = VK_NULL_HANDLE;
};

struct VulkanShader {
  VkShaderModule module = VK_NULL_HANDLE;
};

struct VulkanPipeline {
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkPipelineBindPoint bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
};

/// The single bindless descriptor set: sampler[], texture[] (sampled) and
/// rwTexture[] (storage). Sampler and texture slots are handle indices;
/// rwTexture slots come from a free list because a texture owns one per mip.
struct BindlessHeap {
  VkDescriptorPool pool = VK_NULL_HANDLE;
  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  VkDescriptorSet set = VK_NULL_HANDLE;
  u32 max_samplers = 0;
  u32 max_textures = 0;
  u32 max_rw_textures = 0;
  nx::small_vector<u32, 64> free_rw_slots;
  u32 rw_high_water = 0;
};

class VulkanCommandContext {
public:
  VulkanCommandContext() = default;

  [[nodiscard]] VkCommandBuffer handle() const noexcept { return m_cmd; }
  [[nodiscard]] bool valid() const noexcept { return m_cmd != VK_NULL_HANDLE; }

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

  /// Explicit barriers. The render graph is the intended caller.
  void barrier(const TextureBarrier *textures, u32 texture_count,
               const BufferBarrier *buffers = nullptr, u32 buffer_count = 0);
  void barrier(const TextureBarrier &texture) { barrier(&texture, 1); }

  void push_debug_group(nx::string_view name, f32 r = 1.f, f32 g = 1.f,
                        f32 b = 1.f);
  void pop_debug_group();

private:
  friend class VulkanDevice;

  VkCommandBuffer m_cmd = VK_NULL_HANDLE;
  VulkanDevice *m_device = nullptr;
  bool m_in_render_pass = false;
};

struct FrameContext {
  VkCommandPool command_pool = VK_NULL_HANDLE;
  nx::small_vector<VkCommandBuffer, 8> command_buffers;
  u32 used_command_buffers = 0;
  VkSemaphore acquire_semaphore = VK_NULL_HANDLE;
  u64 timeline_value = 0;
};

class VulkanDevice {
public:
  VulkanDevice() = default;
  ~VulkanDevice();

  VulkanDevice(const VulkanDevice &) = delete;
  VulkanDevice &operator=(const VulkanDevice &) = delete;

  [[nodiscard]] bool init(const DeviceDesc &desc);
  void shutdown() noexcept;

  [[nodiscard]] const DeviceCaps &caps() const noexcept { return m_caps; }

  // Swapchain

  [[nodiscard]] bool create_swapchain(const SwapchainDesc &desc);
  [[nodiscard]] bool resize_swapchain(u32 width, u32 height);
  void destroy_swapchain() noexcept;

  [[nodiscard]] Format swapchain_format() const noexcept {
    return m_swapchain_format;
  }
  [[nodiscard]] Extent2D swapchain_extent() const noexcept {
    return {m_swapchain_extent.width, m_swapchain_extent.height};
  }

  // Frame

  [[nodiscard]] AcquireResult begin_frame(VulkanCommandContext &out_context);
  [[nodiscard]] bool end_frame();

  [[nodiscard]] TextureHandle current_backbuffer() const noexcept {
    return m_backbuffers.empty() ? TextureHandle{}
                                 : m_backbuffers[m_image_index];
  }

  void wait_idle() noexcept;

  // Resources

  [[nodiscard]] TextureHandle create_texture(const TextureDesc &desc);
  void destroy_texture(TextureHandle handle) noexcept;
  [[nodiscard]] const VulkanTexture *
  texture(TextureHandle handle) const noexcept;

  [[nodiscard]] BufferHandle create_buffer(const BufferDesc &desc);
  void destroy_buffer(BufferHandle handle) noexcept;
  [[nodiscard]] const VulkanBuffer *buffer(BufferHandle handle) const noexcept;

  [[nodiscard]] SamplerHandle create_sampler(const SamplerDesc &desc);
  void destroy_sampler(SamplerHandle handle) noexcept;

  // Shaders and pipelines (vk_pipeline.cpp).

  [[nodiscard]] ShaderHandle create_shader(const ShaderDesc &desc);
  void destroy_shader(ShaderHandle handle) noexcept;

  [[nodiscard]] PipelineHandle
  create_graphics_pipeline(const GraphicsPipelineDesc &desc);
  [[nodiscard]] PipelineHandle
  create_compute_pipeline(const ComputePipelineDesc &desc);
  void destroy_pipeline(PipelineHandle handle) noexcept;

  // Bindless indices, for the push block.

  /// Slot to sample this texture through textures[]; INVALID if not Sampled.
  [[nodiscard]] u32 texture_index(TextureHandle handle) const noexcept;
  /// Slot for mip @p mip in rwTextures[]; INVALID if not Storage.
  [[nodiscard]] u32 rw_texture_index(TextureHandle handle,
                                     u32 mip = 0) const noexcept;
  [[nodiscard]] u32 sampler_index(SamplerHandle handle) const noexcept;
  [[nodiscard]] VkDeviceAddress
  buffer_address(BufferHandle handle) const noexcept;

  // Internals used by VulkanCommandContext

  [[nodiscard]] VkDevice vk_device() const noexcept { return m_device; }
  [[nodiscard]] VulkanTexture *texture_mut(TextureHandle handle) noexcept;
  [[nodiscard]] bool debug_utils_enabled() const noexcept {
    return m_debug_utils;
  }
  [[nodiscard]] VkPipelineLayout pipeline_layout() const noexcept {
    return m_pipeline_layout;
  }
  [[nodiscard]] VkDescriptorSet bindless_set() const noexcept {
    return m_bindless.set;
  }
  [[nodiscard]] const VulkanPipeline *
  pipeline(PipelineHandle handle) const noexcept;

private:
  bool create_instance(const DeviceDesc &desc);
  bool select_physical_device();
  bool create_logical_device();
  bool create_allocator();
  bool create_bindless_heap();
  bool create_pipeline_cache(nx::string_view path);
  void save_pipeline_cache() noexcept;
  bool create_frame_contexts(u32 frames_in_flight);
  bool build_swapchain(u32 width, u32 height);
  bool create_texture_views(VulkanTexture &texture, nx::string_view name);
  void destroy_backbuffer_views() noexcept;
  void set_object_name(u64 object, VkObjectType type, nx::string_view name);
  void collect_garbage(u64 completed_timeline_value) noexcept;
  void destroy_texture_now(VulkanTexture &texture) noexcept;

  // Bindless heap writes (vk_bindless.cpp).
  void write_texture_descriptor(u32 index, VkImageView view) noexcept;
  void write_rw_texture_descriptor(u32 index, VkImageView view) noexcept;
  void write_sampler_descriptor(u32 index, VkSampler sampler) noexcept;
  [[nodiscard]] u32 alloc_rw_slot() noexcept;
  void free_rw_slot(u32 slot) noexcept;

  VkInstance m_instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT m_messenger = VK_NULL_HANDLE;
  VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
  VkDevice m_device = VK_NULL_HANDLE;
  VmaAllocator m_allocator = VK_NULL_HANDLE;
  DeviceCaps m_caps;
  nx::string m_device_name;
  bool m_debug_utils = false;
  bool m_break_on_error = false;

  VkQueue m_queues[nx::cast<usize>(QueueType::Count)] = {};
  u32 m_queue_families[nx::cast<usize>(QueueType::Count)] = {};

  BindlessHeap m_bindless;
  VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
  VkPipelineCache m_pipeline_cache = VK_NULL_HANDLE;
  nx::string m_pipeline_cache_path;

  VkSemaphore m_timeline = VK_NULL_HANDLE;
  u64 m_timeline_value = 0;
  nx::small_vector<FrameContext, 3> m_frames;
  u32 m_frame_index = 0;
  u32 m_frames_in_flight = 2;

  VkSurfaceKHR m_surface = VK_NULL_HANDLE;
  VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
  VkExtent2D m_swapchain_extent = {};
  Format m_swapchain_format = Format::Unknown;
  VkPresentModeKHR m_present_mode = VK_PRESENT_MODE_FIFO_KHR;
  nx::small_vector<TextureHandle, 4> m_backbuffers;
  nx::small_vector<VkSemaphore, 4> m_release_semaphores;
  u32 m_image_index = 0;
  bool m_frame_active = false;

  nx::slot_map<VulkanTexture, TextureHandle> m_textures;
  nx::slot_map<VulkanBuffer, BufferHandle> m_buffers;
  nx::slot_map<VulkanSampler, SamplerHandle> m_samplers;
  nx::slot_map<VulkanShader, ShaderHandle> m_shaders;
  nx::slot_map<VulkanPipeline, PipelineHandle> m_pipelines;

  /// A resource freed during frame N stays alive - handle and descriptor slot
  /// included - until the GPU has passed N, then its slot_map entry is retired.
  /// Deferring the slot_map erase (not just the Vk objects) is what stops a
  /// reused handle index from stomping a descriptor still in flight.
  struct PendingDelete {
    u64 timeline_value = 0;
    TextureHandle texture;
    BufferHandle buffer;
    SamplerHandle sampler;
    PipelineHandle pipeline;
  };
  nx::small_vector<PendingDelete, 32> m_pending_deletes;
};

} // namespace nxe::rhi
