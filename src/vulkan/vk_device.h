#pragma once

#include "core/foundation/containers/fixed_vector.h"
#include "core/foundation/containers/slot_map.h"
#include "core/foundation/containers/small_vector.h"
#include "core/foundation/strings/utf8_string.h"
#include "core/foundation/threading/sync.h"
#include "core/foundation/threading/thread_pool.h"
#include "core/foundation/threading/workload.h"
#include "core/rendering/rhi/bindless.h"
#include "core/rendering/rhi/descs.h"
#include "core/rendering/rhi/handles.h"
#include "core/rendering/rhi/pipeline_compiler.h"
#include "core/rendering/rhi/shader_reflect.h"
#include "core/rendering/rhi/types.h"
#include "core/rendering/rhi/vulkan/vk_common.h"
#include "core/rendering/rhi/vulkan/vk_upload.h"

#include <vk_mem_alloc.h>

#include <span>

namespace nxe::rhi {

class VulkanDevice;

struct DeviceDesc {
  nx::string_view application_name = "nx2d";
  u32 application_version = 0;
#if NX_DEBUG
  bool enable_validation = true;
  bool enable_sync_validation = true;
#else
  bool enable_validation = false;
  bool enable_sync_validation = false;
#endif
  bool break_on_validation_error = false;
  nx::string_view pipeline_cache_path;
  u32 frames_in_flight = 2;
};

/// Which queue family owns an image, as far as the uploader's dedicated-
/// transfer path is concerned.
enum class ImageOwner : u8 {
  Unknown,
  Copy,
  Graphics,
};

struct PendingTextureView {
  TextureHandle texture;
  VkImageView view = VK_NULL_HANDLE;
  u64 publish_after = 0;
  u32 base_mip = 0;
};

/// A sampled view spans the resident suffix of the mip chain; attachment and
/// storage views address exactly one mip.
struct VulkanTexture {
  VkImage image = VK_NULL_HANDLE;
  VkImageView srv = VK_NULL_HANDLE;
  VkImageView rtv = VK_NULL_HANDLE;
  nx::small_vector<VkImageView, 4> retired_srvs;
  nx::small_vector<VkImageView, 12> uavs;
  nx::small_vector<u32, 12> uav_indices;
  VmaAllocation allocation = VK_NULL_HANDLE;
  Format format = Format::Unknown;
  TextureDimension dimension = TextureDimension::Tex2D;
  u32 width = 0;
  u32 height = 0;
  u32 depth = 1;
  u32 mip_levels = 1;
  u32 sampled_base_mip = 0;
  u32 layers = 1;
  TextureUsageFlags usage = TextureUsage::None;
  ResourceState state = ResourceState::Undefined;
  ImageOwner owner = ImageOwner::Unknown;
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

struct VulkanQueryPool {
  VkQueryPool pool = VK_NULL_HANDLE;
  u32 count = 0;
};

struct VulkanHeap {
  VmaAllocation allocation = VK_NULL_HANDLE;
  u64 size = 0;
};

struct VulkanShader {
  VkShaderModule module = VK_NULL_HANDLE;
  struct ComputeEntry {
    nx::string name;
    u32 local_size[3] = {};
  };
  nx::small_vector<ComputeEntry, 4> compute_entries;
  ShaderReflection reflection;
};

struct VulkanPipeline {
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkPipelineBindPoint bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
};

struct VulkanGraphicsPipelineCompileJob;
struct VulkanComputePipelineCompileJob;
class VulkanCompiledPipeline;

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

struct DeviceRoutes {
  bool core_12 = false;
  bool core_13 = false;
  bool draw_indirect_count = false;
  bool storage_image_extended_formats = false;
  /// Empty when the device can do everything the engine needs, and the first
  /// thing it could not when it cannot.
  nx::string_view missing;
};

class VulkanCommandContext {
public:
  VulkanCommandContext() = default;

  [[nodiscard]] VkCommandBuffer handle() const noexcept { return m_cmd; }
  [[nodiscard]] bool valid() const noexcept { return m_cmd != VK_NULL_HANDLE; }

  void begin_render_pass(const RenderPassDesc &desc);
  void end_render_pass();

  void begin_compute_pass(nx::string_view name = "compute");
  void end_compute_pass();

  void set_viewport(const Viewport &viewport);
  void set_scissor(const Rect2D &rect);

  /// Clears @p count slots. Every slot written must be reset first, and a reset
  /// is invalid inside a render pass - so this belongs at the top of a frame.
  void reset_queries(QueryPoolHandle pool, u32 first, u32 count);
  void write_timestamp(QueryPoolHandle pool, u32 index);

  void bind_pipeline(PipelineHandle pipeline);
  /// Sets the whole 128-byte push block. Offset/size in bytes.
  void push_constants(const void *data, u32 size, u32 offset = 0);

  void push_constants(const BlockWriter &block) {
    push_constants(block.bytes().data(), nx::cast<u32>(block.bytes().size()));
  }

  /// The only buffer binding in the API: everything else a shader reads is
  /// reached bindlessly by device address. An index buffer cannot be, because
  /// the index fetch is fixed-function on both backends.
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
  friend class VulkanDevice;

  VkCommandBuffer m_cmd = VK_NULL_HANDLE;
  VulkanDevice *m_device = nullptr;
  bool m_in_render_pass = false;
};

struct CommandSlot {
  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
};

struct FrameContext {
  CommandSlot slot;
  VkSemaphore acquire_semaphore = VK_NULL_HANDLE;
  u64 timeline_value = 0;
};

/// A slot_map the render and main threads share. The render thread reads
/// resources (barriers, upload recording, present) while the main thread
/// creates and evicts them, so the raw slot_map's dense vector faults when an
/// insert reallocs or an erase swap-pops under a concurrent try_get. Elements
/// live behind a stable heap pointer, so a try_get result outlives any later
/// insert/erase of *other* entries; the mutex only guards the lookup itself.
/// An entry in active use this frame is never the one erased - the deferred
/// delete queue holds it for frames after its last reference drops.
template <typename T, typename Handle> class SyncSlotTable {
public:
  [[nodiscard]] Handle insert(T &&value) {
    nx::scoped_lock lock(m_mutex);
    return m_slots.insert(nx::make_unique<T>(std::move(value)));
  }

  [[nodiscard]] T *try_get(Handle h) noexcept {
    nx::scoped_lock lock(m_mutex);
    nx::unique_ptr<T> *slot = m_slots.try_get(h);
    return slot ? slot->get() : nullptr;
  }

  [[nodiscard]] const T *try_get(Handle h) const noexcept {
    nx::scoped_lock lock(m_mutex);
    const nx::unique_ptr<T> *slot = m_slots.try_get(h);
    return slot ? slot->get() : nullptr;
  }

  [[nodiscard]] bool contains(Handle h) const noexcept {
    nx::scoped_lock lock(m_mutex);
    return m_slots.contains(h);
  }

  bool erase(Handle h) {
    nx::scoped_lock lock(m_mutex);
    return m_slots.erase(h);
  }

  void clear() noexcept {
    nx::scoped_lock lock(m_mutex);
    m_slots.clear();
  }

  template <typename Fn> void for_each(Fn &&fn) {
    nx::scoped_lock lock(m_mutex);
    for (nx::unique_ptr<T> &slot : m_slots)
      fn(*slot);
  }

private:
  mutable nx::mutex m_mutex;
  nx::slot_map<nx::unique_ptr<T>, Handle> m_slots;
};

class VulkanDevice {
public:
  VulkanDevice() = default;
  ~VulkanDevice();

  [[nodiscard]] static DeviceRoutes
  probe_routes(VkPhysicalDevice candidate,
               const VkPhysicalDeviceProperties &props);

  VulkanDevice(const VulkanDevice &) = delete;
  VulkanDevice &operator=(const VulkanDevice &) = delete;

  [[nodiscard]] bool init(const DeviceDesc &desc);
  void shutdown() noexcept;

  [[nodiscard]] const DeviceCaps &caps() const noexcept { return m_caps; }

  [[nodiscard]] u32 frames_in_flight() const noexcept {
    return m_frames_in_flight;
  }

  [[nodiscard]] VulkanUploader &uploader() noexcept { return m_uploader; }

  /// a caller with no swapchain - a test, an asset tool - otherwise has no way
  /// to reach the graphics queue at all, and its uploads stay stranded on the
  /// transfer family.
  void flush_uploads() noexcept;

  [[nodiscard]] bool create_swapchain(const SwapchainDesc &desc);
  [[nodiscard]] bool resize_swapchain(u32 width, u32 height);
  void destroy_swapchain() noexcept;

  [[nodiscard]] bool supports(Format format, TextureUsageFlags usage) const;

  [[nodiscard]] Format swapchain_format() const noexcept {
    return m_swapchain_format;
  }
  [[nodiscard]] Extent2D swapchain_extent() const noexcept {
    return {m_swapchain_extent.width, m_swapchain_extent.height};
  }

  [[nodiscard]] AcquireResult begin_frame(VulkanCommandContext &out_context);
  [[nodiscard]] bool end_frame();

  [[nodiscard]] bool begin_headless_frame(VulkanCommandContext &out_context);

  [[nodiscard]] bool end_headless_frame();

  [[nodiscard]] TextureHandle current_backbuffer() const noexcept {
    return m_backbuffers.empty() ? TextureHandle{}
                                 : m_backbuffers[m_image_index];
  }

  void wait_idle() noexcept;

  [[nodiscard]] TextureHandle create_texture(const TextureDesc &desc);
  void destroy_texture(TextureHandle handle) noexcept;
  [[nodiscard]] bool replace_texture(TextureHandle target,
                                     TextureHandle replacement);
  [[nodiscard]] bool set_texture_view_base_mip(TextureHandle handle,
                                               u32 base_mip);
  /// True when the device can sample a texture whose finer mips are not yet
  /// uploaded (via TextureDesc::sampled_base_mip and subsequent view updates).
  [[nodiscard]] bool supports_mip_streaming() const noexcept { return true; }
  [[nodiscard]] const VulkanTexture *
  texture(TextureHandle handle) const noexcept;
  [[nodiscard]] TextureUsageFlags
  texture_usage(TextureHandle handle) const noexcept;

  [[nodiscard]] BufferHandle create_buffer(const BufferDesc &desc);
  void destroy_buffer(BufferHandle handle) noexcept;
  [[nodiscard]] const VulkanBuffer *buffer(BufferHandle handle) const noexcept;

  [[nodiscard]] SamplerHandle create_sampler(const SamplerDesc &desc);
  void destroy_sampler(SamplerHandle handle) noexcept;

  [[nodiscard]] QueryPoolHandle create_timestamp_pool(u32 count);
  void destroy_query_pool(QueryPoolHandle handle) noexcept;
  [[nodiscard]] bool read_timestamps(QueryPoolHandle handle,
                                     std::span<u64> out) const;
  [[nodiscard]] const VulkanQueryPool *
  query_pool(QueryPoolHandle handle) const noexcept {
    return m_query_pools.try_get(handle);
  }

  // Memory aliasing Two transients whose lifetimes do not overlap can share
  // bytes. The render graph knows those lifetimes, so it does the packing: it
  // asks what each texture needs, creates one heap, and places them at computed
  // offsets.

  [[nodiscard]] MemoryRequirements
  texture_requirements(const TextureDesc &desc) const;
  [[nodiscard]] HeapHandle create_heap(const HeapDesc &desc);
  void destroy_heap(HeapHandle handle) noexcept;
  /// The offset must satisfy the alignment from texture_requirements().
  [[nodiscard]] TextureHandle
  create_texture_in_heap(const TextureDesc &desc, HeapHandle heap, u64 offset);

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

  [[nodiscard]] u32 texture_index(TextureHandle handle) const noexcept;
  [[nodiscard]] u32 rw_texture_index(TextureHandle handle,
                                     u32 mip = 0) const noexcept;
  [[nodiscard]] u32 sampler_index(SamplerHandle handle) const noexcept;
  [[nodiscard]] VkDeviceAddress
  buffer_address(BufferHandle handle) const noexcept;

  [[nodiscard]] void *buffer_mapped(BufferHandle handle) const noexcept;
  void invalidate_buffer(BufferHandle handle, u64 offset = 0,
                         u64 size = ~0ull) noexcept;

  [[nodiscard]] bool video_decode_available() const noexcept {
    return m_video_decode.available;
  }
  [[nodiscard]] VkPhysicalDevice vk_physical_device() const noexcept {
    return m_physical_device;
  }
  [[nodiscard]] u32 video_decode_family() const noexcept {
    return m_video_decode.family;
  }
  [[nodiscard]] VkQueue video_decode_queue() const noexcept {
    return m_video_decode.queue;
  }
  [[nodiscard]] VkQueue graphics_queue() const noexcept {
    return m_queues[nx::cast<usize>(QueueType::Graphics)];
  }
  [[nodiscard]] u32 graphics_family() const noexcept {
    return m_queue_families[nx::cast<usize>(QueueType::Graphics)];
  }

  [[nodiscard]] VkDevice vk_device() const noexcept { return m_device; }
  [[nodiscard]] VulkanTexture *texture_mut(TextureHandle handle) noexcept;

  [[nodiscard]] const ShaderReflection &
  reflection(ShaderHandle handle) const noexcept;

  [[nodiscard]] ResourceState
  texture_state(TextureHandle handle) const noexcept;
  void set_texture_state(TextureHandle handle, ResourceState state) noexcept;
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
  friend class VulkanUploader;

  [[nodiscard]] bool
  prepare_graphics_pipeline(const GraphicsPipelineDesc &desc,
                            VulkanGraphicsPipelineCompileJob &out) const;
  [[nodiscard]] bool
  prepare_compute_pipeline(const ComputePipelineDesc &desc,
                           VulkanComputePipelineCompileJob &out) const;
  [[nodiscard]] VulkanCompiledPipeline
  compile_graphics_pipeline(const VulkanGraphicsPipelineCompileJob &job) const;
  [[nodiscard]] VulkanCompiledPipeline
  compile_compute_pipeline(const VulkanComputePipelineCompileJob &job) const;

  bool create_instance(const DeviceDesc &desc);
  bool select_physical_device();
  bool create_logical_device();
  [[nodiscard]] u32 find_video_decode_family() const;
  void route_promoted_entry_points() const noexcept;
  bool create_allocator();
  bool create_bindless_heap();
  bool create_pipeline_cache(nx::string_view path);
  void save_pipeline_cache() noexcept;
  bool create_frame_contexts(u32 frames_in_flight);

  bool m_headless_frame = false;
  bool build_swapchain(u32 width, u32 height);
  bool create_texture_views(VulkanTexture &texture, nx::string_view name);
  void destroy_backbuffer_views() noexcept;
  void set_object_name(u64 object, VkObjectType type, nx::string_view name);
  void collect_garbage(u64 completed_timeline_value) noexcept;
  void destroy_texture_now(VulkanTexture &texture) noexcept;

  void write_texture_descriptor(u32 index, VkImageView view) noexcept;
  void write_rw_texture_descriptor(u32 index, VkImageView view) noexcept;
  void write_sampler_descriptor(u32 index, VkSampler sampler) noexcept;
  [[nodiscard]] u32 alloc_rw_slot() noexcept;
  void free_rw_slot(u32 slot) noexcept;

  VkInstance m_instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT m_messenger = VK_NULL_HANDLE;
  VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
  DeviceRoutes m_routes;
  VkDevice m_device = VK_NULL_HANDLE;
  VmaAllocator m_allocator = VK_NULL_HANDLE;
  DeviceCaps m_caps;
  nx::string m_device_name;
  bool m_debug_utils = false;
  bool m_break_on_error = false;
  bool m_sync_validation = false;

  VkQueue m_queues[nx::cast<usize>(QueueType::Count)] = {};
  u32 m_queue_families[nx::cast<usize>(QueueType::Count)] = {};

  struct VideoDecode {
    bool available = false;
    u32 family = ~0u;
    VkQueue queue = VK_NULL_HANDLE;
  };
  VideoDecode m_video_decode;

  BindlessHeap m_bindless;
  VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
  VkPipelineCache m_pipeline_cache = VK_NULL_HANDLE;
  mutable nx::mutex m_pipeline_cache_lock;
  nx::workload_tracker m_pipeline_compile_workload;
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

  SyncSlotTable<VulkanTexture, TextureHandle> m_textures;
  nx::slot_map<VulkanBuffer, BufferHandle> m_buffers;
  nx::slot_map<VulkanSampler, SamplerHandle> m_samplers;
  nx::slot_map<VulkanQueryPool, QueryPoolHandle> m_query_pools;
  nx::slot_map<VulkanHeap, HeapHandle> m_heaps;
  nx::slot_map<VulkanShader, ShaderHandle> m_shaders;
  nx::slot_map<VulkanPipeline, PipelineHandle> m_pipelines;

  VulkanUploader m_uploader;
  /// Copy-timeline value the current frame's graphics submit must wait on so
  /// that resources uploaded before the frame are ready; 0 when none pending.
  u64 m_frame_upload_wait = 0;

  struct PendingDelete {
    u64 timeline_value = 0;
    TextureHandle texture;
    BufferHandle buffer;
    SamplerHandle sampler;
    PipelineHandle pipeline;
  };
  nx::small_vector<PendingDelete, 32> m_pending_deletes;
  nx::small_vector<PendingTextureView, 16> m_pending_texture_views;
};

} // namespace nxe::rhi
