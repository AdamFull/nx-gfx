#pragma once

#include "core/foundation/containers/fixed_vector.h"
#include "core/foundation/containers/slot_map.h"
#include "core/foundation/containers/small_vector.h"
#include "core/foundation/containers/sync_slot_map.h"
#include "core/foundation/strings/utf8_string.h"
#include "core/foundation/threading/sync.h"
#include "core/foundation/threading/thread_pool.h"
#include "core/foundation/threading/workload.h"
#include "rendering/rhi/bindless.h"
#include "rendering/rhi/descs.h"
#include "rendering/rhi/handles.h"
#include "rendering/rhi/native.h"
#include "rendering/rhi/native_driver.h"
#include "rendering/rhi/pipeline_compiler.h"
#include "rendering/rhi/shader_reflect.h"
#include "rendering/rhi/types.h"
#include "rendering/rhi/vulkan/vk_common.h"
#include "rendering/rhi/vulkan/vk_gpu_crash.h"
#include "rendering/rhi/vulkan/vk_gpu_profiler.h"
#include "rendering/rhi/vulkan/vk_mem_alloc_include.h"
#include "rendering/rhi/vulkan/vk_upload.h"
#if defined(VK_USE_PLATFORM_WIN32_KHR)
#include "rendering/rhi/vulkan/vk_composition_win32.h"
#endif

#include <span>

namespace nxe::rhi {

class VulkanDevice;

struct DeviceDesc {
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
  GpuCrashMode gpu_crash_mode = GpuCrashMode::Basic;
  nx::string_view pipeline_cache_path;
  NativeDriverView native_driver;
  u32 frames_in_flight = 2;
  /// Asked what to enable before the device is made; only during init().
  std::span<NativeDeviceClient *const> native_clients{};
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
  nx::small_vector<nx::string, 4> entry_points;
  ShaderReflection reflection;
};

struct VulkanPipeline {
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkPipelineBindPoint bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
  /// Dynamic in every graphics pipeline; binding one restores its own.
  VkCullModeFlags cull = VK_CULL_MODE_NONE;
  VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
};

struct VulkanGraphicsPipelineCompileJob;
struct VulkanComputePipelineCompileJob;
class VulkanCompiledPipeline;

struct BindlessHeap {
  VkDescriptorPool pool = VK_NULL_HANDLE;
  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  VkDescriptorSet set = VK_NULL_HANDLE;
  /// The globals uniform, followed by the null page its address names.
  VkBuffer globals = VK_NULL_HANDLE;
  VmaAllocation globals_allocation = nullptr;
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
  /// Optional features 3D uses, enabled when the device has them.
  bool mesh_shaders = false;
  bool buffer_int64_atomics = false;
  bool image_int64_atomics = false;
  bool shader_float16 = false;
  bool fragment_barycentrics = false;
  bool sample_rate_shading = false;
  bool fragment_stores = false;
  bool subgroup_size_control = false;
  bool acceleration_structures = false;
  bool ray_queries = false;
  VulkanFaultRoute device_fault = VulkanFaultRoute::None;
  bool device_fault_vendor_binary = false;
  bool address_binding_report = false;
  /// Only taken up by a profiling build, where Tracy calibrates against these
  /// and resets the copy queue's queries from the host.
  VulkanCalibrationRoute calibrated_timestamps = VulkanCalibrationRoute::None;
  bool host_query_reset = false;
  /// Win32 memory and semaphore import, for presenting through
  /// DirectComposition when the surface cannot composite alpha.
  bool win32_interop = false;
  /// Empty when the device can do everything the engine needs.
  nx::small_vector<nx::string_view, 16> missing;
};

struct CommandSlot;

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
  /// Override the bound graphics pipeline's own until the next bind.
  void set_cull_mode(CullMode cull);
  void set_front_face(FrontFace face);

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
  /// DrawMeshTasksIndirectCommand records, for a mesh pipeline; nothing where
  /// the device has no mesh shaders.
  void draw_mesh_tasks_indirect(BufferHandle buffer, u64 offset = 0,
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
  /// Copies @p size bytes of @p source from @p source_offset into
  /// @p destination at @p destination_offset, outside a render/compute pass,
  /// and makes them visible to the host. The source must be CopySrc, the
  /// destination CopyDst, both in usage and state.
  [[nodiscard]] bool copy_buffer(BufferHandle source, u64 source_offset,
                                 BufferHandle destination,
                                 u64 destination_offset, u64 size);

  void push_debug_group(nx::string_view name, f32 r = 1.f, f32 g = 1.f,
                        f32 b = 1.f);
  void pop_debug_group();

  /// The command buffer itself (VkCommandBuffer), for a vendor library to
  /// record into between here and end_native(); outside a render pass.
  [[nodiscard]] void *begin_native() noexcept;
  /// Takes the commands back: what the library bound in their place is
  /// bound again.
  void end_native() noexcept;

private:
  friend class VulkanDevice;

  VkCommandBuffer m_cmd = VK_NULL_HANDLE;
  VulkanDevice *m_device = nullptr;
  VulkanGpuProfiler::Zones *m_zones = nullptr;
  /// Set by begin_recording: the pool this context holds, and the place in
  /// the frame's submission its command buffer fills.
  CommandSlot *m_slot = nullptr;
  u32 m_position = 0;
  bool m_in_render_pass = false;
};

/// A command pool and the command buffers allocated from it. One recorder uses
/// it at a time; it is reset when its frame slot comes round again.
struct CommandSlot {
  VkCommandPool pool = VK_NULL_HANDLE;
  nx::small_vector<VkCommandBuffer, 2> cmds;
  u32 used = 0;
  VulkanGpuProfiler::Zones zones;
};

struct FrameContext {
  /// What the frame's own context records into.
  CommandSlot slot;
  /// One for each recording split off the frame that ran at the same time.
  nx::small_vector<nx::unique_ptr<CommandSlot>, 4> recorders;
  nx::small_vector<CommandSlot *, 4> idle;
  /// Every command buffer the frame submits, in submission order.
  nx::small_vector<VkCommandBuffer, 8> order;
  u64 timeline_value = 0;
};

/// One window's swapchain: a surface's, one presented through
/// DirectComposition, or images of the device's own that nothing presents.
struct VulkanSwapchain {
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  VkExtent2D extent = {};
  VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
  SurfaceAlpha requested_alpha = SurfaceAlpha::Opaque;
  SurfaceAlpha alpha = SurfaceAlpha::Opaque;
#if defined(VK_USE_PLATFORM_WIN32_KHR)
  // Set instead of swapchain when frames reach the screen through
  // DirectComposition. The backbuffers are imported images of its own.
  nx::unique_ptr<CompositionPresenter> composition;
  VkSemaphore composed_rendered = VK_NULL_HANDLE;
  VkSemaphore composed_copied = VK_NULL_HANDLE;
  nx::small_vector<VkDeviceMemory, 4> composed_memory;
  /// The frame that last rendered each image; its copy must finish before the
  /// image is written again.
  nx::small_vector<u64, 4> composed_serials;
#endif
  nx::small_vector<TextureHandle, 4> backbuffers;
  /// One per image, signalled when the frame that drew it is done.
  nx::small_vector<VkSemaphore, 4> release_semaphores;
  /// One per frame slot, signalled when the image acquired into it is ready.
  nx::small_vector<VkSemaphore, 3> acquire_semaphores;
  u32 image_index = 0;
  /// The backbuffers are images this device made rather than a surface's, so
  /// they are destroyed as textures and neither acquired nor presented.
  bool offscreen = false;
  /// Acquired by the frame being recorded.
  bool acquired = false;
  /// The last present found the surface changed under it.
  bool stale = false;

  [[nodiscard]] bool composed() const noexcept {
#if defined(VK_USE_PLATFORM_WIN32_KHR)
    return composition != nullptr;
#else
    return false;
#endif
  }
  [[nodiscard]] bool presents() const noexcept {
    return swapchain != VK_NULL_HANDLE;
  }
};

class VulkanDevice {
public:
  VulkanDevice() = default;
  ~VulkanDevice();

  [[nodiscard]] static DeviceRoutes
  probe_routes(VkPhysicalDevice candidate,
               const VkPhysicalDeviceProperties &props,
               u32 instance_api_version);

  VulkanDevice(const VulkanDevice &) = delete;
  VulkanDevice &operator=(const VulkanDevice &) = delete;

  [[nodiscard]] bool init(const DeviceDesc &desc);
  void shutdown() noexcept;

  [[nodiscard]] nx::string_view init_error() const noexcept {
    return m_init_error;
  }
  [[nodiscard]] const DeviceInitReport &init_report() const noexcept {
    return m_init_report;
  }

  [[nodiscard]] const DeviceCaps &caps() const noexcept { return m_caps; }
  [[nodiscard]] VkStateInfo
  state_info(const ResourceState state) const noexcept {
    VkPipelineStageFlags2 shaders = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    // Naming a stage of a feature the device did not enable is invalid.
    if (m_caps.mesh_shaders)
      shaders |= VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
                 VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
    return to_vk_state(state, shaders);
  }
  [[nodiscard]] GpuDeviceStatus device_status() const noexcept {
    return m_gpu_crash.status();
  }

  [[nodiscard]] u32 frames_in_flight() const noexcept {
    return m_frames_in_flight;
  }

  [[nodiscard]] VulkanUploader &uploader() noexcept { return m_uploader; }

  /// a caller with no swapchain - a test, an asset tool - otherwise has no way
  /// to reach the graphics queue at all, and its uploads stay stranded on the
  /// transfer family.
  void flush_uploads() noexcept;

  /// A swapchain for one window; a device has one per window it draws. Every
  /// swapchain shares swapchain_format(), so a surface that cannot offer it
  /// while others exist is refused. Invalid when it cannot be made.
  [[nodiscard]] SwapchainHandle create_swapchain(const SwapchainDesc &desc);
  /// A swapchain with no surface: images of its own that nothing presents.
  /// Everything that reads swapchain_format(), swapchain_extent() or
  /// current_backbuffer() sees a normal swapchain, so a renderer draws a whole
  /// frame without a window behind it; acquire() and end_frame() drop only
  /// the acquire and the present.
  [[nodiscard]] SwapchainHandle
  create_offscreen_swapchain(u32 width, u32 height, Format format);
  [[nodiscard]] bool resize_swapchain(SwapchainHandle swapchain, u32 width,
                                      u32 height);
  /// Waits for the GPU first; not between acquire() and end_frame().
  void destroy_swapchain(SwapchainHandle swapchain) noexcept;

  /// Whether @p swapchain is one with no surface behind it.
  [[nodiscard]] bool offscreen_swapchain(SwapchainHandle swapchain) const;

  [[nodiscard]] bool supports(Format format, TextureUsageFlags usage) const;

  /// The format every swapchain has: chosen by the first one made while none
  /// exists, and kept after it goes, so pipelines built for it stay valid.
  [[nodiscard]] Format swapchain_format() const noexcept {
    return m_swapchain_format;
  }
  [[nodiscard]] Extent2D swapchain_extent(SwapchainHandle swapchain) const;
  [[nodiscard]] SurfaceAlpha swapchain_alpha(SwapchainHandle swapchain) const;
  /// The last present found the surface no longer matches: resize it.
  [[nodiscard]] bool swapchain_stale(SwapchainHandle swapchain) const;

  /// Starts GPU frame N: waits for its slot, then flushes uploads. False when
  /// the device is lost. Swapchains are acquired separately.
  [[nodiscard]] bool begin_frame(VulkanCommandContext &out_context);
  /// Takes @p swapchain's next image for this frame, once per frame. A
  /// swapchain that is not acquired is neither drawn nor presented this
  /// frame; the others still are.
  [[nodiscard]] AcquireResult acquire(SwapchainHandle swapchain);
  /// Submits the frame and presents every swapchain it acquired, together.
  /// False when the submission failed; a swapchain its present found out of
  /// date is marked stale instead.
  [[nodiscard]] bool end_frame();

  /// Ends the frame context's command buffer and reserves @p count more after
  /// it, to be recorded apart; the context then continues in a new one, and
  /// must not record until every split recording has ended. An empty split
  /// means none was made.
  [[nodiscard]] RecordingSplit split_frame(VulkanCommandContext &frame,
                                           u32 count);
  /// Begins recording @p index of @p split with a command pool of its own. Any
  /// thread or job may record it, one at a time, and must end it before the
  /// frame ends.
  [[nodiscard]] bool begin_recording(RecordingSplit split, u32 index,
                                     VulkanCommandContext &out);
  void end_recording(VulkanCommandContext &recording);

  [[nodiscard]] bool begin_headless_frame(VulkanCommandContext &out_context);

  [[nodiscard]] bool end_headless_frame();

  /// Ends and submits the open frame with no present and no swapchain
  /// semaphores, then waits for it. Shared by the headless frame and the
  /// offscreen swapchain, which differ only in whether a backbuffer exists.
  [[nodiscard]] bool submit_without_present();

  /// The image @p swapchain acquired for this frame.
  [[nodiscard]] TextureHandle
  current_backbuffer(SwapchainHandle swapchain) const noexcept;

  /// Waits for the GPU, then frees every retired resource. Never while
  /// another thread records or submits a frame: that frame may hold what this
  /// frees (an engine waits for its render job first, Engine::wait_gpu_idle).
  void wait_idle() noexcept;
  /// Waits for the GPU to finish the frames submitted so far, freeing nothing
  /// and touching no queue, so the render job may call it while it records a
  /// frame.
  void wait_submitted() noexcept;
  /// Writes the driver's pipeline cache now, from any thread, while compiles
  /// continue. False when nothing was written.
  bool save_pipeline_cache() noexcept;

  /// Frame work up to @p serial may name resources and has not reached this
  /// device: whatever is destroyed from now on outlives it, however idle the
  /// GPU is, until release_retirements() reaches @p serial. For a producer
  /// that runs ahead of rendering; without one, a deletion waits only for the
  /// work already submitted.
  void hold_retirements(u64 serial) noexcept;
  /// Frame work up to @p serial has been submitted, or dropped.
  void release_retirements(u64 serial) noexcept;

  [[nodiscard]] TextureHandle create_texture(const TextureDesc &desc);
  void destroy_texture(TextureHandle handle) noexcept;
  /// Gives @p target @p replacement's image under its own handle and
  /// bindless index, and takes ownership of @p replacement. Under a hold the
  /// frame work before the held one may still be recording against the old
  /// image, so the swap waits for the first frame begun after it is released.
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

  /// Copies every input of @p desc on the owner thread into a job to compile
  /// anywhere (PipelineJob::compile, compile_pipeline_async) and publish here.
  /// Invalid when @p desc is refused.
  [[nodiscard]] PipelineJob prepare_pipeline(const GraphicsPipelineDesc &desc);
  [[nodiscard]] PipelineJob prepare_pipeline(const ComputePipelineDesc &desc);
  /// Invalid when @p job did not compile.
  [[nodiscard]] PipelineHandle publish_pipeline(PipelineJob &&job);
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
  /// Makes CPU writes to a mapped buffer visible to the GPU. This is a no-op
  /// for host-coherent allocations.
  void flush_buffer(BufferHandle handle, u64 offset = 0,
                    u64 size = ~0ull) noexcept;
  void invalidate_buffer(BufferHandle handle, u64 offset = 0,
                         u64 size = ~0ull) noexcept;

  [[nodiscard]] VkPhysicalDevice vk_physical_device() const noexcept {
    return m_physical_device;
  }
  [[nodiscard]] VkQueue graphics_queue() const noexcept {
    return m_queues[nx::cast<usize>(QueueType::Graphics)];
  }
  [[nodiscard]] u32 graphics_family() const noexcept {
    return m_queue_families[nx::cast<usize>(QueueType::Graphics)];
  }

  [[nodiscard]] VkDevice vk_device() const noexcept { return m_device; }

  /// The device's own objects (see native.h).
  [[nodiscard]] NativeDevice native() const noexcept;
  /// @p handle's own objects; empty where it is not live.
  [[nodiscard]] NativeTexture
  native_texture(TextureHandle handle) const noexcept;
  /// The VkImageLayout a texture the graph left in @p state is in.
  [[nodiscard]] u32 native_layout(ResourceState state) const noexcept;
  /// Whether the instance or the device was made with extension @p name.
  [[nodiscard]] bool native_extension(nx::string_view name) const noexcept;
  /// The queue made for @p client (NativeDeviceClient::queue_family), or none.
  [[nodiscard]] NativeQueue
  native_queue(const NativeDeviceClient &client) const noexcept;
  [[nodiscard]] VulkanTexture *texture_mut(TextureHandle handle) noexcept;

  [[nodiscard]] const ShaderReflection &
  reflection(ShaderHandle handle) const noexcept;

  [[nodiscard]] ResourceState
  texture_state(TextureHandle handle) const noexcept;
  void set_texture_state(TextureHandle handle, ResourceState state) noexcept;
  [[nodiscard]] bool debug_utils_enabled() const noexcept {
    return m_debug_utils;
  }
  [[nodiscard]] VulkanGpuProfiler &gpu_profiler() noexcept {
    return m_gpu_profiler;
  }
  [[nodiscard]] VulkanGpuProfiler &copy_profiler() noexcept {
    return m_copy_profiler;
  }
  [[nodiscard]] VkPipelineLayout pipeline_layout() const noexcept {
    return m_pipeline_layout;
  }
  [[nodiscard]] VkDescriptorSet bindless_set() const noexcept {
    return m_bindless.set;
  }
  [[nodiscard]] const VulkanPipeline *
  pipeline(PipelineHandle handle) const noexcept;

  /// Backend calls returning DEVICE_LOST pass through this owner so one
  /// coherent report is collected before the device is torn down.
  VkResult check_result(VkResult result, const char *call, const char *file,
                        i32 line) const noexcept;

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
  /// PipelineJob's compile step, on a VulkanPipelineJob payload.
  static bool compile_job(void *payload);

  bool create_instance(const DeviceDesc &desc);
  bool rebuild_debug_messenger(bool address_bindings) noexcept;
  bool select_physical_device();
  bool create_logical_device(std::span<NativeDeviceClient *const> clients);
  [[nodiscard]] bool route_promoted_entry_points() noexcept;
  bool create_allocator();
  bool create_bindless_heap();
  bool create_default_sampler();
  bool create_bindless_globals();
  bool create_pipeline_cache(nx::string_view path);
  bool create_frame_contexts(u32 frames_in_flight);

  bool m_headless_frame = false;
  /// What begin_frame and begin_headless_frame share: the slot wait, garbage,
  /// the slot's command buffer and the upload batch.
  [[nodiscard]] bool open_frame(VulkanCommandContext &out_context);
  [[nodiscard]] bool build_swapchain(VulkanSwapchain &sc, u32 width,
                                     u32 height);
  [[nodiscard]] bool create_acquire_semaphores(VulkanSwapchain &sc);
  [[nodiscard]] bool make_offscreen_images(VulkanSwapchain &sc, u32 width,
                                           u32 height);
  /// Everything @p sc holds; the GPU must be done with it.
  void release_swapchain(VulkanSwapchain &sc) noexcept;
  /// Moves the acquired image to where its presenter takes it from.
  void finish_backbuffer(VulkanSwapchain &sc, VkCommandBuffer cmd) noexcept;
#if defined(VK_USE_PLATFORM_WIN32_KHR)
  [[nodiscard]] bool create_composed_swapchain(VulkanSwapchain &sc,
                                               const SwapchainDesc &desc);
  [[nodiscard]] bool import_composed_images(VulkanSwapchain &sc, u32 width,
                                            u32 height);
  void release_composed_images(VulkanSwapchain &sc) noexcept;
  /// Everything DirectComposition gave @p sc, leaving its surface.
  void release_composition(VulkanSwapchain &sc) noexcept;
  void acquire_composed_image(VulkanSwapchain &sc,
                              VkCommandBuffer cmd) noexcept;
#endif
  bool create_texture_views(VulkanTexture &texture, nx::string_view name);
  void destroy_backbuffer_views(VulkanSwapchain &sc) noexcept;
  void set_object_name(u64 object, VkObjectType type, nx::string_view name);
  void collect_garbage(u64 completed_timeline_value) noexcept;
  [[nodiscard]] VkCommandBuffer next_command_buffer(CommandSlot &slot);
  [[nodiscard]] VkCommandBuffer open_frame_slot(FrameContext &frame);
  [[nodiscard]] bool
  close_frame(FrameContext &frame,
              nx::small_vector<VkCommandBufferSubmitInfo, 8> &out);
  [[nodiscard]] VkResult wait_timeline(u64 value) noexcept;
  void destroy_texture_now(VulkanTexture &texture) noexcept;

  void write_texture_descriptor(u32 index, VkImageView view) noexcept;
  void write_rw_texture_descriptor(u32 index, VkImageView view) noexcept;
  void write_sampler_descriptor(u32 index, VkSampler sampler) noexcept;
  [[nodiscard]] u32 alloc_rw_slot() noexcept;
  void free_rw_slot(u32 slot) noexcept;

  VkInstance m_instance = VK_NULL_HANDLE;
  u32 m_instance_api_version = VK_API_VERSION_1_0;
  VkDebugUtilsMessengerEXT m_messenger = VK_NULL_HANDLE;
  VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
  DeviceRoutes m_routes;
  VkDevice m_device = VK_NULL_HANDLE;
  VmaAllocator m_allocator = VK_NULL_HANDLE;
  DeviceCaps m_caps;
  nx::string m_device_name;
  nx::string m_init_error;
  DeviceInitReport m_init_report;
  bool m_debug_utils = false;
  bool m_sync_validation = false;
  /// Every instance and device extension enabled, the clients' among them.
  nx::small_vector<nx::string, 48> m_native_extensions;
  mutable VulkanGpuCrashReporter m_gpu_crash;
  VulkanGpuProfiler m_gpu_profiler;
  VulkanGpuProfiler m_copy_profiler;

  VkQueue m_queues[nx::cast<usize>(QueueType::Count)] = {};
  u32 m_queue_families[nx::cast<usize>(QueueType::Count)] = {};

  struct ClientQueue {
    const NativeDeviceClient *client = nullptr;
    NativeQueue queue;
  };
  nx::small_vector<ClientQueue, 2> m_client_queues;

  BindlessHeap m_bindless;
  VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
  VkPipelineCache m_pipeline_cache = VK_NULL_HANDLE;
  mutable nx::mutex m_pipeline_cache_lock;
  nx::workload_tracker m_pipeline_compile_workload{"pipeline::compile"};
  nx::string m_pipeline_cache_path;

  VkSemaphore m_timeline = VK_NULL_HANDLE;
  u64 m_timeline_value = 0;
  std::atomic<u64> m_report_frame{0};
  nx::small_vector<FrameContext, 3> m_frames;
  /// Guards the recorders and idle lists of the frame being recorded.
  nx::mutex m_recording_lock;
  u32 m_frame_index = 0;
  u32 m_frames_in_flight = 2;

  nx::slot_map<VulkanSwapchain, SwapchainHandle> m_swapchains;
  /// The swapchains the frame being recorded acquired, in acquire order.
  nx::small_vector<SwapchainHandle, 4> m_acquired;
  Format m_swapchain_format = Format::Unknown;
  bool m_frame_active = false;

  nx::sync_slot_map<VulkanTexture, TextureHandle> m_textures;
  nx::sync_slot_map<VulkanBuffer, BufferHandle> m_buffers;
  nx::slot_map<VulkanSampler, SamplerHandle> m_samplers;
  nx::slot_map<VulkanQueryPool, QueryPoolHandle> m_query_pools;
  nx::slot_map<VulkanHeap, HeapHandle> m_heaps;
  nx::slot_map<VulkanShader, ShaderHandle> m_shaders;
  nx::sync_slot_map<VulkanPipeline, PipelineHandle> m_pipelines;

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
    HeapHandle heap;
    /// The held frame work it waits on first; 0 once only the timeline is left.
    u64 hold = 0;
  };
  // m_garbage_lock held.
  void retire(PendingDelete pending) noexcept;

  struct PendingReplace {
    TextureHandle target;
    TextureHandle replacement;
    /// The last held frame work that may name the target's old image.
    u64 after = 0;
  };
  // Neither takes m_garbage_lock on entry; each waits for the GPU first.
  bool swap_texture(TextureHandle target, TextureHandle replacement) noexcept;
  void apply_replacements() noexcept;
  // The render job collects garbage while the owner creates and destroys
  // resources: this guards the deferred lists, the handle tables collection
  // erases from, and the texture views it publishes.
  nx::mutex m_garbage_lock;
  nx::small_vector<PendingDelete, 32> m_pending_deletes;
  u64 m_retire_hold = 0;
  u64 m_retire_released = 0;
  nx::small_vector<PendingReplace, 4> m_pending_replaces;
  nx::small_vector<PendingTextureView, 16> m_pending_texture_views;
  // Taken after m_garbage_lock, never before: writes to the one bindless set
  // must be host-synchronized, and both sides make them.
  nx::mutex m_bindless_lock;
};

} // namespace nxe::rhi
