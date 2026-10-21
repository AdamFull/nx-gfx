#pragma once

#include "core/foundation/containers/fixed_vector.h"
#include "core/foundation/containers/slot_map.h"
#include "core/foundation/containers/small_vector.h"
#include "core/foundation/containers/sync_slot_map.h"
#include "core/foundation/strings/utf8_string.h"
#include "core/foundation/threading/sync.h"
#include "core/foundation/threading/thread_pool.h"
#include "core/foundation/threading/workload.h"
#include "nx/gfx/bindless.h"
#include "nx/gfx/gfx.h"
#include "nx/gfx/native.h"
#include "nx/gfx/pipeline_job.h"
#include "nx/gfx/reflection.h"
#include "nx/gfx/types.h"
#include "nx/gfx/vulkan.h"
#include "src/vulkan/vk_common.h"
#include "src/vulkan/vk_gpu_crash.h"
#include "src/vulkan/vk_gpu_profiler.h"
#include "src/vulkan/vk_mem_alloc_include.h"
#include "src/vulkan/vk_render_pass.h"
#include "src/vulkan/vk_upload.h"
#if defined(VK_USE_PLATFORM_WIN32_KHR)
#include "src/vulkan/vk_composition_win32.h"
#endif

#include <span>

namespace nx::gfx::detail {

class VulkanDevice;

/// Which queue family owns an image, as far as the uploader's dedicated-
/// transfer path is concerned.
enum class ImageOwner : u8 {
  Unknown,
  Copy,
  Graphics,
};

struct PendingTextureView {
  gfx::texture_handle texture;
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
  gfx::format format = gfx::format::Unknown;
  gfx::texture_dimension dimension = gfx::texture_dimension::Tex2D;
  u32 width = 0;
  u32 height = 0;
  u32 depth = 1;
  u32 mip_levels = 1;
  u32 sampled_base_mip = 0;
  u32 layers = 1;
  u32 samples = 1;
  gfx::texture_usage_flags usage = gfx::texture_usage::None;
  /// As the image was created; an imageless framebuffer is made for them.
  VkImageUsageFlags image_usage = 0;
  VkImageCreateFlags image_flags = 0;
  gfx::resource_state state = gfx::resource_state::Undefined;
  ImageOwner owner = ImageOwner::Unknown;
  bool owns_image = true;
  /// A wrapped image's owner, called once the device is done with it.
  void (*release)(void *user, VkImage image) noexcept = nullptr;
  void *release_user = nullptr;
};

/// The views create_texture_views makes, as create infos: the sampled view
/// from @p base_mip down, and one storage view per mip. The descriptor-heap
/// route writes slots from these rather than from the views.
[[nodiscard]] VkImageViewCreateInfo
sampled_view_info(const VulkanTexture &texture, u32 base_mip) noexcept;
[[nodiscard]] VkImageViewCreateInfo
storage_view_info(const VulkanTexture &texture, u32 mip) noexcept;

struct VulkanBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  VkDeviceAddress device_address = 0;
  void *mapped = nullptr;
  u64 size = 0;
  gfx::buffer_usage_flags usage = gfx::buffer_usage::None;
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
  gfx::shader_reflection reflection;
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

/// One heap of the descriptor-heap route: a mapped buffer the slots are
/// written into, with the range the implementation reserves at its end.
struct DescriptorHeapBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation allocation = nullptr;
  u8 *data = nullptr;
  VkDeviceAddress address = 0;
  VkDeviceSize size = 0;
  VkDeviceSize reserved_offset = 0;
  VkDeviceSize reserved_size = 0;
};

struct BindlessHeap {
  /// The descriptor-set route.
  VkDescriptorPool pool = VK_NULL_HANDLE;
  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  VkDescriptorSet set = VK_NULL_HANDLE;
  /// The descriptor-heap route: textures, then storage textures, then the
  /// globals uniform in the resource heap; samplers in the sampler heap.
  /// Every pipeline maps the set's bindings onto these offsets.
  DescriptorHeapBuffer resources;
  DescriptorHeapBuffer samplers;
  VkDeviceSize image_stride = 0;
  VkDeviceSize sampler_stride = 0;
  VkDeviceSize rw_offset = 0;
  VkDeviceSize globals_offset = 0;
  VkDescriptorSetAndBindingMappingEXT mappings[4] = {};
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
  /// Passes record through dynamic rendering; otherwise through render passes
  /// over imageless framebuffers.
  bool dynamic_rendering = false;
  bool imageless_framebuffer = false;
  /// VK_ATTACHMENT_STORE_OP_NONE in a render pass: core from 1.3, else the
  /// extension named here.
  bool store_op_none = false;
  const char *store_op_none_extension = nullptr;
  /// Bindless slots in descriptor heaps (with maintenance5, which they need).
  bool descriptor_heap = false;
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

  void begin_render_pass(const gfx::render_pass_desc &desc);
  void end_render_pass();

  void begin_compute_pass(nx::string_view name = "compute");
  void end_compute_pass();

  void set_viewport(const gfx::viewport &viewport);
  void set_scissor(const gfx::rect2d &rect);
  /// Override the bound graphics pipeline's own until the next bind.
  void set_cull_mode(gfx::cull_mode cull);
  void set_front_face(gfx::front_face face);

  /// Clears @p count slots. Every slot written must be reset first, and a reset
  /// is invalid inside a render pass - so this belongs at the top of a frame.
  void reset_queries(gfx::query_pool_handle pool, u32 first, u32 count);
  void write_timestamp(gfx::query_pool_handle pool, u32 index);

  void bind_pipeline(gfx::pipeline_handle pipeline);
  /// Sets the whole 128-byte push block. Offset/size in bytes.
  void push_constants(const void *data, u32 size, u32 offset = 0);

  /// The only buffer binding in the API: everything else a shader reads is
  /// reached bindlessly by device address. An index buffer cannot be, because
  /// the index fetch is fixed-function on both backends.
  void bind_index_buffer(gfx::buffer_handle buffer, gfx::index_type type,
                         u64 offset = 0);

  void draw(u32 vertex_count, u32 instance_count = 1, u32 first_vertex = 0,
            u32 first_instance = 0);
  void draw_indexed(u32 index_count, u32 instance_count = 1,
                    u32 first_index = 0, i32 vertex_offset = 0,
                    u32 first_instance = 0);
  void dispatch(u32 groups_x, u32 groups_y = 1, u32 groups_z = 1);

  void draw_indirect(gfx::buffer_handle buffer, u64 offset = 0,
                     u32 draw_count = 1, u32 stride = 0);
  void draw_indexed_indirect(gfx::buffer_handle buffer, u64 offset = 0,
                             u32 draw_count = 1, u32 stride = 0);
  /// gfx::draw_mesh_tasks_indirect_command records, for a mesh pipeline;
  /// nothing where the device has no mesh shaders.
  void draw_mesh_tasks_indirect(gfx::buffer_handle buffer, u64 offset = 0,
                                u32 draw_count = 1, u32 stride = 0);
  void dispatch_indirect(gfx::buffer_handle buffer, u64 offset = 0);

  void barrier(const gfx::texture_barrier *textures, u32 texture_count,
               const gfx::buffer_barrier *buffers = nullptr,
               u32 buffer_count = 0,
               gfx::state_tracking tracking = gfx::state_tracking::Update);
  void barrier(const gfx::texture_barrier &texture) { barrier(&texture, 1); }

  /// Copies one tightly packed colour mip into a host-readable buffer. The
  /// source must already be in CopySrc and this must be called outside a
  /// render/compute pass. row_pitch may include padding between rows.
  [[nodiscard]] bool copy_texture_to_buffer(gfx::texture_handle texture,
                                            gfx::buffer_handle buffer,
                                            u64 row_pitch, u32 mip = 0,
                                            u32 layer = 0);
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
  gfx::surface_alpha requested_alpha = gfx::surface_alpha::Opaque;
  gfx::surface_alpha alpha = gfx::surface_alpha::Opaque;
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
  nx::small_vector<gfx::texture_handle, 4> backbuffers;
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
  probe_routes(gfx::route_flags disabled, gfx::route_flags enabled,
               VkPhysicalDevice candidate,
               const VkPhysicalDeviceProperties &props,
               u32 instance_api_version);

  VulkanDevice(const VulkanDevice &) = delete;
  VulkanDevice &operator=(const VulkanDevice &) = delete;

  [[nodiscard]] bool init(const gfx::device_desc &desc);
  void shutdown() noexcept;

  [[nodiscard]] nx::string_view init_error() const noexcept {
    return m_init_error;
  }
  [[nodiscard]] const gfx::device_init_report &init_report() const noexcept {
    return m_init_report;
  }

  [[nodiscard]] const gfx::device_caps &caps() const noexcept { return m_caps; }
  [[nodiscard]] VkStateInfo
  state_info(const gfx::resource_state state) const noexcept {
    VkPipelineStageFlags2 shaders = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    // Naming a stage of a feature the device did not enable is invalid.
    if (m_caps.mesh_shaders)
      shaders |= VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
                 VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
    return to_vk_state(state, shaders);
  }
  [[nodiscard]] gfx::gpu_device_status device_status() const noexcept {
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
  [[nodiscard]] gfx::swapchain_handle
  create_swapchain(const gfx::swapchain_desc &desc);
  /// A swapchain with no surface: images of its own that nothing presents.
  /// Everything that reads swapchain_format(), swapchain_extent() or
  /// current_backbuffer() sees a normal swapchain, so a renderer draws a whole
  /// frame without a window behind it; acquire() and end_frame() drop only
  /// the acquire and the present.
  [[nodiscard]] gfx::swapchain_handle
  create_offscreen_swapchain(u32 width, u32 height, gfx::format format);
  [[nodiscard]] bool resize_swapchain(gfx::swapchain_handle swapchain,
                                      u32 width, u32 height);
  /// Waits for the GPU first; not between acquire() and end_frame().
  void destroy_swapchain(gfx::swapchain_handle swapchain) noexcept;

  /// Whether @p swapchain is one with no surface behind it.
  [[nodiscard]] bool offscreen_swapchain(gfx::swapchain_handle swapchain) const;

  [[nodiscard]] bool supports(gfx::format format,
                              gfx::texture_usage_flags usage) const;

  /// The format every swapchain has: chosen by the first one made while none
  /// exists, and kept after it goes, so pipelines built for it stay valid.
  [[nodiscard]] gfx::format swapchain_format() const noexcept {
    return m_swapchain_format;
  }
  [[nodiscard]] gfx::extent2d
  swapchain_extent(gfx::swapchain_handle swapchain) const;
  [[nodiscard]] gfx::surface_alpha
  swapchain_alpha(gfx::swapchain_handle swapchain) const;
  /// The last present found the surface no longer matches: resize it.
  [[nodiscard]] bool swapchain_stale(gfx::swapchain_handle swapchain) const;

  /// Starts GPU frame N: waits for its slot, then flushes uploads. False when
  /// the device is lost. Swapchains are acquired separately.
  [[nodiscard]] bool begin_frame(VulkanCommandContext &out_context);
  /// Takes @p swapchain's next image for this frame, once per frame. A
  /// swapchain that is not acquired is neither drawn nor presented this
  /// frame; the others still are.
  [[nodiscard]] gfx::acquire_result acquire(gfx::swapchain_handle swapchain);
  /// Submits the frame and presents every swapchain it acquired, together.
  /// False when the submission failed; a swapchain its present found out of
  /// date is marked stale instead.
  [[nodiscard]] bool end_frame();

  /// Ends the frame context's command buffer and reserves @p count more after
  /// it, to be recorded apart; the context then continues in a new one, and
  /// must not record until every split recording has ended. An empty split
  /// means none was made.
  [[nodiscard]] gfx::recording_split split_frame(VulkanCommandContext &frame,
                                                 u32 count);
  /// Begins recording @p index of @p split with a command pool of its own. Any
  /// thread or job may record it, one at a time, and must end it before the
  /// frame ends.
  [[nodiscard]] bool begin_recording(gfx::recording_split split, u32 index,
                                     VulkanCommandContext &out);
  void end_recording(VulkanCommandContext &recording);

  [[nodiscard]] bool begin_headless_frame(VulkanCommandContext &out_context);

  [[nodiscard]] bool end_headless_frame();

  /// Ends and submits the open frame with no present and no swapchain
  /// semaphores, then waits for it. Shared by the headless frame and the
  /// offscreen swapchain, which differ only in whether a backbuffer exists.
  [[nodiscard]] bool submit_without_present();

  /// The image @p swapchain acquired for this frame.
  [[nodiscard]] gfx::texture_handle
  current_backbuffer(gfx::swapchain_handle swapchain) const noexcept;

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

  [[nodiscard]] gfx::texture_handle
  create_texture(const gfx::texture_desc &desc);
  [[nodiscard]] gfx::texture_handle
  wrap_texture(const gfx::vk::wrapped_image &image);
  [[nodiscard]] VkSemaphore acquire_frame_wait() noexcept;
  void wait_before_frame(const gfx::vk::timeline_point &point) noexcept;
  void release_frame_wait(VkSemaphore semaphore) noexcept;
  void destroy_texture(gfx::texture_handle handle) noexcept;
  /// Gives @p target @p replacement's image under its own handle and
  /// bindless index, and takes ownership of @p replacement. Under a hold the
  /// frame work before the held one may still be recording against the old
  /// image, so the swap waits for the first frame begun after it is released.
  [[nodiscard]] bool replace_texture(gfx::texture_handle target,
                                     gfx::texture_handle replacement);
  [[nodiscard]] bool set_texture_view_base_mip(gfx::texture_handle handle,
                                               u32 base_mip);
  /// True when the device can sample a texture whose finer mips are not yet
  /// uploaded (via gfx::texture_desc::sampled_base_mip and subsequent view
  /// updates).
  [[nodiscard]] bool supports_mip_streaming() const noexcept { return true; }
  [[nodiscard]] const VulkanTexture *
  texture(gfx::texture_handle handle) const noexcept;
  [[nodiscard]] gfx::texture_usage_flags
  texture_usage(gfx::texture_handle handle) const noexcept;

  [[nodiscard]] gfx::buffer_handle create_buffer(const gfx::buffer_desc &desc);
  void destroy_buffer(gfx::buffer_handle handle) noexcept;
  [[nodiscard]] const VulkanBuffer *
  buffer(gfx::buffer_handle handle) const noexcept;

  [[nodiscard]] gfx::sampler_handle
  create_sampler(const gfx::sampler_desc &desc);
  void destroy_sampler(gfx::sampler_handle handle) noexcept;

  [[nodiscard]] gfx::query_pool_handle create_timestamp_pool(u32 count);
  void destroy_query_pool(gfx::query_pool_handle handle) noexcept;
  [[nodiscard]] bool read_timestamps(gfx::query_pool_handle handle,
                                     std::span<u64> out) const;
  [[nodiscard]] const VulkanQueryPool *
  query_pool(gfx::query_pool_handle handle) const noexcept {
    return m_query_pools.try_get(handle);
  }

  // Memory aliasing Two transients whose lifetimes do not overlap can share
  // bytes. The render graph knows those lifetimes, so it does the packing: it
  // asks what each texture needs, creates one heap, and places them at computed
  // offsets.

  [[nodiscard]] gfx::memory_requirements
  texture_requirements(const gfx::texture_desc &desc) const;
  [[nodiscard]] gfx::heap_handle create_heap(const gfx::heap_desc &desc);
  void destroy_heap(gfx::heap_handle handle) noexcept;
  /// The offset must satisfy the alignment from texture_requirements().
  [[nodiscard]] gfx::texture_handle
  create_texture_in_heap(const gfx::texture_desc &desc, gfx::heap_handle heap,
                         u64 offset);

  [[nodiscard]] gfx::shader_handle create_shader(const gfx::shader_desc &desc);
  void destroy_shader(gfx::shader_handle handle) noexcept;

  /// Copies every input of @p desc on the owner thread into a job to compile
  /// anywhere (gfx::pipeline_job::compile, compile_pipeline_async) and publish
  /// here. Invalid when @p desc is refused.
  [[nodiscard]] gfx::pipeline_job
  prepare_pipeline(const gfx::graphics_pipeline_desc &desc);
  [[nodiscard]] gfx::pipeline_job
  prepare_pipeline(const gfx::compute_pipeline_desc &desc);
  /// Invalid when @p job did not compile.
  [[nodiscard]] gfx::pipeline_handle publish_pipeline(gfx::pipeline_job &&job);
  [[nodiscard]] nx::workload_statistics
  pipeline_compile_statistics() const noexcept {
    return m_pipeline_compile_workload.statistics();
  }
  void destroy_pipeline(gfx::pipeline_handle handle) noexcept;

  [[nodiscard]] u32 texture_index(gfx::texture_handle handle) const noexcept;
  [[nodiscard]] u32 rw_texture_index(gfx::texture_handle handle,
                                     u32 mip = 0) const noexcept;
  [[nodiscard]] u32 sampler_index(gfx::sampler_handle handle) const noexcept;
  [[nodiscard]] VkDeviceAddress
  buffer_address(gfx::buffer_handle handle) const noexcept;

  [[nodiscard]] void *buffer_mapped(gfx::buffer_handle handle) const noexcept;
  /// Makes CPU writes to a mapped buffer visible to the GPU. This is a no-op
  /// for host-coherent allocations.
  void flush_buffer(gfx::buffer_handle handle, u64 offset = 0,
                    u64 size = ~0ull) noexcept;
  void invalidate_buffer(gfx::buffer_handle handle, u64 offset = 0,
                         u64 size = ~0ull) noexcept;

  [[nodiscard]] VkPhysicalDevice vk_physical_device() const noexcept {
    return m_physical_device;
  }
  /// Held around every submit and present to @p queue's VkQueue, which the
  /// host must synchronize: the frame, the uploader and native code all
  /// submit, and queues of different types may be one VkQueue. Never held
  /// across a wait.
  [[nodiscard]] nx::mutex &queue_lock(const gfx::queue_type queue) noexcept {
    return m_queue_locks[m_queue_lock_of[nx::cast<usize>(queue)]];
  }
  [[nodiscard]] VkQueue graphics_queue() const noexcept {
    return m_queues[nx::cast<usize>(gfx::queue_type::Graphics)];
  }
  [[nodiscard]] u32 graphics_family() const noexcept {
    return m_queue_families[nx::cast<usize>(gfx::queue_type::Graphics)];
  }
  [[nodiscard]] std::span<const u32> queue_families() const noexcept {
    return {m_distinct_families.data(), m_distinct_families.size()};
  }

  [[nodiscard]] VkDevice vk_device() const noexcept { return m_device; }
  /// Passes record through dynamic rendering, or through render passes.
  [[nodiscard]] bool dynamic_rendering() const noexcept {
    return m_routes.dynamic_rendering;
  }
  /// The render-pass route's passes and framebuffers; any thread.
  [[nodiscard]] VulkanRenderPasses &render_passes() const noexcept {
    return m_render_passes;
  }

  /// The device's own objects (see native.h).
  [[nodiscard]] gfx::native_device native() const noexcept;
  /// @p handle's own objects; empty where it is not live.
  [[nodiscard]] gfx::native_texture
  native_texture(gfx::texture_handle handle) const noexcept;
  /// The VkImageLayout a texture the graph left in @p state is in.
  [[nodiscard]] u32 native_layout(gfx::resource_state state) const noexcept;
  /// Whether the instance or the device was made with extension @p name.
  [[nodiscard]] bool native_extension(nx::string_view name) const noexcept;
  /// The queue made for @p client (gfx::native_device_client::queue_family), or
  /// none.
  [[nodiscard]] gfx::native_queue
  native_queue(const gfx::native_device_client &client) const noexcept;
  [[nodiscard]] VulkanTexture *texture_mut(gfx::texture_handle handle) noexcept;

  [[nodiscard]] const gfx::shader_reflection &
  reflection(gfx::shader_handle handle) const noexcept;

  [[nodiscard]] gfx::resource_state
  texture_state(gfx::texture_handle handle) const noexcept;
  void set_texture_state(gfx::texture_handle handle,
                         gfx::resource_state state) noexcept;
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
  [[nodiscard]] bool descriptor_heap() const noexcept {
    return m_routes.descriptor_heap;
  }
  /// Binds the bindless slots, set or heaps, for every bind point.
  void bind_bindless(VkCommandBuffer cmd) const noexcept;
  /// What a pipeline on the descriptor-heap route maps the bindless set's
  /// bindings with; chained into each of its stages.
  [[nodiscard]] VkShaderDescriptorSetAndBindingMappingInfoEXT
  heap_mappings() const noexcept;
  [[nodiscard]] const VulkanPipeline *
  pipeline(gfx::pipeline_handle handle) const noexcept;

  /// Backend calls returning DEVICE_LOST pass through this owner so one
  /// coherent report is collected before the device is torn down.
  VkResult check_result(VkResult result, const char *call, const char *file,
                        i32 line) const noexcept;

private:
  friend class VulkanUploader;

  [[nodiscard]] bool
  prepare_graphics_pipeline(const gfx::graphics_pipeline_desc &desc,
                            VulkanGraphicsPipelineCompileJob &out) const;
  [[nodiscard]] bool
  prepare_compute_pipeline(const gfx::compute_pipeline_desc &desc,
                           VulkanComputePipelineCompileJob &out) const;
  [[nodiscard]] VulkanCompiledPipeline
  compile_graphics_pipeline(const VulkanGraphicsPipelineCompileJob &job) const;
  [[nodiscard]] VulkanCompiledPipeline
  compile_compute_pipeline(const VulkanComputePipelineCompileJob &job) const;
  /// gfx::pipeline_job's compile step, on a VulkanPipelineJob payload.
  static bool compile_job(void *payload);

  bool create_instance(const gfx::device_desc &desc);
  bool rebuild_debug_messenger(bool address_bindings) noexcept;
  bool select_physical_device();
  bool
  create_logical_device(std::span<gfx::native_device_client *const> clients);
  [[nodiscard]] bool route_promoted_entry_points() noexcept;
  bool create_allocator();
  bool create_bindless_heap();
  bool create_default_sampler();
  bool create_bindless_globals();
  bool create_descriptor_heaps();
  bool create_descriptor_heap_buffer(DescriptorHeapBuffer &heap,
                                     VkDeviceSize alignment,
                                     nx::string_view name);
  void destroy_bindless() noexcept;
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
                                               const gfx::swapchain_desc &desc);
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

  void write_texture_descriptor(u32 index,
                                const VulkanTexture &texture) noexcept;
  void write_rw_texture_descriptor(u32 index, const VulkanTexture &texture,
                                   u32 mip) noexcept;
  void write_sampler_descriptor(u32 index, VkSampler sampler,
                                const VkSamplerCreateInfo &info) noexcept;
  void write_heap_descriptor(const VkResourceDescriptorInfoEXT &resource,
                             VkDeviceSize offset) noexcept;
  [[nodiscard]] u32 alloc_rw_slot() noexcept;
  void free_rw_slot(u32 slot) noexcept;

  VkInstance m_instance = VK_NULL_HANDLE;
  u32 m_instance_api_version = VK_API_VERSION_1_0;
  VkDebugUtilsMessengerEXT m_messenger = VK_NULL_HANDLE;
  VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
  DeviceRoutes m_routes;
  mutable VulkanRenderPasses m_render_passes;
  gfx::route_flags m_disabled_routes = gfx::route::None;
  gfx::route_flags m_enabled_routes = gfx::route::None;
  VkDevice m_device = VK_NULL_HANDLE;
  VmaAllocator m_allocator = VK_NULL_HANDLE;
  gfx::device_caps m_caps;
  nx::string m_device_name;
  nx::string m_init_error;
  gfx::device_init_report m_init_report;
  bool m_debug_utils = false;
  bool m_sync_validation = false;
  /// Every instance and device extension enabled, the clients' among them.
  nx::small_vector<nx::string, 48> m_native_extensions;
  mutable VulkanGpuCrashReporter m_gpu_crash;
  VulkanGpuProfiler m_gpu_profiler;
  VulkanGpuProfiler m_copy_profiler;

  VkQueue m_queues[nx::cast<usize>(gfx::queue_type::Count)] = {};
  u32 m_queue_families[nx::cast<usize>(gfx::queue_type::Count)] = {};
  nx::fixed_vector<u32, nx::cast<usize>(gfx::queue_type::Count)>
      m_distinct_families;
  nx::mutex m_queue_locks[nx::cast<usize>(gfx::queue_type::Count)];
  /// Which lock each queue type takes: the first type with the same VkQueue.
  u8 m_queue_lock_of[nx::cast<usize>(gfx::queue_type::Count)] = {0, 1, 2};

  struct ClientQueue {
    const gfx::native_device_client *client = nullptr;
    gfx::native_queue queue;
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

  nx::slot_map<VulkanSwapchain, gfx::swapchain_handle> m_swapchains;
  /// The swapchains the frame being recorded acquired, in acquire order.
  nx::small_vector<gfx::swapchain_handle, 4> m_acquired;
  gfx::format m_swapchain_format = gfx::format::Unknown;
  bool m_frame_active = false;

  nx::sync_slot_map<VulkanTexture, gfx::texture_handle> m_textures;
  nx::sync_slot_map<VulkanBuffer, gfx::buffer_handle> m_buffers;
  nx::slot_map<VulkanSampler, gfx::sampler_handle> m_samplers;
  nx::slot_map<VulkanQueryPool, gfx::query_pool_handle> m_query_pools;
  nx::slot_map<VulkanHeap, gfx::heap_handle> m_heaps;
  nx::slot_map<VulkanShader, gfx::shader_handle> m_shaders;
  nx::sync_slot_map<VulkanPipeline, gfx::pipeline_handle> m_pipelines;

  VulkanUploader m_uploader;
  /// Copy-timeline value the current frame's graphics submit must wait on so
  /// that resources uploaded before the frame are ready; 0 when none pending.
  u64 m_frame_upload_wait = 0;

  /// A client's timeline that every frame waits on.
  struct FrameWait {
    VkSemaphore semaphore = VK_NULL_HANDLE;
    u64 value = 0;
    bool released = false;
    /// Once released, the first frame submitted without it.
    u64 retire_after = 0;
  };
  static constexpr usize MAX_FRAME_WAITS = 8;
  // The frame thread, submitting frame @p frame_value.
  void add_frame_waits(nx::small_vector<VkSemaphoreSubmitInfo, 8> &waits,
                       u64 frame_value) noexcept;
  void collect_frame_waits(u64 completed_timeline_value) noexcept;
  nx::mutex m_frame_wait_lock;
  nx::fixed_vector<FrameWait, MAX_FRAME_WAITS> m_frame_waits;

  struct PendingDelete {
    u64 timeline_value = 0;
    gfx::texture_handle texture;
    gfx::buffer_handle buffer;
    gfx::sampler_handle sampler;
    gfx::pipeline_handle pipeline;
    gfx::heap_handle heap;
    /// The held frame work it waits on first; 0 once only the timeline is left.
    u64 hold = 0;
  };
  // m_garbage_lock held.
  void retire(PendingDelete pending) noexcept;

  struct PendingReplace {
    gfx::texture_handle target;
    gfx::texture_handle replacement;
    /// The last held frame work that may name the target's old image.
    u64 after = 0;
  };
  // Neither takes m_garbage_lock on entry; each waits for the GPU first.
  bool swap_texture(gfx::texture_handle target,
                    gfx::texture_handle replacement) noexcept;
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

} // namespace nx::gfx::detail
