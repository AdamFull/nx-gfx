#pragma once

/**
 * @file vk_device.h
 * @brief Vulkan 1.3 device and swapchain.
 *
 */

#include "foundation/containers/fixed_vector.h"
#include "foundation/containers/slot_map.h"
#include "foundation/containers/small_vector.h"
#include "foundation/strings/utf8_string.h"
#include "rhi/descs.h"
#include "rhi/handles.h"
#include "rhi/types.h"
#include "rhi/vulkan/vk_common.h"

// VMA's header is included here because VmaAllocation/VmaAllocator appear in
// members below. Nothing above rhi/ sees it.
#include <vk_mem_alloc.h>

namespace nxe::rhi {

class VulkanDevice;

/// Options for device creation, filled in by the renderer.
struct DeviceDesc {
  nx::string_view application_name = "nx2d";
  u32 application_version = 0;
  /// Install the validation layer and debug messenger. Defaults on in debug.
  /// NX_DEBUG is only *defined* in debug builds rather than defined to 0/1, so
  /// this has to branch in the preprocessor - `NX_DEBUG != 0` in an initialiser
  /// is an undeclared identifier in release.
#if NX_DEBUG
  bool enable_validation = true;
#else
  bool enable_validation = false;
#endif
  /// Abort on a validation error rather than only logging it. Useful in CI.
  bool break_on_validation_error = false;
};

/// A texture as the backend sees it. Swapchain images are also textures, but
/// with `owns_image` false because the swapchain owns them.
struct VulkanTexture {
  VkImage image = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  Format format = Format::Unknown;
  u32 width = 0;
  u32 height = 0;
  u32 mip_levels = 1;
  u32 layers = 1;
  TextureUsageFlags usage = TextureUsage::None;
  /// Tracked so barriers can be generated without the caller restating it.
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

/// Records commands. One per recording thread per frame; contexts are handed
/// out by the device and returned to it on submit.
class VulkanCommandContext {
public:
  VulkanCommandContext() = default;

  [[nodiscard]] VkCommandBuffer handle() const noexcept { return m_cmd; }
  [[nodiscard]] bool valid() const noexcept { return m_cmd != VK_NULL_HANDLE; }

  /// Dynamic rendering only: there are no render pass or framebuffer objects
  /// in this API, because Vulkan 1.3 and Metal both do without them.
  void begin_render_pass(const RenderPassDesc &desc);
  void end_render_pass();

  void set_viewport(const Viewport &viewport);
  void set_scissor(const Rect2D &rect);

  /// Explicit barriers. The render graph is the intended caller.
  void barrier(const TextureBarrier *textures, u32 texture_count,
               const BufferBarrier *buffers = nullptr, u32 buffer_count = 0);
  void barrier(const TextureBarrier &texture) { barrier(&texture, 1); }

  /// Debug label scopes, surfaced in RenderDoc / NSight / Xcode.
  void push_debug_group(nx::string_view name, f32 r = 1.f, f32 g = 1.f,
                        f32 b = 1.f);
  void pop_debug_group();

private:
  friend class VulkanDevice;

  VkCommandBuffer m_cmd = VK_NULL_HANDLE;
  VulkanDevice *m_device = nullptr;
  bool m_in_render_pass = false;
};

/// One frame's worth of per-frame GPU objects.
struct FrameContext {
  VkCommandPool command_pool = VK_NULL_HANDLE;
  nx::small_vector<VkCommandBuffer, 8> command_buffers;
  u32 used_command_buffers = 0;
  /// Signalled by the present engine when this frame's image is acquirable.
  VkSemaphore acquire_semaphore = VK_NULL_HANDLE;
  /// Timeline value this frame's submission will signal.
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
  /// Rebuilds for a new size, reusing the existing surface. Safe to call with
  /// a zero size (returns false, nothing renderable).
  [[nodiscard]] bool resize_swapchain(u32 width, u32 height);
  void destroy_swapchain() noexcept;

  [[nodiscard]] Format swapchain_format() const noexcept {
    return m_swapchain_format;
  }
  [[nodiscard]] Extent2D swapchain_extent() const noexcept {
    return {m_swapchain_extent.width, m_swapchain_extent.height};
  }

  // Frame

  /// Waits for this frame's slot to be free, acquires an image, and opens a
  /// command context. On anything but Success no frame is in progress and
  /// end_frame() must not be called.
  [[nodiscard]] AcquireResult begin_frame(VulkanCommandContext &out_context);

  /// Closes the context, submits it, and presents. Returns false if the
  /// swapchain needs recreating (the frame still rendered).
  [[nodiscard]] bool end_frame();

  /// The image this frame is rendering into.
  [[nodiscard]] TextureHandle current_backbuffer() const noexcept {
    return m_backbuffers.empty() ? TextureHandle{}
                                 : m_backbuffers[m_image_index];
  }

  /// Blocks until the GPU has finished everything. Only for shutdown and
  /// resource teardown - never in a frame.
  void wait_idle() noexcept;

  // Resources

  [[nodiscard]] TextureHandle create_texture(const TextureDesc &desc);
  void destroy_texture(TextureHandle handle) noexcept;
  [[nodiscard]] const VulkanTexture *
  texture(TextureHandle handle) const noexcept;

  [[nodiscard]] BufferHandle create_buffer(const BufferDesc &desc);
  void destroy_buffer(BufferHandle handle) noexcept;
  [[nodiscard]] const VulkanBuffer *buffer(BufferHandle handle) const noexcept;

  // Internals used by VulkanCommandContext

  [[nodiscard]] VkDevice vk_device() const noexcept { return m_device; }
  [[nodiscard]] VulkanTexture *texture_mut(TextureHandle handle) noexcept;
  [[nodiscard]] bool debug_utils_enabled() const noexcept {
    return m_debug_utils;
  }

private:
  bool create_instance(const DeviceDesc &desc);
  bool select_physical_device();
  bool create_logical_device();
  bool create_allocator();
  bool create_frame_contexts(u32 frames_in_flight);
  bool build_swapchain(u32 width, u32 height);
  void destroy_backbuffer_views() noexcept;
  void set_object_name(u64 object, VkObjectType type, nx::string_view name);
  /// Retires resources whose frame has certainly finished on the GPU.
  void collect_garbage(u64 completed_timeline_value) noexcept;

  // Instance / device
  VkInstance m_instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT m_messenger = VK_NULL_HANDLE;
  VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
  VkDevice m_device = VK_NULL_HANDLE;
  VmaAllocator m_allocator = VK_NULL_HANDLE;
  DeviceCaps m_caps;
  nx::string m_device_name;
  bool m_debug_utils = false;
  bool m_break_on_error = false;

  // Queues. Compute/copy alias onto graphics when the device has no dedicated
  // family, so callers never have to special-case it.
  VkQueue m_queues[nx::cast<usize>(QueueType::Count)] = {};
  u32 m_queue_families[nx::cast<usize>(QueueType::Count)] = {};

  // Frame pacing.
  VkSemaphore m_timeline = VK_NULL_HANDLE;
  u64 m_timeline_value = 0;
  nx::small_vector<FrameContext, 3> m_frames;
  u32 m_frame_index = 0;
  u32 m_frames_in_flight = 2;

  // Swapchain
  VkSurfaceKHR m_surface = VK_NULL_HANDLE;
  VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
  VkExtent2D m_swapchain_extent = {};
  Format m_swapchain_format = Format::Unknown;
  VkPresentModeKHR m_present_mode = VK_PRESENT_MODE_FIFO_KHR;
  nx::small_vector<TextureHandle, 4> m_backbuffers;
  /// Signalled when rendering into image i is done; one per image, not per
  /// frame, because present waits on the image's own semaphore.
  nx::small_vector<VkSemaphore, 4> m_release_semaphores;
  u32 m_image_index = 0;
  bool m_frame_active = false;

  // Resource storage. slot_map gives stable, generation-checked handles whose
  // index doubles as the bindless descriptor slot.
  nx::slot_map<VulkanTexture, TextureHandle> m_textures;
  nx::slot_map<VulkanBuffer, BufferHandle> m_buffers;

  /// Deferred destruction: a resource freed during frame N cannot be released
  /// until the GPU has passed N, so it waits on the timeline value.
  struct PendingDelete {
    u64 timeline_value = 0;
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
  };
  nx::small_vector<PendingDelete, 32> m_pending_deletes;
};

} // namespace nxe::rhi
