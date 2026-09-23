
#include "core/foundation/diagnostics/profiler.h"
#include "rendering/rhi/vulkan/vk_device.h"

#include "core/foundation/diagnostics/log.h"
#include "core/foundation/fibers/scheduler.h"

namespace nxe::rhi {
namespace {

[[nodiscard]] bool surface_resized(VkPhysicalDevice physical,
                                   VkSurfaceKHR surface,
                                   const VkExtent2D current) noexcept {
  VkSurfaceCapabilitiesKHR caps{};
  if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps) !=
      VK_SUCCESS)
    return false;
  if (caps.currentExtent.width == 0xFFFFFFFFu)
    return false;
  return caps.currentExtent.width != current.width ||
         caps.currentExtent.height != current.height;
}

[[nodiscard]] VkCompositeAlphaFlagBitsKHR
pick_composite_alpha(const VkCompositeAlphaFlagsKHR supported) noexcept {
  constexpr VkCompositeAlphaFlagBitsKHR order[] = {
      VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
      VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
      VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
  };
  for (const VkCompositeAlphaFlagBitsKHR bit : order) {
    if ((supported & nx::cast<VkCompositeAlphaFlagsKHR>(bit)) != 0)
      return bit;
  }
  nx::logw(nx::log::rhi,
           "surface reports no composite alpha mode; asking for opaque");
  return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
}

VkSurfaceKHR create_surface(VkInstance instance,
                            const nx::NativeWindowHandle &window) noexcept {
  VkSurfaceKHR surface = VK_NULL_HANDLE;

  switch (window.kind) {
#if defined(VK_USE_PLATFORM_WIN32_KHR)
  case nx::NativeWindowKind::Win32: {
    VkWin32SurfaceCreateInfoKHR info{
        VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    info.hinstance = nx::cast<HINSTANCE>(GetModuleHandleW(nullptr));
    info.hwnd = nx::cast<HWND>(window.window);
    NX_VK_CHECK(vkCreateWin32SurfaceKHR(instance, &info, nullptr, &surface));
    break;
  }
#endif
#if defined(VK_USE_PLATFORM_ANDROID_KHR)
  case nx::NativeWindowKind::Android: {
    VkAndroidSurfaceCreateInfoKHR info{
        VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
    info.window = nx::cast<ANativeWindow *>(window.window);
    NX_VK_CHECK(vkCreateAndroidSurfaceKHR(instance, &info, nullptr, &surface));
    break;
  }
#endif
#if defined(VK_USE_PLATFORM_XLIB_KHR)
  case nx::NativeWindowKind::Xlib: {
    VkXlibSurfaceCreateInfoKHR info{
        VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR};
    info.dpy = nx::cast<Display *>(window.display);
    info.window =
        nx::cast<::Window>(reinterpret_cast<uintptr_t>(window.window));
    NX_VK_CHECK(vkCreateXlibSurfaceKHR(instance, &info, nullptr, &surface));
    break;
  }
#endif
#if defined(VK_USE_PLATFORM_WAYLAND_KHR)
  case nx::NativeWindowKind::Wayland: {
    VkWaylandSurfaceCreateInfoKHR info{
        VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR};
    info.display = nx::cast<wl_display *>(window.display);
    info.surface = nx::cast<wl_surface *>(window.window);
    NX_VK_CHECK(vkCreateWaylandSurfaceKHR(instance, &info, nullptr, &surface));
    break;
  }
#endif
#if defined(VK_USE_PLATFORM_METAL_EXT)
  case nx::NativeWindowKind::UIKit:
  case nx::NativeWindowKind::Cocoa: {
    VkMetalSurfaceCreateInfoEXT info{
        VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT};
    info.pLayer = nx::cast<const CAMetalLayer *>(window.window);
    NX_VK_CHECK(vkCreateMetalSurfaceEXT(instance, &info, nullptr, &surface));
    break;
  }
#endif
  default:
    nx::logc(nx::log::rhi, "unsupported native window kind for this Vulkan build");
    break;
  }

  return surface;
}

} // namespace

bool VulkanDevice::create_swapchain(const SwapchainDesc &desc) {
  if (device_status() == GpuDeviceStatus::Lost ||
      desc.window.kind == nx::NativeWindowKind::None ||
      desc.window.window == nullptr) {
    return false;
  }

  // Full (re)creation. On mobile the native window is a different object after
  // every suspend/resume, so the old surface must go rather than be reused.
  if (m_surface != VK_NULL_HANDLE) {
    wait_idle();
    if (device_status() == GpuDeviceStatus::Lost)
      return false;
    destroy_swapchain();
  }

  m_surface = create_surface(m_instance, desc.window);
  if (m_surface == VK_NULL_HANDLE)
    return false;

  VkBool32 present_supported = VK_FALSE;
  NX_VK_TRY(vkGetPhysicalDeviceSurfaceSupportKHR(
      m_physical_device, m_queue_families[nx::cast<usize>(QueueType::Graphics)],
      m_surface, &present_supported));
  if (present_supported != VK_TRUE) {
    nx::logc(nx::log::rhi, "graphics queue cannot present to this surface");
    return false;
  }

  // Pick the present mode once; it cannot change without a rebuild anyway.
  u32 mode_count = 0;
  NX_VK_TRY(vkGetPhysicalDeviceSurfacePresentModesKHR(
      m_physical_device, m_surface, &mode_count, nullptr));
  nx::small_vector<VkPresentModeKHR, 8> modes(mode_count);
  NX_VK_TRY(vkGetPhysicalDeviceSurfacePresentModesKHR(
      m_physical_device, m_surface, &mode_count, modes.data()));

  const VkPresentModeKHR wanted = to_vk_present_mode(desc.present_mode);
  m_present_mode = VK_PRESENT_MODE_FIFO_KHR;
  for (const VkPresentModeKHR mode : modes) {
    if (mode == wanted) {
      m_present_mode = wanted;
      break;
    }
  }
  if (m_present_mode != wanted)
    nx::logw(nx::log::rhi, "present mode unsupported, falling back to FIFO");

  // Surface format: prefer the requested one, else the first sRGB-capable
  // 8-bit format, else whatever comes first.
  u32 format_count = 0;
  NX_VK_TRY(vkGetPhysicalDeviceSurfaceFormatsKHR(m_physical_device, m_surface,
                                                 &format_count, nullptr));
  if (format_count == 0) {
    nx::logc(nx::log::rhi, "surface reports no formats");
    return false;
  }
  nx::small_vector<VkSurfaceFormatKHR, 32> formats(format_count);
  NX_VK_TRY(vkGetPhysicalDeviceSurfaceFormatsKHR(
      m_physical_device, m_surface, &format_count, formats.data()));

  const VkFormat preferred = to_vk_format(desc.preferred_format);
  VkSurfaceFormatKHR chosen = formats[0];
  for (const auto &format : formats) {
    if (format.format == preferred &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      chosen = format;
      break;
    }
  }
  m_swapchain_format = from_vk_format(chosen.format);
  if (m_swapchain_format == Format::Unknown) {
    nx::logw(nx::log::rhi, "surface format {} has no portable equivalent",
             nx::cast<i32>(chosen.format));
  }

  return build_swapchain(desc.width, desc.height);
}

bool VulkanDevice::create_offscreen_swapchain(const u32 width, const u32 height,
                                             const Format format) {
  if (device_status() == GpuDeviceStatus::Lost || width == 0 || height == 0 ||
      format == Format::Unknown)
    return false;

  if (m_swapchain != VK_NULL_HANDLE || !m_backbuffers.empty()) {
    wait_idle();
    if (device_status() == GpuDeviceStatus::Lost)
      return false;
    destroy_swapchain();
  }

  m_swapchain_format = format;
  m_swapchain_extent = {width, height};
  m_offscreen = true;

  // One image per frame in flight, same as a surface swapchain gets, so a
  // frame never writes the image the one before it is still reading.
  // CopySrc because a frame capture reads the backbuffer back, and Sampled
  // because a pass may sample what an earlier one wrote to it.
  for (u32 i = 0; i < m_frames_in_flight; ++i) {
    const nx::string name = nx::format("offscreen backbuffer {}", i);
    const TextureHandle image = create_texture({
        .name = name.view(),
        .format = format,
        .width = width,
        .height = height,
        .usage = TextureUsage::RenderTarget | TextureUsage::Sampled |
                 TextureUsage::CopySrc,
    });
    if (!image.valid()) {
      destroy_swapchain();
      return false;
    }
    m_backbuffers.push_back(image);
  }

  m_image_index = 0;
  nx::logd(nx::log::rhi, "offscreen swapchain: {}x{}, {} images, {}", width,
           height, m_frames_in_flight, to_string(format));
  return true;
}

bool VulkanDevice::build_swapchain(const u32 width, const u32 height) {
  VkSurfaceCapabilitiesKHR caps{};
  NX_VK_TRY(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physical_device,
                                                      m_surface, &caps));

  VkExtent2D extent = caps.currentExtent;
  // 0xFFFFFFFF means "you choose"; otherwise the surface dictates the size and
  // the requested one is only a hint.
  if (extent.width == 0xFFFFFFFFu) {
    extent.width = width;
    extent.height = height;
  }
  extent.width = extent.width < caps.minImageExtent.width
                     ? caps.minImageExtent.width
                     : (extent.width > caps.maxImageExtent.width
                            ? caps.maxImageExtent.width
                            : extent.width);
  extent.height = extent.height < caps.minImageExtent.height
                      ? caps.minImageExtent.height
                      : (extent.height > caps.maxImageExtent.height
                             ? caps.maxImageExtent.height
                             : extent.height);

  if (extent.width == 0 || extent.height == 0) {
    m_swapchain_extent = {0, 0};
    return false;
  }

  u32 image_count = m_frames_in_flight + 1;
  if (image_count < caps.minImageCount)
    image_count = caps.minImageCount;
  if (caps.maxImageCount > 0 && image_count > caps.maxImageCount)
    image_count = caps.maxImageCount;

  VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  info.surface = m_surface;
  info.minImageCount = image_count;
  info.imageFormat = to_vk_format(m_swapchain_format);
  info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  info.imageExtent = extent;
  info.imageArrayLayers = 1;
  info.imageUsage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if ((caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0)
    info.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.preTransform =
      (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
          ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
          : caps.currentTransform;
  info.compositeAlpha = pick_composite_alpha(caps.supportedCompositeAlpha);
  info.presentMode = m_present_mode;
  info.clipped = VK_TRUE;
  info.oldSwapchain = m_swapchain;

  VkSwapchainKHR new_swapchain = VK_NULL_HANDLE;
  NX_VK_DEVICE_TRY(
      *this, vkCreateSwapchainKHR(m_device, &info, nullptr, &new_swapchain));

  destroy_backbuffer_views();
  if (m_swapchain != VK_NULL_HANDLE)
    vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
  m_swapchain = new_swapchain;
  m_swapchain_extent = extent;
  set_object_name(vk_handle_value(m_swapchain), VK_OBJECT_TYPE_SWAPCHAIN_KHR,
                  "main swapchain");

  u32 actual_count = 0;
  NX_VK_DEVICE_TRY(*this, vkGetSwapchainImagesKHR(m_device, m_swapchain,
                                                  &actual_count, nullptr));
  nx::small_vector<VkImage, 4> images(actual_count);
  NX_VK_DEVICE_TRY(*this,
                   vkGetSwapchainImagesKHR(m_device, m_swapchain, &actual_count,
                                           images.data()));

  for (u32 i = 0; i < actual_count; ++i) {
    VulkanTexture texture;
    texture.image = images[i];
    texture.format = m_swapchain_format;
    texture.width = extent.width;
    texture.height = extent.height;
    texture.usage = TextureUsage::RenderTarget;
    if ((info.imageUsage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0)
      texture.usage |= TextureUsage::CopySrc;
    texture.state = ResourceState::Undefined;
    texture.owns_image = false;

    VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = images[i];
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = to_vk_format(m_swapchain_format);
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.layerCount = 1;
    NX_VK_DEVICE_TRY(
        *this, vkCreateImageView(m_device, &view_info, nullptr, &texture.rtv));

    set_object_name(vk_handle_value(images[i]), VK_OBJECT_TYPE_IMAGE,
                    nx::format("swapchain image {}", i));
    set_object_name(vk_handle_value(texture.rtv), VK_OBJECT_TYPE_IMAGE_VIEW,
                    nx::format("swapchain image {} attachment", i));

    m_backbuffers.push_back(m_textures.insert(std::move(texture)));

    VkSemaphore release = VK_NULL_HANDLE;
    VkSemaphoreCreateInfo semaphore_info{
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    NX_VK_DEVICE_TRY(
        *this, vkCreateSemaphore(m_device, &semaphore_info, nullptr, &release));
    set_object_name(vk_handle_value(release), VK_OBJECT_TYPE_SEMAPHORE,
                    nx::format("swapchain image {} release", i));
    m_release_semaphores.push_back(release);
  }

  nx::logd(nx::log::rhi, "swapchain: {}x{}, {} images, {}", extent.width,
           extent.height, actual_count, to_string(m_swapchain_format));
  return true;
}

bool VulkanDevice::resize_swapchain(const u32 width, const u32 height) {
  if (m_offscreen)
    return create_offscreen_swapchain(width, height, m_swapchain_format);
  if (m_surface == VK_NULL_HANDLE || device_status() == GpuDeviceStatus::Lost)
    return false;
  wait_idle();
  if (device_status() == GpuDeviceStatus::Lost)
    return false;
  return build_swapchain(width, height);
}

void VulkanDevice::destroy_backbuffer_views() noexcept {
  for (const TextureHandle handle : m_backbuffers) {
    if (const VulkanTexture *texture = m_textures.try_get(handle)) {
      if (texture->rtv != VK_NULL_HANDLE)
        vkDestroyImageView(m_device, texture->rtv, nullptr);
    }
    m_textures.erase(handle);
  }
  m_backbuffers.clear();

  for (const VkSemaphore semaphore : m_release_semaphores)
    if (semaphore != VK_NULL_HANDLE)
      vkDestroySemaphore(m_device, semaphore, nullptr);
  m_release_semaphores.clear();
}

void VulkanDevice::destroy_swapchain() noexcept {
  if (m_device == VK_NULL_HANDLE)
    return;

  if (m_offscreen) {
    // These images are this device's own, so they go back the way any texture
    // does - destroy_backbuffer_views() would drop the views and leak the
    // allocations behind them.
    for (const TextureHandle handle : m_backbuffers)
      destroy_texture(handle);
    m_backbuffers.clear();
    m_offscreen = false;
    m_swapchain_extent = {0, 0};
    m_swapchain_format = Format::Unknown;
    m_image_index = 0;
    return;
  }

  destroy_backbuffer_views();

  if (m_swapchain != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
    m_swapchain = VK_NULL_HANDLE;
  }
  if (m_surface != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    m_surface = VK_NULL_HANDLE;
  }
  m_swapchain_extent = {0, 0};
}

// A render job that would block in the driver waits on a blocking-service
// thread, so its worker keeps running other jobs meanwhile.
VkResult VulkanDevice::wait_timeline(const u64 value) noexcept {
  u64 reached = 0;
  VkResult result =
      vkGetSemaphoreCounterValueKHR(m_device, m_timeline, &reached);
  if (result != VK_SUCCESS || reached >= value)
    return result;
  VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  wait.semaphoreCount = 1;
  wait.pSemaphores = &m_timeline;
  wait.pValues = &value;
  nx::fiber::run_blocking(
      [&] { result = vkWaitSemaphoresKHR(m_device, &wait, UINT64_MAX); });
  return result;
}

// Takes the slot's next command buffer, allocating it the first time, and
// begins it with the bindless set bound at both bind points.
VkCommandBuffer VulkanDevice::next_command_buffer(CommandSlot &slot) {
  if (slot.pool == VK_NULL_HANDLE) {
    VkCommandPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex =
        m_queue_families[nx::cast<usize>(QueueType::Graphics)];
    if (NX_VK_DEVICE_CHECK(*this, vkCreateCommandPool(m_device, &pool_info,
                                                      nullptr, &slot.pool)) !=
        VK_SUCCESS)
      return VK_NULL_HANDLE;
    set_object_name(vk_handle_value(slot.pool), VK_OBJECT_TYPE_COMMAND_POOL,
                    nx::format("frame {} command pool", m_frame_index));
  }
  if (slot.used == slot.cmds.size()) {
    VkCommandBufferAllocateInfo alloc{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = slot.pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (NX_VK_DEVICE_CHECK(*this, vkAllocateCommandBuffers(m_device, &alloc,
                                                           &cmd)) != VK_SUCCESS)
      return VK_NULL_HANDLE;
    set_object_name(vk_handle_value(cmd), VK_OBJECT_TYPE_COMMAND_BUFFER,
                    nx::format("frame {} command buffer {}", m_frame_index,
                               slot.cmds.size()));
    slot.cmds.push_back(cmd);
  }
  const VkCommandBuffer cmd = slot.cmds[slot.used];

  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (NX_VK_DEVICE_CHECK(*this, vkBeginCommandBuffer(cmd, &begin)) !=
      VK_SUCCESS)
    return VK_NULL_HANDLE;
  ++slot.used;

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          m_pipeline_layout, BINDLESS_SET, 1, &m_bindless.set,
                          0, nullptr);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          m_pipeline_layout, BINDLESS_SET, 1, &m_bindless.set,
                          0, nullptr);
  return cmd;
}

// Only once the GPU is done with the slot: resets every pool the frame used
// last time round and begins the frame's own command buffer.
VkCommandBuffer VulkanDevice::open_frame_slot(FrameContext &frame) {
  const auto reset = [&](CommandSlot &slot) {
    if (slot.used == 0)
      return true;
    slot.used = 0;
    return NX_VK_DEVICE_CHECK(
               *this, vkResetCommandPool(m_device, slot.pool, 0)) == VK_SUCCESS;
  };
  if (!reset(frame.slot))
    return VK_NULL_HANDLE;
  frame.idle.clear();
  for (const nx::unique_ptr<CommandSlot> &recorder : frame.recorders) {
    if (!reset(*recorder))
      return VK_NULL_HANDLE;
    frame.idle.push_back(recorder.get());
  }
  frame.order.clear();
  const VkCommandBuffer cmd = next_command_buffer(frame.slot);
  if (cmd != VK_NULL_HANDLE)
    frame.order.push_back(cmd);
  return cmd;
}

// Ends the frame context's command buffer and lists what the frame submits,
// in order. A split recording that never ended is left out.
bool VulkanDevice::close_frame(
    FrameContext &frame, nx::small_vector<VkCommandBufferSubmitInfo, 8> &out) {
  m_gpu_profiler.end_open_zones(frame.slot.zones);
  if (frame.order.empty() || frame.order.back() == VK_NULL_HANDLE ||
      NX_VK_DEVICE_CHECK(*this, vkEndCommandBuffer(frame.order.back())) !=
          VK_SUCCESS)
    return false;
  for (const VkCommandBuffer cmd : frame.order) {
    if (cmd == VK_NULL_HANDLE) {
      nx::loge(nx::log::rhi, "a recording split off the frame never ended");
      continue;
    }
    VkCommandBufferSubmitInfo info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    info.commandBuffer = cmd;
    out.push_back(info);
  }
  return true;
}

RecordingSplit VulkanDevice::split_frame(VulkanCommandContext &frame_context,
                                         const u32 count) {
  if (!m_frame_active || count == 0 || frame_context.m_slot != nullptr ||
      frame_context.m_cmd == VK_NULL_HANDLE)
    return {};
  if (frame_context.m_in_render_pass) {
    nx::loge(nx::log::rhi, "split_frame inside a render pass");
    return {};
  }
  FrameContext &frame = m_frames[m_frame_index];
  m_gpu_profiler.end_open_zones(frame.slot.zones);
  if (NX_VK_DEVICE_CHECK(*this, vkEndCommandBuffer(frame_context.m_cmd)) !=
      VK_SUCCESS) {
    frame.order.back() = VK_NULL_HANDLE;
    frame_context.m_cmd = VK_NULL_HANDLE;
    return {};
  }
  const RecordingSplit split{nx::cast<u32>(frame.order.size()), count};
  frame.order.resize(frame.order.size() + count, VK_NULL_HANDLE);
  frame_context.m_cmd = next_command_buffer(frame.slot);
  frame.order.push_back(frame_context.m_cmd);
  return split;
}

bool VulkanDevice::begin_recording(const RecordingSplit split, const u32 index,
                                   VulkanCommandContext &out) {
  if (!m_frame_active || index >= split.count)
    return false;
  FrameContext &frame = m_frames[m_frame_index];
  CommandSlot *slot = nullptr;
  {
    const nx::scoped_lock<nx::mutex> held(m_recording_lock);
    if (!frame.idle.empty()) {
      slot = frame.idle.back();
      frame.idle.pop_back();
    } else {
      slot = frame.recorders.emplace_back(nx::make_unique<CommandSlot>()).get();
    }
  }
  const VkCommandBuffer cmd = next_command_buffer(*slot);
  if (cmd == VK_NULL_HANDLE) {
    const nx::scoped_lock<nx::mutex> held(m_recording_lock);
    frame.idle.push_back(slot);
    return false;
  }
  out = {};
  out.m_cmd = cmd;
  out.m_device = this;
  out.m_zones = &slot->zones;
  out.m_slot = slot;
  out.m_position = split.first + index;
  return true;
}

void VulkanDevice::end_recording(VulkanCommandContext &recording) {
  CommandSlot *slot = recording.m_slot;
  if (slot == nullptr)
    return;
  if (recording.m_in_render_pass) {
    nx::loge(nx::log::rhi, "a split recording ended inside a render pass");
    recording.end_render_pass();
  }
  m_gpu_profiler.end_open_zones(slot->zones);
  FrameContext &frame = m_frames[m_frame_index];
  if (NX_VK_DEVICE_CHECK(*this, vkEndCommandBuffer(recording.m_cmd)) ==
      VK_SUCCESS)
    frame.order[recording.m_position] = recording.m_cmd;
  {
    const nx::scoped_lock<nx::mutex> held(m_recording_lock);
    frame.idle.push_back(slot);
  }
  recording = {};
}

AcquireResult VulkanDevice::begin_frame(VulkanCommandContext &out_context) {
  NX_PROFILE_ZONE("vk::begin_frame");
  if (device_status() == GpuDeviceStatus::Lost ||
      m_swapchain_extent.width == 0 ||
      (m_swapchain == VK_NULL_HANDLE && !m_offscreen))
    return AcquireResult::SurfaceLost;

  FrameContext &frame = m_frames[m_frame_index];

  if (frame.timeline_value > 0) {
    NX_PROFILE_ZONE_COLOR("vk::wait_gpu", 0x604040);
    if (NX_VK_DEVICE_CHECK(*this, wait_timeline(frame.timeline_value)) !=
        VK_SUCCESS)
      return AcquireResult::SurfaceLost;
  }

  u64 completed = 0;
  if (NX_VK_DEVICE_CHECK(*this, vkGetSemaphoreCounterValueKHR(
                                    m_device, m_timeline, &completed)) !=
      VK_SUCCESS)
    return AcquireResult::SurfaceLost;
  collect_garbage(completed);
  m_uploader.reclaim();

  VkResult acquired = VK_SUCCESS;
  if (m_offscreen) {
    // Nothing to acquire from, and so no acquire semaphore for end_frame to
    // wait on: the image for this frame is the one that belongs to its slot.
    m_image_index = m_frame_index;
  } else {
    NX_PROFILE_ZONE_COLOR("vk::acquire", 0x604040);
    const auto acquire = [&](const u64 timeout) {
      return vkAcquireNextImageKHR(m_device, m_swapchain, timeout,
                                   frame.acquire_semaphore, VK_NULL_HANDLE,
                                   &m_image_index);
    };
    acquired = acquire(0);
    if (acquired == VK_NOT_READY || acquired == VK_TIMEOUT)
      nx::fiber::run_blocking([&] { acquired = acquire(UINT64_MAX); });
  }

  if (acquired == VK_ERROR_OUT_OF_DATE_KHR)
    return AcquireResult::OutOfDate;
  if (acquired == VK_ERROR_SURFACE_LOST_KHR)
    return AcquireResult::SurfaceLost;
  if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) {
    NX_VK_DEVICE_CHECK(*this, acquired);
    return AcquireResult::SurfaceLost;
  }

  const VkCommandBuffer cmd = open_frame_slot(frame);
  if (cmd == VK_NULL_HANDLE)
    return AcquireResult::SurfaceLost;
  m_gpu_profiler.collect(cmd);

  // Turn finished async decodes into textures, submit the batch of uploads
  // accumulated since the last frame, then take ownership of everything it
  // produced and note the copy-timeline value this frame's submit must wait on.
  m_uploader.pump();
  m_uploader.flush_batch();
  m_frame_upload_wait = m_uploader.record_acquires(cmd);

  out_context = {};
  out_context.m_cmd = cmd;
  out_context.m_device = this;
  out_context.m_zones = &frame.slot.zones;
  m_frame_active = true;

  const bool suboptimal =
      acquired == VK_SUBOPTIMAL_KHR &&
      surface_resized(m_physical_device, m_surface, m_swapchain_extent);
  return suboptimal ? AcquireResult::Suboptimal : AcquireResult::Success;
}

bool VulkanDevice::begin_headless_frame(VulkanCommandContext &out_context) {
  if (m_device == VK_NULL_HANDLE || m_frame_active ||
      device_status() == GpuDeviceStatus::Lost)
    return false;

  if (m_frames.empty())
    return false;

  FrameContext &frame = m_frames[m_frame_index];
  if (frame.timeline_value > 0) {
    NX_PROFILE_ZONE_COLOR("vk::wait_gpu", 0x604040);
    if (NX_VK_DEVICE_CHECK(*this, wait_timeline(frame.timeline_value)) !=
        VK_SUCCESS)
      return false;
  }

  u64 completed = 0;
  if (NX_VK_DEVICE_CHECK(*this, vkGetSemaphoreCounterValueKHR(
                                    m_device, m_timeline, &completed)) !=
      VK_SUCCESS)
    return false;
  collect_garbage(completed);
  m_uploader.reclaim();

  const VkCommandBuffer cmd = open_frame_slot(frame);
  if (cmd == VK_NULL_HANDLE)
    return false;
  m_gpu_profiler.collect(cmd);
  m_uploader.pump();
  m_uploader.flush_batch();
  m_frame_upload_wait = m_uploader.record_acquires(cmd);

  out_context = {};
  out_context.m_cmd = cmd;
  out_context.m_device = this;
  out_context.m_zones = &frame.slot.zones;
  m_frame_active = true;
  m_headless_frame = true;
  return true;
}

bool VulkanDevice::end_headless_frame() {
  if (!m_frame_active || !m_headless_frame ||
      device_status() == GpuDeviceStatus::Lost)
    return false;
  m_frame_active = false;
  m_headless_frame = false;
  return submit_without_present();
}

bool VulkanDevice::submit_without_present() {
  FrameContext &frame = m_frames[m_frame_index];
  nx::small_vector<VkCommandBufferSubmitInfo, 8> cmd_infos;
  if (!close_frame(frame, cmd_infos))
    return false;

  ++m_timeline_value;
  m_report_frame.store(m_timeline_value, std::memory_order_release);
  frame.timeline_value = m_timeline_value;

  // No acquire semaphore to wait on and no release semaphore to signal - there
  // is no swapchain in either direction. Only the upload timeline still
  // matters, when this frame reads something the transfer queue wrote.
  VkSemaphoreSubmitInfo wait_info{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
  u32 wait_count = 0;
  if (m_frame_upload_wait > 0) {
    wait_info.semaphore = m_uploader.timeline();
    wait_info.value = m_frame_upload_wait;
    wait_info.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    wait_count = 1;
  }

  VkSemaphoreSubmitInfo signal_info{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
  signal_info.semaphore = m_timeline;
  signal_info.value = m_timeline_value;
  signal_info.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

  VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  submit.waitSemaphoreInfoCount = wait_count;
  submit.pWaitSemaphoreInfos = wait_count > 0 ? &wait_info : nullptr;
  submit.commandBufferInfoCount = nx::cast<u32>(cmd_infos.size());
  submit.pCommandBufferInfos = cmd_infos.data();
  submit.signalSemaphoreInfoCount = 1;
  submit.pSignalSemaphoreInfos = &signal_info;

  if (NX_VK_DEVICE_CHECK(
          *this,
          vkQueueSubmit2KHR(m_queues[nx::cast<usize>(QueueType::Graphics)], 1,
                            &submit, VK_NULL_HANDLE)) != VK_SUCCESS)
    return false;

  if (NX_VK_DEVICE_CHECK(*this, wait_timeline(m_timeline_value)) != VK_SUCCESS)
    return false;

  m_frame_index = (m_frame_index + 1) % m_frames_in_flight;
  return true;
}

bool VulkanDevice::end_frame() {
  if (device_status() == GpuDeviceStatus::Lost)
    return false;
  if (!m_frame_active)
    return true;
  // Crossing the two kinds of frame would wait on an acquire semaphore nothing
  // ever signals, which hangs the GPU rather than failing.
  if (m_headless_frame) {
    nx::loge(nx::log::rhi, "end_frame on a headless frame; use end_headless_frame");
    return false;
  }
  m_frame_active = false;

  // No surface, so there is nothing to transition to PRESENT_SRC and nothing
  // to present to: the frame is finished by submitting it. The image keeps
  // whatever state the last pass left it in, which is what a reader wants.
  if (m_offscreen)
    return submit_without_present();

  FrameContext &frame = m_frames[m_frame_index];
  const VkCommandBuffer cmd = frame.order.back();

  // The image must be in PRESENT_SRC before handing it to the present engine.
  if (VulkanTexture *backbuffer =
          m_textures.try_get(m_backbuffers[m_image_index])) {
    if (backbuffer->state != ResourceState::Present) {
      const VkStateInfo from = to_vk_state(backbuffer->state);
      const VkStateInfo to = to_vk_state(ResourceState::Present);

      VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
      barrier.srcStageMask = from.stage;
      barrier.srcAccessMask = from.access;
      barrier.dstStageMask = to.stage;
      barrier.dstAccessMask = to.access;
      barrier.oldLayout = from.layout;
      barrier.newLayout = to.layout;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = backbuffer->image;
      barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

      VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      dependency.imageMemoryBarrierCount = 1;
      dependency.pImageMemoryBarriers = &barrier;
      vkCmdPipelineBarrier2KHR(cmd, &dependency);

      backbuffer->state = ResourceState::Present;
    }
  }

  nx::small_vector<VkCommandBufferSubmitInfo, 8> cmd_infos;
  if (!close_frame(frame, cmd_infos))
    return false;

  ++m_timeline_value;
  m_report_frame.store(m_timeline_value, std::memory_order_release);
  frame.timeline_value = m_timeline_value;

  // Wait on the swapchain acquire, and - when the transfer queue uploaded
  // resources this frame consumes - on the upload timeline as well.
  VkSemaphoreSubmitInfo wait_infos[2] = {};
  wait_infos[0].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
  wait_infos[0].semaphore = frame.acquire_semaphore;
  wait_infos[0].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  u32 wait_count = 1;
  if (m_frame_upload_wait > 0) {
    wait_infos[1].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    wait_infos[1].semaphore = m_uploader.timeline();
    wait_infos[1].value = m_frame_upload_wait;
    wait_infos[1].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    wait_count = 2;
  }

  VkSemaphoreSubmitInfo signal_infos[2] = {};
  signal_infos[0].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
  signal_infos[0].semaphore = m_release_semaphores[m_image_index];
  signal_infos[0].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  signal_infos[1].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
  signal_infos[1].semaphore = m_timeline;
  signal_infos[1].value = m_timeline_value;
  signal_infos[1].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

  VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  submit.waitSemaphoreInfoCount = wait_count;
  submit.pWaitSemaphoreInfos = wait_infos;
  submit.commandBufferInfoCount = nx::cast<u32>(cmd_infos.size());
  submit.pCommandBufferInfos = cmd_infos.data();
  submit.signalSemaphoreInfoCount = 2;
  submit.pSignalSemaphoreInfos = signal_infos;

  if (NX_VK_DEVICE_CHECK(
          *this,
          vkQueueSubmit2KHR(m_queues[nx::cast<usize>(QueueType::Graphics)], 1,
                            &submit, VK_NULL_HANDLE)) != VK_SUCCESS)
    return false;

  VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  present.waitSemaphoreCount = 1;
  present.pWaitSemaphores = &m_release_semaphores[m_image_index];
  present.swapchainCount = 1;
  present.pSwapchains = &m_swapchain;
  present.pImageIndices = &m_image_index;

  VkResult presented = VK_SUCCESS;
  {
    NX_PROFILE_ZONE("vk::present");
    nx::fiber::run_blocking([&] {
      presented = vkQueuePresentKHR(
          m_queues[nx::cast<usize>(QueueType::Graphics)], &present);
    });
  }

  m_frame_index = (m_frame_index + 1) % m_frames_in_flight;

  if (presented == VK_ERROR_OUT_OF_DATE_KHR)
    return false;
  if (presented == VK_SUBOPTIMAL_KHR)
    return !surface_resized(m_physical_device, m_surface, m_swapchain_extent);
  if (presented != VK_SUCCESS) {
    NX_VK_DEVICE_CHECK(*this, presented);
    return false;
  }
  return true;
}

} // namespace nxe::rhi
