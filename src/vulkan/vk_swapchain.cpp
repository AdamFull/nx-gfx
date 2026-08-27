
#include "core/foundation/diagnostics/profiler.h"
#include "core/rendering/rhi/vulkan/vk_device.h"

#include "core/foundation/diagnostics/log.h"

namespace nxe::rhi {
namespace {

const nx::log::Category log_rhi = nx::log::category("rhi");

[[nodiscard]] bool open_slot(VkDevice device, const u32 queue_family,
                             VkDescriptorSet bindless, VkPipelineLayout layout,
                             FrameContext &frame) {
  CommandSlot &slot = frame.slot;

  if (slot.pool == VK_NULL_HANDLE) {
    VkCommandPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = queue_family;
    if (NX_VK_CHECK(vkCreateCommandPool(device, &pool_info, nullptr,
                                        &slot.pool)) != VK_SUCCESS)
      return false;

    VkCommandBufferAllocateInfo alloc{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = slot.pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    if (NX_VK_CHECK(vkAllocateCommandBuffers(device, &alloc, &slot.cmd)) !=
        VK_SUCCESS)
      return false;
  } else {
    NX_VK_CHECK(vkResetCommandPool(device, slot.pool, 0));
  }

  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (NX_VK_CHECK(vkBeginCommandBuffer(slot.cmd, &begin)) != VK_SUCCESS)
    return false;

  vkCmdBindDescriptorSets(slot.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
                          BINDLESS_SET, 1, &bindless, 0, nullptr);
  vkCmdBindDescriptorSets(slot.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout,
                          BINDLESS_SET, 1, &bindless, 0, nullptr);
  return true;
}

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
  nx::logw(log_rhi,
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
    info.window = nx::cast<::Window>(nx::cast<uintptr_t>(window.window));
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
    nx::logc(log_rhi, "unsupported native window kind for this Vulkan build");
    break;
  }

  return surface;
}

}

bool VulkanDevice::create_swapchain(const SwapchainDesc &desc) {
  if (desc.window.kind == nx::NativeWindowKind::None ||
      desc.window.window == nullptr) {
    return false;
  }

  // Full (re)creation. On mobile the native window is a different object after
  // every suspend/resume, so the old surface must go rather than be reused.
  if (m_surface != VK_NULL_HANDLE) {
    wait_idle();
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
    nx::logc(log_rhi, "graphics queue cannot present to this surface");
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
    nx::logw(log_rhi, "present mode unsupported, falling back to FIFO");

  // Surface format: prefer the requested one, else the first sRGB-capable
  // 8-bit format, else whatever comes first.
  u32 format_count = 0;
  NX_VK_TRY(vkGetPhysicalDeviceSurfaceFormatsKHR(m_physical_device, m_surface,
                                                 &format_count, nullptr));
  if (format_count == 0) {
    nx::logc(log_rhi, "surface reports no formats");
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
    nx::logw(log_rhi, "surface format {} has no portable equivalent",
             nx::cast<i32>(chosen.format));
  }

  return build_swapchain(desc.width, desc.height);
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
  NX_VK_TRY(vkCreateSwapchainKHR(m_device, &info, nullptr, &new_swapchain));

  destroy_backbuffer_views();
  if (m_swapchain != VK_NULL_HANDLE)
    vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
  m_swapchain = new_swapchain;
  m_swapchain_extent = extent;

  u32 actual_count = 0;
  NX_VK_TRY(
      vkGetSwapchainImagesKHR(m_device, m_swapchain, &actual_count, nullptr));
  nx::small_vector<VkImage, 4> images(actual_count);
  NX_VK_TRY(vkGetSwapchainImagesKHR(m_device, m_swapchain, &actual_count,
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
    NX_VK_TRY(vkCreateImageView(m_device, &view_info, nullptr, &texture.rtv));

    m_backbuffers.push_back(m_textures.insert(std::move(texture)));

    VkSemaphore release = VK_NULL_HANDLE;
    VkSemaphoreCreateInfo semaphore_info{
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    NX_VK_TRY(vkCreateSemaphore(m_device, &semaphore_info, nullptr, &release));
    m_release_semaphores.push_back(release);
  }

  nx::logi(log_rhi, "swapchain: {}x{}, {} images, {}", extent.width,
           extent.height, actual_count, to_string(m_swapchain_format));
  return true;
}

bool VulkanDevice::resize_swapchain(const u32 width, const u32 height) {
  if (m_surface == VK_NULL_HANDLE)
    return false;
  wait_idle();
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

AcquireResult VulkanDevice::begin_frame(VulkanCommandContext &out_context) {
  NX_PROFILE_ZONE("vk::begin_frame");
  if (m_swapchain == VK_NULL_HANDLE || m_swapchain_extent.width == 0)
    return AcquireResult::SurfaceLost;

  FrameContext &frame = m_frames[m_frame_index];

  if (frame.timeline_value > 0) {
    NX_PROFILE_ZONE_COLOR("vk::wait_gpu", 0x604040);
    VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wait.semaphoreCount = 1;
    wait.pSemaphores = &m_timeline;
    wait.pValues = &frame.timeline_value;
    if (NX_VK_CHECK(vkWaitSemaphoresKHR(m_device, &wait, UINT64_MAX)) !=
        VK_SUCCESS)
      return AcquireResult::SurfaceLost;
  }

  u64 completed = 0;
  NX_VK_CHECK(vkGetSemaphoreCounterValueKHR(m_device, m_timeline, &completed));
  collect_garbage(completed);
  m_uploader.reclaim();

  VkResult acquired = VK_SUCCESS;
  {
    NX_PROFILE_ZONE_COLOR("vk::acquire", 0x604040);
    acquired = vkAcquireNextImageKHR(m_device, m_swapchain, UINT64_MAX,
                                     frame.acquire_semaphore, VK_NULL_HANDLE,
                                     &m_image_index);
  }

  if (acquired == VK_ERROR_OUT_OF_DATE_KHR)
    return AcquireResult::OutOfDate;
  if (acquired == VK_ERROR_SURFACE_LOST_KHR)
    return AcquireResult::SurfaceLost;
  if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) {
    NX_VK_CHECK(acquired);
    return AcquireResult::SurfaceLost;
  }

  if (!open_slot(m_device,
                 m_queue_families[nx::cast<usize>(QueueType::Graphics)],
                 m_bindless.set, m_pipeline_layout, frame))
    return AcquireResult::SurfaceLost;

  const VkCommandBuffer cmd = frame.slot.cmd;

  // Turn finished async decodes into textures, submit the batch of uploads
  // accumulated since the last frame, then take ownership of everything it
  // produced and note the copy-timeline value this frame's submit must wait on.
  m_uploader.pump();
  m_uploader.flush_batch();
  m_frame_upload_wait = m_uploader.record_acquires(cmd);

  out_context.m_cmd = cmd;
  out_context.m_device = this;
  out_context.m_in_render_pass = false;
  m_frame_active = true;

  const bool suboptimal =
      acquired == VK_SUBOPTIMAL_KHR &&
      surface_resized(m_physical_device, m_surface, m_swapchain_extent);
  return suboptimal ? AcquireResult::Suboptimal : AcquireResult::Success;
}

bool VulkanDevice::begin_headless_frame(VulkanCommandContext &out_context) {
  if (m_device == VK_NULL_HANDLE || m_frame_active)
    return false;

  if (m_frames.empty())
    return false;

  FrameContext &frame = m_frames[m_frame_index];
  if (frame.timeline_value > 0) {
    NX_PROFILE_ZONE_COLOR("vk::wait_gpu", 0x604040);
    VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wait.semaphoreCount = 1;
    wait.pSemaphores = &m_timeline;
    wait.pValues = &frame.timeline_value;
    if (NX_VK_CHECK(vkWaitSemaphoresKHR(m_device, &wait, UINT64_MAX)) !=
        VK_SUCCESS)
      return false;
  }

  u64 completed = 0;
  NX_VK_CHECK(vkGetSemaphoreCounterValueKHR(m_device, m_timeline, &completed));
  collect_garbage(completed);
  m_uploader.reclaim();

  if (!open_slot(m_device,
                 m_queue_families[nx::cast<usize>(QueueType::Graphics)],
                 m_bindless.set, m_pipeline_layout, frame))
    return false;

  const VkCommandBuffer cmd = frame.slot.cmd;
  m_uploader.pump();
  m_uploader.flush_batch();
  m_frame_upload_wait = m_uploader.record_acquires(cmd);

  out_context.m_cmd = cmd;
  out_context.m_device = this;
  out_context.m_in_render_pass = false;
  m_frame_active = true;
  m_headless_frame = true;
  return true;
}

bool VulkanDevice::end_headless_frame() {
  if (!m_frame_active || !m_headless_frame)
    return false;
  m_frame_active = false;
  m_headless_frame = false;

  FrameContext &frame = m_frames[m_frame_index];
  NX_VK_CHECK(vkEndCommandBuffer(frame.slot.cmd));

  ++m_timeline_value;
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

  VkCommandBufferSubmitInfo cmd_info{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
  cmd_info.commandBuffer = frame.slot.cmd;

  VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  submit.waitSemaphoreInfoCount = wait_count;
  submit.pWaitSemaphoreInfos = wait_count > 0 ? &wait_info : nullptr;
  submit.commandBufferInfoCount = 1;
  submit.pCommandBufferInfos = &cmd_info;
  submit.signalSemaphoreInfoCount = 1;
  submit.pSignalSemaphoreInfos = &signal_info;

  if (NX_VK_CHECK(vkQueueSubmit2KHR(m_queues[nx::cast<usize>(QueueType::Graphics)],
                                 1, &submit, VK_NULL_HANDLE)) != VK_SUCCESS)
    return false;

  VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  wait.semaphoreCount = 1;
  wait.pSemaphores = &m_timeline;
  wait.pValues = &m_timeline_value;
  if (NX_VK_CHECK(vkWaitSemaphoresKHR(m_device, &wait, UINT64_MAX)) != VK_SUCCESS)
    return false;

  m_frame_index = (m_frame_index + 1) % m_frames_in_flight;
  return true;
}

bool VulkanDevice::end_frame() {
  if (!m_frame_active)
    return true;
  // Crossing the two kinds of frame would wait on an acquire semaphore nothing
  // ever signals, which hangs the GPU rather than failing.
  if (m_headless_frame) {
    nx::loge(log_rhi, "end_frame on a headless frame; use end_headless_frame");
    return false;
  }
  m_frame_active = false;

  FrameContext &frame = m_frames[m_frame_index];
  const VkCommandBuffer cmd = frame.slot.cmd;

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

  NX_VK_CHECK(vkEndCommandBuffer(frame.slot.cmd));

  ++m_timeline_value;
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

  VkCommandBufferSubmitInfo cmd_info{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
  cmd_info.commandBuffer = frame.slot.cmd;

  VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  submit.waitSemaphoreInfoCount = wait_count;
  submit.pWaitSemaphoreInfos = wait_infos;
  submit.commandBufferInfoCount = 1;
  submit.pCommandBufferInfos = &cmd_info;
  submit.signalSemaphoreInfoCount = 2;
  submit.pSignalSemaphoreInfos = signal_infos;

  NX_VK_CHECK(vkQueueSubmit2KHR(m_queues[nx::cast<usize>(QueueType::Graphics)], 1,
                             &submit, VK_NULL_HANDLE));

  VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  present.waitSemaphoreCount = 1;
  present.pWaitSemaphores = &m_release_semaphores[m_image_index];
  present.swapchainCount = 1;
  present.pSwapchains = &m_swapchain;
  present.pImageIndices = &m_image_index;

  VkResult presented = VK_SUCCESS;
  {
    NX_PROFILE_ZONE("vk::present");
    presented = vkQueuePresentKHR(
        m_queues[nx::cast<usize>(QueueType::Graphics)], &present);
  }

  m_frame_index = (m_frame_index + 1) % m_frames_in_flight;

  if (presented == VK_ERROR_OUT_OF_DATE_KHR)
    return false;
  if (presented == VK_SUBOPTIMAL_KHR)
    return !surface_resized(m_physical_device, m_surface, m_swapchain_extent);
  if (presented != VK_SUCCESS) {
    NX_VK_CHECK(presented);
    return false;
  }
  return true;
}

}
