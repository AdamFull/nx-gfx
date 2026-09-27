
#include "core/foundation/diagnostics/profiler.h"
#include "rendering/rhi/vulkan/vk_device.h"

#include "core/foundation/diagnostics/log.h"
#include "core/foundation/fibers/scheduler.h"

#include <span>

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
pick_composite_alpha(const VkCompositeAlphaFlagsKHR supported,
                     const SurfaceAlpha requested) noexcept {
  constexpr VkCompositeAlphaFlagBitsKHR opaque_order[] = {
      VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
      VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
      VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
  };
  // Post-multiplied would multiply premultiplied colour by alpha twice.
  // Inherit promises nothing: Intel's Win32 driver offers it and then drops
  // alpha even under DWM per-pixel composition, so it counts as opaque here.
  constexpr VkCompositeAlphaFlagBitsKHR transparent_order[] = {
      VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
      VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
  };
  const std::span<const VkCompositeAlphaFlagBitsKHR> order =
      requested == SurfaceAlpha::Opaque
          ? std::span<const VkCompositeAlphaFlagBitsKHR>(opaque_order)
          : std::span<const VkCompositeAlphaFlagBitsKHR>(transparent_order);
  for (const VkCompositeAlphaFlagBitsKHR bit : order) {
    if ((supported & nx::cast<VkCompositeAlphaFlagsKHR>(bit)) != 0)
      return bit;
  }
  nx::logw(nx::log::rhi,
           "surface reports no usable composite alpha mode; asking for opaque");
  return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
}

[[nodiscard]] SurfaceAlpha
from_vk_composite_alpha(const VkCompositeAlphaFlagBitsKHR bit) noexcept {
  switch (bit) {
  case VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR:
    return SurfaceAlpha::Premultiplied;
  case VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR:
    return SurfaceAlpha::Unmultiplied;
  case VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR:
    return SurfaceAlpha::WindowSystem;
  default:
    return SurfaceAlpha::Opaque;
  }
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

SwapchainHandle VulkanDevice::create_swapchain(const SwapchainDesc &desc) {
  if (device_status() == GpuDeviceStatus::Lost ||
      desc.window.kind == nx::NativeWindowKind::None ||
      desc.window.window == nullptr) {
    return {};
  }

  VulkanSwapchain sc;
  const auto make = [&]() -> bool {
    sc.surface = create_surface(m_instance, desc.window);
    if (sc.surface == VK_NULL_HANDLE)
      return false;

    VkBool32 present_supported = VK_FALSE;
    NX_VK_TRY(vkGetPhysicalDeviceSurfaceSupportKHR(
        m_physical_device,
        m_queue_families[nx::cast<usize>(QueueType::Graphics)], sc.surface,
        &present_supported));
    if (present_supported != VK_TRUE) {
      nx::logc(nx::log::rhi, "graphics queue cannot present to this surface");
      return false;
    }

    // Pick the present mode once; it cannot change without a rebuild anyway.
    u32 mode_count = 0;
    NX_VK_TRY(vkGetPhysicalDeviceSurfacePresentModesKHR(
        m_physical_device, sc.surface, &mode_count, nullptr));
    nx::small_vector<VkPresentModeKHR, 8> modes(mode_count);
    NX_VK_TRY(vkGetPhysicalDeviceSurfacePresentModesKHR(
        m_physical_device, sc.surface, &mode_count, modes.data()));

    const VkPresentModeKHR wanted = to_vk_present_mode(desc.present_mode);
    sc.present_mode = VK_PRESENT_MODE_FIFO_KHR;
    for (const VkPresentModeKHR mode : modes) {
      if (mode == wanted) {
        sc.present_mode = wanted;
        break;
      }
    }
    if (sc.present_mode != wanted)
      nx::logw(nx::log::rhi, "present mode unsupported, falling back to FIFO");

    u32 format_count = 0;
    NX_VK_TRY(vkGetPhysicalDeviceSurfaceFormatsKHR(
        m_physical_device, sc.surface, &format_count, nullptr));
    if (format_count == 0) {
      nx::logc(nx::log::rhi, "surface reports no formats");
      return false;
    }
    nx::small_vector<VkSurfaceFormatKHR, 32> formats(format_count);
    NX_VK_TRY(vkGetPhysicalDeviceSurfaceFormatsKHR(
        m_physical_device, sc.surface, &format_count, formats.data()));

    // The first swapchain prefers the requested format, else takes whatever
    // comes first. The rest must match it: pipelines are built for one.
    const bool fixed = !m_swapchains.empty();
    const VkFormat preferred =
        to_vk_format(fixed ? m_swapchain_format : desc.preferred_format);
    const VkSurfaceFormatKHR *chosen = fixed ? nullptr : &formats[0];
    for (const VkSurfaceFormatKHR &format : formats) {
      if (format.format == preferred &&
          format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        chosen = &format;
        break;
      }
    }
    if (chosen == nullptr) {
      nx::loge(nx::log::rhi,
               "surface cannot present {}, the format the other swapchains "
               "have",
               to_string(m_swapchain_format));
      return false;
    }
    if (!fixed) {
      m_swapchain_format = from_vk_format(chosen->format);
      if (m_swapchain_format == Format::Unknown)
        nx::logw(nx::log::rhi, "surface format {} has no portable equivalent",
                 nx::cast<i32>(chosen->format));
    }

    sc.requested_alpha = desc.alpha;
#if defined(VK_USE_PLATFORM_WIN32_KHR)
    if (desc.alpha == SurfaceAlpha::Premultiplied) {
      VkSurfaceCapabilitiesKHR caps{};
      NX_VK_TRY(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physical_device,
                                                          sc.surface, &caps));
      if ((caps.supportedCompositeAlpha &
           VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) == 0 &&
          create_composed_swapchain(sc, desc))
        return true;
    }
#endif
    return build_swapchain(sc, desc.width, desc.height) &&
           create_acquire_semaphores(sc);
  };
  if (!make()) {
    release_swapchain(sc);
    return {};
  }
  return m_swapchains.insert(std::move(sc));
}

// One image per frame in flight, same as a surface swapchain gets, so a frame
// never writes the image the one before it is still reading. CopySrc because a
// frame capture reads the backbuffer back, and Sampled because a pass may
// sample what an earlier one wrote to it.
bool VulkanDevice::make_offscreen_images(VulkanSwapchain &sc, const u32 width,
                                         const u32 height) {
  for (u32 i = 0; i < m_frames_in_flight; ++i) {
    const nx::string name = nx::format("offscreen backbuffer {}", i);
    const TextureHandle image = create_texture({
        .name = name.view(),
        .format = m_swapchain_format,
        .width = width,
        .height = height,
        .usage = TextureUsage::RenderTarget | TextureUsage::Sampled |
                 TextureUsage::CopySrc,
    });
    if (!image.valid())
      return false;
    sc.backbuffers.push_back(image);
  }
  sc.extent = {width, height};
  sc.image_index = 0;
  nx::logd(nx::log::rhi, "offscreen swapchain: {}x{}, {} images, {}", width,
           height, m_frames_in_flight, to_string(m_swapchain_format));
  return true;
}

SwapchainHandle VulkanDevice::create_offscreen_swapchain(const u32 width,
                                                        const u32 height,
                                                        const Format format) {
  if (device_status() == GpuDeviceStatus::Lost || width == 0 || height == 0 ||
      format == Format::Unknown)
    return {};
  if (!m_swapchains.empty() && format != m_swapchain_format) {
    nx::loge(nx::log::rhi,
             "offscreen swapchain in {} while the other swapchains are {}",
             to_string(format), to_string(m_swapchain_format));
    return {};
  }

  m_swapchain_format = format;
  VulkanSwapchain sc;
  sc.offscreen = true;
  if (!make_offscreen_images(sc, width, height)) {
    release_swapchain(sc);
    return {};
  }
  return m_swapchains.insert(std::move(sc));
}

bool VulkanDevice::create_acquire_semaphores(VulkanSwapchain &sc) {
  VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  for (u32 i = 0; i < m_frames_in_flight; ++i) {
    VkSemaphore semaphore = VK_NULL_HANDLE;
    NX_VK_DEVICE_TRY(*this,
                     vkCreateSemaphore(m_device, &info, nullptr, &semaphore));
    set_object_name(vk_handle_value(semaphore), VK_OBJECT_TYPE_SEMAPHORE,
                    nx::format("frame {} acquire", i));
    sc.acquire_semaphores.push_back(semaphore);
  }
  return true;
}

bool VulkanDevice::build_swapchain(VulkanSwapchain &sc, const u32 width,
                                   const u32 height) {
  VkSurfaceCapabilitiesKHR caps{};
  NX_VK_TRY(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physical_device,
                                                      sc.surface, &caps));

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
    sc.extent = {0, 0};
    return false;
  }

  u32 image_count = m_frames_in_flight + 1;
  if (image_count < caps.minImageCount)
    image_count = caps.minImageCount;
  if (caps.maxImageCount > 0 && image_count > caps.maxImageCount)
    image_count = caps.maxImageCount;

  VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  info.surface = sc.surface;
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
  info.compositeAlpha =
      pick_composite_alpha(caps.supportedCompositeAlpha, sc.requested_alpha);
  info.presentMode = sc.present_mode;
  info.clipped = VK_TRUE;
  info.oldSwapchain = sc.swapchain;

  VkSwapchainKHR new_swapchain = VK_NULL_HANDLE;
  NX_VK_DEVICE_TRY(
      *this, vkCreateSwapchainKHR(m_device, &info, nullptr, &new_swapchain));

  destroy_backbuffer_views(sc);
  if (sc.swapchain != VK_NULL_HANDLE)
    vkDestroySwapchainKHR(m_device, sc.swapchain, nullptr);
  sc.swapchain = new_swapchain;
  sc.extent = extent;
  sc.alpha = from_vk_composite_alpha(info.compositeAlpha);
  sc.image_index = 0;
  set_object_name(vk_handle_value(sc.swapchain), VK_OBJECT_TYPE_SWAPCHAIN_KHR,
                  "swapchain");

  u32 actual_count = 0;
  NX_VK_DEVICE_TRY(*this, vkGetSwapchainImagesKHR(m_device, sc.swapchain,
                                                  &actual_count, nullptr));
  nx::small_vector<VkImage, 4> images(actual_count);
  NX_VK_DEVICE_TRY(*this,
                   vkGetSwapchainImagesKHR(m_device, sc.swapchain,
                                           &actual_count, images.data()));

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

    sc.backbuffers.push_back(m_textures.insert(std::move(texture)));

    VkSemaphore release = VK_NULL_HANDLE;
    VkSemaphoreCreateInfo semaphore_info{
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    NX_VK_DEVICE_TRY(
        *this, vkCreateSemaphore(m_device, &semaphore_info, nullptr, &release));
    set_object_name(vk_handle_value(release), VK_OBJECT_TYPE_SEMAPHORE,
                    nx::format("swapchain image {} release", i));
    sc.release_semaphores.push_back(release);
  }

  nx::logd(nx::log::rhi, "swapchain: {}x{}, {} images, {}", extent.width,
           extent.height, actual_count, to_string(m_swapchain_format));
  return true;
}

bool VulkanDevice::resize_swapchain(const SwapchainHandle handle,
                                    const u32 width, const u32 height) {
  VulkanSwapchain *const sc = m_swapchains.try_get(handle);
  if (sc == nullptr || device_status() == GpuDeviceStatus::Lost)
    return false;
  if (sc->acquired) {
    nx::loge(nx::log::rhi, "resize_swapchain between acquire and end_frame");
    return false;
  }
  if (sc->offscreen && (width == 0 || height == 0))
    return false;
  wait_idle();
  if (device_status() == GpuDeviceStatus::Lost)
    return false;
  sc->stale = false;

  if (sc->offscreen) {
    for (const TextureHandle image : sc->backbuffers)
      destroy_texture(image);
    sc->backbuffers.clear();
    return make_offscreen_images(*sc, width, height);
  }
#if defined(VK_USE_PLATFORM_WIN32_KHR)
  if (sc->composed()) {
    release_composed_images(*sc);
    if (!sc->composition->resize(width, height) ||
        !import_composed_images(*sc, width, height))
      return false;
    sc->extent = {width, height};
    return true;
  }
#endif
  if (sc->surface == VK_NULL_HANDLE)
    return false;
  return build_swapchain(*sc, width, height);
}

void VulkanDevice::destroy_backbuffer_views(VulkanSwapchain &sc) noexcept {
  for (const TextureHandle handle : sc.backbuffers) {
    if (const VulkanTexture *texture = m_textures.try_get(handle)) {
      if (texture->rtv != VK_NULL_HANDLE)
        vkDestroyImageView(m_device, texture->rtv, nullptr);
    }
    m_textures.erase(handle);
  }
  sc.backbuffers.clear();

  for (const VkSemaphore semaphore : sc.release_semaphores)
    if (semaphore != VK_NULL_HANDLE)
      vkDestroySemaphore(m_device, semaphore, nullptr);
  sc.release_semaphores.clear();
}

void VulkanDevice::release_swapchain(VulkanSwapchain &sc) noexcept {
  if (m_device == VK_NULL_HANDLE)
    return;
#if defined(VK_USE_PLATFORM_WIN32_KHR)
  release_composition(sc);
#endif
  if (sc.offscreen) {
    // These images are this device's own, so they go back the way any texture
    // does - destroy_backbuffer_views() would drop the views and leak the
    // allocations behind them.
    for (const TextureHandle handle : sc.backbuffers)
      destroy_texture(handle);
    sc.backbuffers.clear();
  }
  destroy_backbuffer_views(sc);
  for (const VkSemaphore semaphore : sc.acquire_semaphores)
    vkDestroySemaphore(m_device, semaphore, nullptr);
  sc.acquire_semaphores.clear();
  if (sc.swapchain != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(m_device, sc.swapchain, nullptr);
    sc.swapchain = VK_NULL_HANDLE;
  }
  if (sc.surface != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(m_instance, sc.surface, nullptr);
    sc.surface = VK_NULL_HANDLE;
  }
  sc.extent = {0, 0};
}

void VulkanDevice::destroy_swapchain(const SwapchainHandle handle) noexcept {
  VulkanSwapchain *const sc = m_swapchains.try_get(handle);
  if (sc == nullptr)
    return;
  if (sc->acquired) {
    nx::loge(nx::log::rhi, "destroy_swapchain between acquire and end_frame");
    return;
  }
  if (m_device != VK_NULL_HANDLE)
    wait_idle();
  release_swapchain(*sc);
  m_swapchains.erase(handle);
}

bool VulkanDevice::offscreen_swapchain(const SwapchainHandle handle) const {
  const VulkanSwapchain *const sc = m_swapchains.try_get(handle);
  return sc != nullptr && sc->offscreen;
}

Extent2D VulkanDevice::swapchain_extent(const SwapchainHandle handle) const {
  const VulkanSwapchain *const sc = m_swapchains.try_get(handle);
  return sc != nullptr ? Extent2D{sc->extent.width, sc->extent.height}
                       : Extent2D{};
}

SurfaceAlpha VulkanDevice::swapchain_alpha(const SwapchainHandle handle) const {
  const VulkanSwapchain *const sc = m_swapchains.try_get(handle);
  return sc != nullptr ? sc->alpha : SurfaceAlpha::Opaque;
}

bool VulkanDevice::swapchain_stale(const SwapchainHandle handle) const {
  const VulkanSwapchain *const sc = m_swapchains.try_get(handle);
  return sc != nullptr && sc->stale;
}

TextureHandle
VulkanDevice::current_backbuffer(const SwapchainHandle handle) const noexcept {
  const VulkanSwapchain *const sc = m_swapchains.try_get(handle);
  return sc != nullptr && !sc->backbuffers.empty()
             ? sc->backbuffers[sc->image_index]
             : TextureHandle{};
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

bool VulkanDevice::open_frame(VulkanCommandContext &out_context) {
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

  // Submit the batch of uploads accumulated since the last frame, then take
  // ownership of everything it produced and note the copy-timeline value this
  // frame's submit must wait on.
  m_uploader.flush_batch();
  m_frame_upload_wait = m_uploader.record_acquires(cmd);

  out_context = {};
  out_context.m_cmd = cmd;
  out_context.m_device = this;
  out_context.m_zones = &frame.slot.zones;
  m_frame_active = true;
  return true;
}

bool VulkanDevice::begin_frame(VulkanCommandContext &out_context) {
  NX_PROFILE_ZONE("vk::begin_frame");
  if (m_device == VK_NULL_HANDLE || m_frames.empty() ||
      device_status() == GpuDeviceStatus::Lost)
    return false;
  if (m_frame_active) {
    nx::loge(nx::log::rhi, "begin_frame called twice without an end_frame");
    return false;
  }
  m_acquired.clear();
  return open_frame(out_context);
}

AcquireResult VulkanDevice::acquire(const SwapchainHandle handle) {
  VulkanSwapchain *const sc = m_swapchains.try_get(handle);
  if (!m_frame_active || m_headless_frame || sc == nullptr ||
      device_status() == GpuDeviceStatus::Lost)
    return AcquireResult::SurfaceLost;
  if (sc->acquired) {
    nx::loge(nx::log::rhi, "a swapchain was acquired twice in one frame");
    return AcquireResult::SurfaceLost;
  }
  if (sc->extent.width == 0 ||
      (!sc->presents() && !sc->offscreen && !sc->composed()))
    return AcquireResult::SurfaceLost;

  VkResult acquired = VK_SUCCESS;
  if (sc->offscreen || sc->composed()) {
    // Nothing to acquire from, and so no acquire semaphore for end_frame to
    // wait on: the image for this frame is the one that belongs to its slot.
    sc->image_index = m_frame_index;
  } else {
    NX_PROFILE_ZONE_COLOR("vk::acquire", 0x604040);
    const VkSemaphore semaphore = sc->acquire_semaphores[m_frame_index];
    const auto acquire_image = [&](const u64 timeout) {
      return vkAcquireNextImageKHR(m_device, sc->swapchain, timeout, semaphore,
                                   VK_NULL_HANDLE, &sc->image_index);
    };
    acquired = acquire_image(0);
    if (acquired == VK_NOT_READY || acquired == VK_TIMEOUT)
      nx::fiber::run_blocking([&] { acquired = acquire_image(UINT64_MAX); });
  }

  if (acquired == VK_ERROR_OUT_OF_DATE_KHR)
    return AcquireResult::OutOfDate;
  if (acquired == VK_ERROR_SURFACE_LOST_KHR)
    return AcquireResult::SurfaceLost;
  if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) {
    NX_VK_DEVICE_CHECK(*this, acquired);
    return AcquireResult::SurfaceLost;
  }

#if defined(VK_USE_PLATFORM_WIN32_KHR)
  if (sc->composed())
    acquire_composed_image(*sc, m_frames[m_frame_index].order.back());
#endif
  sc->acquired = true;
  m_acquired.push_back(handle);

  const bool suboptimal =
      acquired == VK_SUBOPTIMAL_KHR &&
      surface_resized(m_physical_device, sc->surface, sc->extent);
  return suboptimal ? AcquireResult::Suboptimal : AcquireResult::Success;
}

bool VulkanDevice::begin_headless_frame(VulkanCommandContext &out_context) {
  if (m_device == VK_NULL_HANDLE || m_frame_active || m_frames.empty() ||
      device_status() == GpuDeviceStatus::Lost)
    return false;
  m_acquired.clear();
  if (!open_frame(out_context))
    return false;
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

void VulkanDevice::finish_backbuffer(VulkanSwapchain &sc,
                                     const VkCommandBuffer cmd) noexcept {
  // No surface: the image keeps whatever state the last pass left it in,
  // which is what a reader wants.
  if (sc.offscreen)
    return;
  VulkanTexture *const image = m_textures.try_get(sc.backbuffers[sc.image_index]);
  if (image == nullptr)
    return;
#if defined(VK_USE_PLATFORM_WIN32_KHR)
  if (sc.composed()) {
    // General first, then handed to D3D11: a release and the acquire that
    // matches it must describe the same layout transition, and this way both
    // describe none.
    const VkStateInfo from = to_vk_state(image->state);
    VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.srcStageMask = from.stage;
    barrier.srcAccessMask = from.access;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
    barrier.oldLayout = from.layout;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image->image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2KHR(cmd, &dependency);

    VkImageMemoryBarrier2 release = barrier;
    release.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    release.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    release.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
    release.dstAccessMask = 0;
    release.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    release.srcQueueFamilyIndex =
        m_queue_families[nx::cast<usize>(QueueType::Graphics)];
    release.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
    dependency.pImageMemoryBarriers = &release;
    vkCmdPipelineBarrier2KHR(cmd, &dependency);
    image->state = ResourceState::General;
    return;
  }
#endif
  // The image must be in PRESENT_SRC before handing it to the present engine.
  if (image->state == ResourceState::Present)
    return;
  const VkStateInfo from = to_vk_state(image->state);
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
  barrier.image = image->image;
  barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.imageMemoryBarrierCount = 1;
  dependency.pImageMemoryBarriers = &barrier;
  vkCmdPipelineBarrier2KHR(cmd, &dependency);
  image->state = ResourceState::Present;
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

  FrameContext &frame = m_frames[m_frame_index];
  nx::small_vector<VulkanSwapchain *, 4> acquired;
  for (const SwapchainHandle handle : m_acquired)
    if (VulkanSwapchain *const sc = m_swapchains.try_get(handle))
      acquired.push_back(sc);
  m_acquired.clear();
  const auto let_go = [&] {
    for (VulkanSwapchain *const sc : acquired)
      sc->acquired = false;
  };

  const VkCommandBuffer cmd = frame.order.back();
  for (VulkanSwapchain *const sc : acquired)
    finish_backbuffer(*sc, cmd);

  nx::small_vector<VkCommandBufferSubmitInfo, 8> cmd_infos;
  if (!close_frame(frame, cmd_infos)) {
    let_go();
    return false;
  }

  ++m_timeline_value;
  m_report_frame.store(m_timeline_value, std::memory_order_release);
  frame.timeline_value = m_timeline_value;

  // Every acquired image, and - when the transfer queue uploaded resources this
  // frame consumes - the upload timeline too.
  const auto semaphore = [](const VkSemaphore handle, const u64 value) {
    VkSemaphoreSubmitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    info.semaphore = handle;
    info.value = value;
    info.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    return info;
  };
  nx::small_vector<VkSemaphoreSubmitInfo, 8> waits;
  nx::small_vector<VkSemaphoreSubmitInfo, 8> signals;
  if (m_frame_upload_wait > 0)
    waits.push_back(semaphore(m_uploader.timeline(), m_frame_upload_wait));
  bool offscreen_only = !acquired.empty();
  for (VulkanSwapchain *const sc : acquired) {
    if (sc->presents()) {
      waits.push_back(semaphore(sc->acquire_semaphores[m_frame_index], 0));
      signals.push_back(semaphore(sc->release_semaphores[sc->image_index], 0));
    }
#if defined(VK_USE_PLATFORM_WIN32_KHR)
    if (sc->composed()) {
      // The image's previous copy into the swap chain must be done before
      // this frame writes it again.
      if (const u64 copied = sc->composed_serials[sc->image_index]; copied > 0)
        waits.push_back(semaphore(sc->composed_copied, copied));
      signals.push_back(semaphore(sc->composed_rendered, m_timeline_value));
    }
#endif
    offscreen_only = offscreen_only && sc->offscreen;
  }
  signals.push_back(semaphore(m_timeline, m_timeline_value));

  VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  submit.waitSemaphoreInfoCount = nx::cast<u32>(waits.size());
  submit.pWaitSemaphoreInfos = waits.data();
  submit.commandBufferInfoCount = nx::cast<u32>(cmd_infos.size());
  submit.pCommandBufferInfos = cmd_infos.data();
  submit.signalSemaphoreInfoCount = nx::cast<u32>(signals.size());
  submit.pSignalSemaphoreInfos = signals.data();
  if (NX_VK_DEVICE_CHECK(
          *this,
          vkQueueSubmit2KHR(m_queues[nx::cast<usize>(QueueType::Graphics)], 1,
                            &submit, VK_NULL_HANDLE)) != VK_SUCCESS) {
    let_go();
    return false;
  }

  // One present for every surface, so FIFO waits once per frame rather than
  // once per window.
  nx::small_vector<VkSwapchainKHR, 4> chains;
  nx::small_vector<u32, 4> indices;
  nx::small_vector<VkSemaphore, 4> ready;
  nx::small_vector<VulkanSwapchain *, 4> presented;
  for (VulkanSwapchain *const sc : acquired) {
    if (!sc->presents())
      continue;
    chains.push_back(sc->swapchain);
    indices.push_back(sc->image_index);
    ready.push_back(sc->release_semaphores[sc->image_index]);
    presented.push_back(sc);
  }
  if (!chains.empty()) {
    nx::small_vector<VkResult, 4> results(chains.size(), VK_SUCCESS);
    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = nx::cast<u32>(ready.size());
    present.pWaitSemaphores = ready.data();
    present.swapchainCount = nx::cast<u32>(chains.size());
    present.pSwapchains = chains.data();
    present.pImageIndices = indices.data();
    present.pResults = results.data();
    {
      NX_PROFILE_ZONE("vk::present");
      nx::fiber::run_blocking([&] {
        (void)vkQueuePresentKHR(m_queues[nx::cast<usize>(QueueType::Graphics)],
                                &present);
      });
    }
    for (usize i = 0; i < presented.size(); ++i) {
      VulkanSwapchain &sc = *presented[i];
      const VkResult result = results[i];
      if (result == VK_ERROR_OUT_OF_DATE_KHR)
        sc.stale = true;
      else if (result == VK_SUBOPTIMAL_KHR)
        sc.stale = surface_resized(m_physical_device, sc.surface, sc.extent);
      else if (result != VK_SUCCESS) {
        NX_VK_DEVICE_CHECK(*this, result);
        sc.stale = true;
      }
    }
  }

#if defined(VK_USE_PLATFORM_WIN32_KHR)
  for (VulkanSwapchain *const sc : acquired) {
    if (!sc->composed())
      continue;
    sc->composed_serials[sc->image_index] = m_timeline_value;
    NX_PROFILE_ZONE("vk::present_composed");
    const bool vsync = sc->present_mode == VK_PRESENT_MODE_FIFO_KHR ||
                       sc->present_mode == VK_PRESENT_MODE_FIFO_RELAXED_KHR;
    bool shown = false;
    nx::fiber::run_blocking([&] {
      shown = sc->composition->present(sc->image_index, m_timeline_value,
                                       vsync);
    });
    if (!shown)
      sc->stale = true;
  }
#endif
  let_go();

  // Nothing presented to pace the frame, so a frame drawn only offscreen is
  // finished before the next begins, as a reader of it expects.
  bool finished = true;
  if (offscreen_only)
    finished = NX_VK_DEVICE_CHECK(*this, wait_timeline(m_timeline_value)) ==
               VK_SUCCESS;
  m_frame_index = (m_frame_index + 1) % m_frames_in_flight;
  return finished;
}

#if defined(VK_USE_PLATFORM_WIN32_KHR)

bool VulkanDevice::create_composed_swapchain(VulkanSwapchain &sc,
                                             const SwapchainDesc &desc) {
  if (!m_routes.win32_interop || (m_swapchain_format != Format::BGRA8_UNORM &&
                                  m_swapchain_format != Format::RGBA8_UNORM))
    return false;

  VkPhysicalDeviceIDProperties id{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
  VkPhysicalDeviceProperties2 props{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  props.pNext = &id;
  vkGetPhysicalDeviceProperties2(m_physical_device, &props);
  if (id.deviceLUIDValid != VK_TRUE)
    return false;

  auto presenter = nx::make_unique<CompositionPresenter>();
  if (!presenter->create(desc.window.window, id.deviceLUID, desc.width,
                         desc.height, m_frames_in_flight,
                         m_swapchain_format == Format::RGBA8_UNORM))
    return false;
  sc.composition = std::move(presenter);

  const auto import_fence = [&](VkSemaphore &out, void *const handle,
                                const nx::string_view name) {
    if (handle == nullptr)
      return false;
    VkSemaphoreTypeCreateInfo type{
        VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    info.pNext = &type;
    bool ok = vkCreateSemaphore(m_device, &info, nullptr, &out) == VK_SUCCESS;
    if (ok) {
      VkImportSemaphoreWin32HandleInfoKHR import{
          VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
      import.semaphore = out;
      import.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
      import.handle = handle;
      ok = vkImportSemaphoreWin32HandleKHR(m_device, &import) == VK_SUCCESS;
    }
    CloseHandle(handle);
    if (ok)
      set_object_name(vk_handle_value(out), VK_OBJECT_TYPE_SEMAPHORE, name);
    return ok;
  };
  if (!import_fence(sc.composed_rendered,
                    sc.composition->rendered_fence_handle(),
                    "composition rendered") ||
      !import_fence(sc.composed_copied, sc.composition->copied_fence_handle(),
                    "composition copied") ||
      !import_composed_images(sc, desc.width, desc.height)) {
    // What is left is the surface, for an ordinary swapchain to try.
    release_composition(sc);
    return false;
  }

  // The surface was only asked what it could composite; DirectComposition
  // owns the window's content from here.
  vkDestroySurfaceKHR(m_instance, sc.surface, nullptr);
  sc.surface = VK_NULL_HANDLE;
  sc.extent = {desc.width, desc.height};
  sc.alpha = SurfaceAlpha::Premultiplied;
  sc.image_index = 0;
  nx::logd(nx::log::rhi,
           "swapchain: {}x{} through DirectComposition, {} images, {}",
           desc.width, desc.height, sc.composition->image_count(),
           to_string(m_swapchain_format));
  return true;
}

bool VulkanDevice::import_composed_images(VulkanSwapchain &sc, const u32 width,
                                          const u32 height) {
  constexpr VkExternalMemoryHandleTypeFlagBits handle_type =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
  const VkFormat format = to_vk_format(m_swapchain_format);
  const u32 count = sc.composition->image_count();
  sc.composed_serials.clear();
  for (u32 i = 0; i < count; ++i)
    sc.composed_serials.push_back(0);

  for (u32 i = 0; i < count; ++i) {
    void *const handle = sc.composition->image_handle(i);
    if (handle == nullptr)
      return false;

    VkExternalMemoryImageCreateInfo external{
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    external.handleTypes = handle_type;
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.pNext = &external;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {width, height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    const auto give_up = [&] {
      CloseHandle(handle);
      if (view != VK_NULL_HANDLE)
        vkDestroyImageView(m_device, view, nullptr);
      if (image != VK_NULL_HANDLE)
        vkDestroyImage(m_device, image, nullptr);
      if (memory != VK_NULL_HANDLE)
        vkFreeMemory(m_device, memory, nullptr);
      return false;
    };
    if (vkCreateImage(m_device, &info, nullptr, &image) != VK_SUCCESS)
      return give_up();

    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(m_device, image, &requirements);
    VkMemoryWin32HandlePropertiesKHR handle_props{
        VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR};
    if (vkGetMemoryWin32HandlePropertiesKHR(m_device, handle_type, handle,
                                            &handle_props) != VK_SUCCESS)
      return give_up();
    const u32 types = requirements.memoryTypeBits & handle_props.memoryTypeBits;
    if (types == 0)
      return give_up();
    u32 type_index = 0;
    while ((types & (1u << type_index)) == 0)
      ++type_index;

    VkMemoryDedicatedAllocateInfo dedicated{
        VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.image = image;
    VkImportMemoryWin32HandleInfoKHR import{
        VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
    import.pNext = &dedicated;
    import.handleType = handle_type;
    import.handle = handle;
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.pNext = &import;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = type_index;
    if (vkAllocateMemory(m_device, &allocate, nullptr, &memory) != VK_SUCCESS ||
        vkBindImageMemory(m_device, image, memory, 0) != VK_SUCCESS)
      return give_up();

    VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(m_device, &view_info, nullptr, &view) != VK_SUCCESS)
      return give_up();
    // Importing an NT handle leaves it open; the allocation keeps its own
    // reference.
    CloseHandle(handle);

    set_object_name(vk_handle_value(image), VK_OBJECT_TYPE_IMAGE,
                    nx::format("composition image {}", i));
    VulkanTexture texture;
    texture.image = image;
    texture.rtv = view;
    texture.format = m_swapchain_format;
    texture.width = width;
    texture.height = height;
    texture.usage = TextureUsage::RenderTarget | TextureUsage::CopySrc;
    texture.state = ResourceState::Undefined;
    texture.owns_image = false;
    sc.backbuffers.push_back(m_textures.insert(std::move(texture)));
    sc.composed_memory.push_back(memory);
  }
  return true;
}

void VulkanDevice::release_composed_images(VulkanSwapchain &sc) noexcept {
  for (const TextureHandle handle : sc.backbuffers) {
    if (const VulkanTexture *texture = m_textures.try_get(handle)) {
      vkDestroyImageView(m_device, texture->rtv, nullptr);
      vkDestroyImage(m_device, texture->image, nullptr);
    }
    m_textures.erase(handle);
  }
  sc.backbuffers.clear();
  for (const VkDeviceMemory memory : sc.composed_memory)
    vkFreeMemory(m_device, memory, nullptr);
  sc.composed_memory.clear();
  sc.composed_serials.clear();
}

void VulkanDevice::release_composition(VulkanSwapchain &sc) noexcept {
  // The images of a surface's own swapchain are not ours to destroy.
  if (!sc.composed())
    return;
  release_composed_images(sc);
  for (VkSemaphore *const semaphore :
       {&sc.composed_rendered, &sc.composed_copied}) {
    if (*semaphore != VK_NULL_HANDLE)
      vkDestroySemaphore(m_device, *semaphore, nullptr);
    *semaphore = VK_NULL_HANDLE;
  }
  sc.composition.reset();
}

void VulkanDevice::acquire_composed_image(VulkanSwapchain &sc,
                                          const VkCommandBuffer cmd) noexcept {
  // An image D3D11 has not copied from yet was never released to it.
  if (sc.composed_serials[sc.image_index] == 0)
    return;
  VulkanTexture *const image = m_textures.try_get(sc.backbuffers[sc.image_index]);
  if (image == nullptr)
    return;

  VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
  barrier.dstQueueFamilyIndex =
      m_queue_families[nx::cast<usize>(QueueType::Graphics)];
  barrier.image = image->image;
  barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.imageMemoryBarrierCount = 1;
  dependency.pImageMemoryBarriers = &barrier;
  vkCmdPipelineBarrier2KHR(cmd, &dependency);
  // The frame redraws the whole image, so its old contents may go.
  image->state = ResourceState::Undefined;
}

#endif

} // namespace nxe::rhi
