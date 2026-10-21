#pragma once

#include "nx/gfx/types.h"

#include <volk.h>

#include <span>

// Typed native access for libraries that work on the Vulkan objects
// themselves (video decode, vendor SDKs). Opt-in: only the Vulkan backend
// defines it.
namespace nx::gfx::vk {

[[nodiscard]] VkPhysicalDevice physical_device(const device &dev) noexcept;
[[nodiscard]] VkDevice logical_device(const device &dev) noexcept;
[[nodiscard]] u32 graphics_family(const device &dev) noexcept;
/// The distinct families of the device's own queues. An image made outside
/// the library is shared (VK_SHARING_MODE_CONCURRENT) with all of them.
[[nodiscard]] std::span<const u32> queue_families(const device &dev) noexcept;

/// An image made outside the library: 2D, one mip and layer, with the
/// usage @ref usage implies and shared with queue_families(). It is sampled
/// and may be copied from, never uploaded into.
struct wrapped_image {
  nx::string_view name;
  VkImage image = VK_NULL_HANDLE;
  gfx::format format = gfx::format::Unknown;
  u32 width = 0;
  u32 height = 0;
  texture_usage_flags usage = texture_usage::Sampled;
  resource_state state = resource_state::Undefined;
  /// Runs once the handle's retirement is over, on whichever thread collects
  /// the device's garbage; it must not call into the device.
  void (*release)(void *user, VkImage image) noexcept = nullptr;
  void *user = nullptr;
};

/// Adopts @p image under a handle and a bindless slot, with views of the
/// device's own. destroy_texture retires it as any texture, then calls
/// release instead of freeing the image.
[[nodiscard]] texture_handle wrap(device &dev, const wrapped_image &image);

/// Submits to the graphics queue, which the device's frames submit to, under
/// the same lock: Vulkan requires a queue's submits to be serialized.
[[nodiscard]] VkResult submit_graphics(device &dev,
                                       std::span<const VkSubmitInfo> submits,
                                       VkFence fence) noexcept;

/// Passes a native call's result through the device as its own calls do, so
/// a lost device is reported once, with the call that saw it.
VkResult check(const device &dev, VkResult result, const char *call,
               const char *file, i32 line) noexcept;

} // namespace nx::gfx::vk

#define NX_GFX_VK_CHECK(dev, expr)                                             \
  ::nx::gfx::vk::check((dev), (expr), #expr, __FILE__, __LINE__)

#define NX_GFX_VK_TRY(dev, expr)                                               \
  do {                                                                         \
    if (NX_GFX_VK_CHECK(dev, expr) != VK_SUCCESS)                              \
      return false;                                                            \
  } while (false)
