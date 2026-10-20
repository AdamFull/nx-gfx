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
