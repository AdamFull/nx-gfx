#include "nx/gfx/vulkan.h"
#include "rendering/rhi/device.h"

namespace nx::gfx::vk {

VkPhysicalDevice physical_device(const device &dev) noexcept {
  return dev.vk_physical_device();
}

VkDevice logical_device(const device &dev) noexcept { return dev.vk_device(); }

u32 graphics_family(const device &dev) noexcept {
  return dev.graphics_family();
}

VkResult submit_graphics(device &dev,
                         const std::span<const VkSubmitInfo> submits,
                         const VkFence fence) noexcept {
  const nx::scoped_lock<nx::mutex> held(dev.queue_lock(queue_type::Graphics));
  return vkQueueSubmit(dev.graphics_queue(), nx::cast<u32>(submits.size()),
                       submits.data(), fence);
}

VkResult check(const device &dev, const VkResult result, const char *const call,
               const char *const file, const i32 line) noexcept {
  return dev.check_result(result, call, file, line);
}

} // namespace nx::gfx::vk
