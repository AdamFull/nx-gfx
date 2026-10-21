#include "nx/gfx/vulkan.h"
#include "rendering/rhi/device.h"

namespace nx::gfx::vk {

namespace {

auto &be(device &dev) noexcept { return detail::backend_access::of(dev); }
const auto &be(const device &dev) noexcept {
  return detail::backend_access::of(dev);
}

} // namespace

VkPhysicalDevice physical_device(const device &dev) noexcept {
  return be(dev).vk_physical_device();
}

VkDevice logical_device(const device &dev) noexcept {
  return be(dev).vk_device();
}

u32 graphics_family(const device &dev) noexcept {
  return be(dev).graphics_family();
}

VkResult submit_graphics(device &dev,
                         const std::span<const VkSubmitInfo> submits,
                         const VkFence fence) noexcept {
  const nx::scoped_lock<nx::mutex> held(
      be(dev).queue_lock(queue_type::Graphics));
  return vkQueueSubmit(be(dev).graphics_queue(), nx::cast<u32>(submits.size()),
                       submits.data(), fence);
}

VkResult check(const device &dev, const VkResult result, const char *const call,
               const char *const file, const i32 line) noexcept {
  return be(dev).check_result(result, call, file, line);
}

} // namespace nx::gfx::vk
