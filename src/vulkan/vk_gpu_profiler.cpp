#include "rendering/rhi/vulkan/vk_gpu_profiler.h"

#if NX_PROFILE_ENABLED

#include "core/foundation/strings/format.h"

#include <tracy/TracyVulkan.hpp>

#include <cstring>
#include <memory>
#include <new>

namespace nxe::rhi {

void VulkanGpuProfiler::init(
    const VkPhysicalDevice physical, const VkDevice device, const VkQueue queue,
    const u32 family, const nx::string_view name,
    const PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT domains,
    const PFN_vkGetCalibratedTimestampsEXT calibrated) {
  static_assert(sizeof(tracy::VkCtxScope) <= ZONE_BYTES);
  static_assert(alignof(tracy::VkCtxScope) <= alignof(std::max_align_t));
  if (m_context != nullptr)
    return;

  VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  pool_info.queueFamilyIndex = family;
  VkCommandPool pool = VK_NULL_HANDLE;
  if (vkCreateCommandPool(device, &pool_info, nullptr, &pool) != VK_SUCCESS)
    return;

  VkCommandBufferAllocateInfo alloc{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  alloc.commandPool = pool;
  alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(device, &alloc, &cmd) == VK_SUCCESS) {
    m_context = tracy::CreateVkContext(physical, device, queue, cmd, domains,
                                       calibrated);
    const nx::string label = nx::format("Vulkan {}", name);
    m_context->Name(label.c_str(), nx::cast<u16>(label.size()));
  }
  vkDestroyCommandPool(device, pool, nullptr);
}

void VulkanGpuProfiler::shutdown() noexcept {
  // The command buffers the open zones were recorded into are gone by now.
  m_depth = 0;
  if (m_context != nullptr)
    tracy::DestroyVkContext(m_context);
  m_context = nullptr;
}

void VulkanGpuProfiler::collect(const VkCommandBuffer cmd) {
  if (m_context != nullptr)
    m_context->Collect(cmd);
}

void VulkanGpuProfiler::begin_zone(const VkCommandBuffer cmd,
                                   const nx::string_view name) {
  if (m_context == nullptr || cmd == VK_NULL_HANDLE)
    return;
  if (m_depth < MAX_DEPTH)
    new (m_zones[m_depth]) tracy::VkCtxScope(
        m_context, __LINE__, __FILE__, sizeof(__FILE__) - 1, __func__,
        std::strlen(__func__), name.data(), name.size(), cmd, true);
  ++m_depth;
}

void VulkanGpuProfiler::end_zone() {
  if (m_depth == 0)
    return;
  --m_depth;
  if (m_depth < MAX_DEPTH)
    std::destroy_at(
        std::launder(reinterpret_cast<tracy::VkCtxScope *>(m_zones[m_depth])));
}

void VulkanGpuProfiler::end_open_zones() {
  while (m_depth > 0)
    end_zone();
}

} // namespace nxe::rhi

#else

namespace nxe::rhi {

void VulkanGpuProfiler::init(VkPhysicalDevice, VkDevice, VkQueue, u32,
                             nx::string_view,
                             PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT,
                             PFN_vkGetCalibratedTimestampsEXT) {}
void VulkanGpuProfiler::shutdown() noexcept {}
void VulkanGpuProfiler::collect(VkCommandBuffer) {}
void VulkanGpuProfiler::begin_zone(VkCommandBuffer, nx::string_view) {}
void VulkanGpuProfiler::end_zone() {}
void VulkanGpuProfiler::end_open_zones() {}

} // namespace nxe::rhi

#endif
