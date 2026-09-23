#include "rendering/rhi/vulkan/vk_gpu_profiler.h"

#if NX_PROFILE_ENABLED

#include "core/foundation/strings/format.h"

#include <tracy/TracyVulkan.hpp>

#include <cstring>
#include <memory>
#include <new>
#include <time.h>

namespace nxe::rhi {
namespace {

/// Tracy's host-reset context asserts it found a host clock, which release
/// builds compile out, so look first - the way Tracy looks: the first four
/// domains, one clock per platform.
[[nodiscard]] bool has_host_clock(
    const VkPhysicalDevice physical,
    const PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT domains) noexcept {
#if defined(_WIN32) || (defined(__linux__) && defined(CLOCK_MONOTONIC_RAW))
#if defined(_WIN32)
  constexpr VkTimeDomainEXT host = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT;
#else
  constexpr VkTimeDomainEXT host = VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT;
#endif
  u32 count = 0;
  domains(physical, &count, nullptr);
  count = nx::min(count, 4u);
  VkTimeDomainEXT found[4] = {};
  domains(physical, &count, found);
  for (u32 i = 0; i < count; ++i)
    if (found[i] == host)
      return true;
#else
  (void)physical;
  (void)domains;
#endif
  return false;
}

} // namespace

void VulkanGpuProfiler::init(
    const VkPhysicalDevice physical, const VkDevice device, const VkQueue queue,
    const u32 family, const nx::string_view label,
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
    set_label(label);
  }
  vkDestroyCommandPool(device, pool, nullptr);
}

void VulkanGpuProfiler::init_host_reset(
    const VkPhysicalDevice physical, const VkDevice device,
    const nx::string_view label,
    const PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT domains,
    const PFN_vkGetCalibratedTimestampsEXT calibrated) {
  if (m_context != nullptr || vkResetQueryPool == nullptr ||
      domains == nullptr || calibrated == nullptr ||
      !has_host_clock(physical, domains))
    return;
  m_context = tracy::CreateVkContext(physical, device, vkResetQueryPool,
                                     domains, calibrated);
  set_label(label);
}

void VulkanGpuProfiler::set_label(const nx::string_view label) {
  const nx::string full = nx::format("Vulkan {}", label);
  m_context->Name(full.c_str(), nx::cast<u16>(full.size()));
}

void VulkanGpuProfiler::shutdown() noexcept {
  // The command buffers the open zones were recorded into are gone by now.
  m_zones.depth = 0;
  if (m_context != nullptr)
    tracy::DestroyVkContext(m_context);
  m_context = nullptr;
}

void VulkanGpuProfiler::collect(const VkCommandBuffer cmd) {
  if (m_context != nullptr)
    m_context->Collect(cmd);
}

void VulkanGpuProfiler::collect_host(const bool queue_idle) {
  if (m_context == nullptr ||
      (!queue_idle && !tracy::GetProfiler().IsConnected()))
    return;
  m_context->Collect(VK_NULL_HANDLE);
}

void VulkanGpuProfiler::begin_zone(Zones &zones, const VkCommandBuffer cmd,
                                   const nx::string_view name) {
  if (m_context == nullptr || cmd == VK_NULL_HANDLE)
    return;
  if (zones.depth < MAX_DEPTH)
    new (zones.slots[zones.depth]) tracy::VkCtxScope(
        m_context, __LINE__, __FILE__, sizeof(__FILE__) - 1, __func__,
        std::strlen(__func__), name.data(), name.size(), cmd, true);
  ++zones.depth;
}

void VulkanGpuProfiler::end_zone(Zones &zones) {
  if (zones.depth == 0)
    return;
  --zones.depth;
  if (zones.depth < MAX_DEPTH)
    std::destroy_at(std::launder(
        reinterpret_cast<tracy::VkCtxScope *>(zones.slots[zones.depth])));
}

void VulkanGpuProfiler::end_open_zones(Zones &zones) {
  while (zones.depth > 0)
    end_zone(zones);
}

} // namespace nxe::rhi

#else

namespace nxe::rhi {

void VulkanGpuProfiler::init(VkPhysicalDevice, VkDevice, VkQueue, u32,
                             nx::string_view,
                             PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT,
                             PFN_vkGetCalibratedTimestampsEXT) {}
void VulkanGpuProfiler::init_host_reset(
    VkPhysicalDevice, VkDevice, nx::string_view,
    PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT,
    PFN_vkGetCalibratedTimestampsEXT) {}
void VulkanGpuProfiler::shutdown() noexcept {}
void VulkanGpuProfiler::collect(VkCommandBuffer) {}
void VulkanGpuProfiler::collect_host(bool) {}
void VulkanGpuProfiler::begin_zone(Zones &, VkCommandBuffer, nx::string_view) {}
void VulkanGpuProfiler::end_zone(Zones &) {}
void VulkanGpuProfiler::end_open_zones(Zones &) {}

} // namespace nxe::rhi

#endif
