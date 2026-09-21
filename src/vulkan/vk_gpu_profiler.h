#pragma once

#include "core/foundation/diagnostics/profiler.h"
#include "core/foundation/strings/utf8_string_view.h"
#include "rendering/rhi/vulkan/vk_common.h"

#include <cstddef>

namespace tracy {
class VkCtx;
}

namespace nxe::rhi {

enum class VulkanCalibrationRoute : u8 { None, Khr, Ext };

/// Tracy's GPU timeline for the graphics queue. Every debug group a command
/// context opens is a zone, so render passes, compute passes and graph passes
/// show up without their callers knowing. Empty without the profiler.
class VulkanGpuProfiler {
public:
  /// Needs the queue idle: Tracy calibrates with a blocking submit.
  /// `calibrated` names the calibrated-timestamps entry points, or is null.
  void init(VkPhysicalDevice physical, VkDevice device, VkQueue queue,
            u32 family, nx::string_view name,
            PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT domains,
            PFN_vkGetCalibratedTimestampsEXT calibrated);
  void shutdown() noexcept;

  /// Reads finished zones and resets their queries, so it belongs at the top
  /// of a frame's command buffer, outside any render pass.
  void collect(VkCommandBuffer cmd);

  void begin_zone(VkCommandBuffer cmd, nx::string_view name);
  void end_zone();
  /// Closes whatever a frame left open; a zone cannot span command buffers.
  void end_open_zones();

#if NX_PROFILE_ENABLED
private:
  static constexpr u32 MAX_DEPTH = 16;
  static constexpr usize ZONE_BYTES = 32;

  tracy::VkCtx *m_context = nullptr;
  /// Past MAX_DEPTH, zones are counted but not recorded.
  u32 m_depth = 0;
  alignas(std::max_align_t) std::byte m_zones[MAX_DEPTH][ZONE_BYTES];
#endif
};

} // namespace nxe::rhi
