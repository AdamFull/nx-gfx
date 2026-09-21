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

/// One queue's timeline in Tracy. On the graphics queue every debug group a
/// command context opens is a zone, so render passes, compute passes and graph
/// passes show up without their callers knowing. Empty without the profiler.
class VulkanGpuProfiler {
public:
  /// Needs the queue idle: Tracy calibrates with a blocking submit, and resets
  /// queries in command buffers, which a transfer-only queue cannot record.
  /// `calibrated` names the calibrated-timestamps entry points, or is null.
  void init(VkPhysicalDevice physical, VkDevice device, VkQueue queue,
            u32 family, nx::string_view label,
            PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT domains,
            PFN_vkGetCalibratedTimestampsEXT calibrated);
  /// For any queue: resets queries from the host instead. Needs hostQueryReset
  /// enabled and a host clock to calibrate against; stays off without one.
  void
  init_host_reset(VkPhysicalDevice physical, VkDevice device,
                  nx::string_view label,
                  PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT domains,
                  PFN_vkGetCalibratedTimestampsEXT calibrated);
  void shutdown() noexcept;

  /// Reads finished zones and resets their queries, so it belongs at the top
  /// of a command buffer, outside any render pass.
  void collect(VkCommandBuffer cmd);
  /// collect() for a host-reset context. With no viewer connected Tracy resets
  /// every query, so that waits until nothing recorded is left to run.
  void collect_host(bool queue_idle);

  void begin_zone(VkCommandBuffer cmd, nx::string_view name);
  void end_zone();
  /// Closes whatever a frame left open; a zone cannot span command buffers.
  void end_open_zones();

  /// A zone for one scope. Its command buffer must still be recording when the
  /// scope ends; a null one times nothing.
  class Scope {
  public:
    Scope(VulkanGpuProfiler &profiler, const VkCommandBuffer cmd,
          const nx::string_view name)
        : m_profiler(profiler), m_open(cmd != VK_NULL_HANDLE) {
      if (m_open)
        m_profiler.begin_zone(cmd, name);
    }
    ~Scope() {
      if (m_open)
        m_profiler.end_zone();
    }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

  private:
    VulkanGpuProfiler &m_profiler;
    bool m_open;
  };

#if NX_PROFILE_ENABLED
private:
  static constexpr u32 MAX_DEPTH = 16;
  static constexpr usize ZONE_BYTES = 32;

  void set_label(nx::string_view label);

  tracy::VkCtx *m_context = nullptr;
  /// Past MAX_DEPTH, zones are counted but not recorded.
  u32 m_depth = 0;
  alignas(std::max_align_t) std::byte m_zones[MAX_DEPTH][ZONE_BYTES];
#endif
};

} // namespace nxe::rhi
