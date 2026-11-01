#pragma once

#include "nx/gfx/markers.h"
#include "src/vulkan/vk_common.h"

#include <atomic>
#include <span>

namespace nx::gfx::detail {

/// Debug groups as timestamps, for whoever reads them with read_gpu_markers.
/// Apart from VulkanGpuProfiler, which feeds Tracy: this works in every build
/// but Shipping's and names no profiler.
class VulkanGpuMarkers {
public:
  /// Groups a frame can time; two timestamps each.
  static constexpr u32 MOST_A_FRAME = 1024;
  static constexpr u32 MAX_DEPTH = GPU_MARKER_MOST_DEPTH;
  static constexpr usize READY_MARKERS = GPU_MARKERS_HELD;

  struct Slot {
    struct Record {
      u16 depth = 0;
      u8 name_size = 0;
      char name[GPU_MARKER_NAME_BYTES] = {};
    };

    VkQueryPool pool = VK_NULL_HANDLE;
    nx::vector<Record> records;
    u32 open[MAX_DEPTH] = {};
    u32 depth = 0;
    /// Queries this frame wrote, to read when the slot comes round.
    u32 written = 0;
    /// Queries read already, which need a reset before they are written again.
    u32 to_reset = 0;
    bool ever_reset = false;
    bool submitted = false;
    u64 submitted_ns = 0;
  };

  /// @p calibrated reads the device's clock with its error, or is null.
  void init(VkDevice device, u32 frames, f32 period_ns, u32 valid_bits,
            PFN_vkGetCalibratedTimestampsEXT calibrated, bool inside_passes);
  void shutdown() noexcept;

  void set_enabled(bool on) noexcept {
    m_wanted.store(on, std::memory_order_release);
  }

  /// Frame slot @p frame is free: its last frame has finished. Reads what that
  /// frame timed, and readies the slot on @p cmd, outside a render pass. The
  /// context to time on, or null while off.
  [[nodiscard]] Slot *open_frame(u32 frame, VkCommandBuffer cmd);
  void frame_submitted(u32 frame) noexcept;

  void push(Slot &slot, VkCommandBuffer cmd, nx::string_view name,
            bool in_render_pass);
  void pop(Slot &slot, VkCommandBuffer cmd) noexcept;
  /// Closes whatever a command buffer left open, before it ends.
  void end_open(Slot &slot, VkCommandBuffer cmd) noexcept;

  usize read(std::span<gpu_marker> out) noexcept;
  [[nodiscard]] gpu_marker_stats stats() const noexcept;

private:
  void collect(Slot &slot);
  void calibrate() noexcept;

  VkDevice m_device = VK_NULL_HANDLE;
  f64 m_period_ns = 0.0;
  u64 m_valid_mask = ~u64{0};
  PFN_vkGetCalibratedTimestampsEXT m_calibrated = nullptr;
  bool m_inside_passes = true;
  std::atomic<bool> m_wanted{false};
  nx::vector<Slot> m_slots;
  nx::vector<gpu_marker> m_ready;
  usize m_ready_first = 0;
  usize m_ready_count = 0;
  /// Host nanoseconds minus device nanoseconds.
  i64 m_offset_ns = 0;
  bool m_have_offset = false;
  u64 m_calibrated_at_ns = 0;
  gpu_marker_stats m_stats;
  nx::vector<u64> m_ticks;
};

} // namespace nx::gfx::detail
