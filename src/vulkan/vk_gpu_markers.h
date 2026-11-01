#pragma once

#include "core/foundation/containers/small_vector.h"
#include "core/foundation/threading/sync.h"
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
  /// Lists one frame can time.
  static constexpr u32 MOST_LISTS = 32;
  /// Copies waiting for their queue.
  static constexpr u32 COPIES_WAITING = 256;
  static constexpr u32 NO_COPY = ~u32{0};

  struct Slot;

  /// What one command buffer times. One thread at a time.
  struct List {
    struct Record {
      u32 query = 0;
      u16 depth = 0;
      u8 name_size = 0;
      char name[GPU_MARKER_NAME_BYTES] = {};
    };

    Slot *slot = nullptr;
    /// Where its command buffer comes in the frame's submit.
    u32 position = 0;
    nx::vector<Record> records;
    u32 open[MAX_DEPTH] = {};
    u32 depth = 0;
  };

  struct Slot {
    VkQueryPool pool = VK_NULL_HANDLE;
    nx::vector<List> lists;
    std::atomic<u32> lists_used{0};
    /// Queries this frame took, which may run past the pool's end.
    std::atomic<u32> next_query{0};
    /// Queries read already, which need a reset before they are written again.
    u32 to_reset = 0;
    bool ever_reset = false;
    bool submitted = false;
    u64 submitted_ns = 0;
  };

  /// @p calibrated reads the device's clock with its error, or is null. A
  /// @p copy_valid_bits of 0 means copies are not timed.
  void init(VkDevice device, u32 frames, f32 period_ns, u32 valid_bits,
            u32 copy_valid_bits, PFN_vkGetCalibratedTimestampsEXT calibrated,
            bool inside_passes);
  void shutdown() noexcept;

  void set_enabled(bool on) noexcept {
    m_wanted.store(on, std::memory_order_release);
  }

  /// Frame slot @p frame is free: its last frame has finished. Reads what that
  /// frame timed, and readies the slot on @p cmd, outside a render pass. The
  /// list to time the frame's first command buffer on, or null while off.
  [[nodiscard]] List *open_frame(u32 frame, VkCommandBuffer cmd);
  /// Says the frame is being submitted, and that it times nothing more.
  void frame_submitted(u32 frame) noexcept;
  /// A list for another command buffer of the open frame, from any thread, or
  /// null while off or past MOST_LISTS.
  [[nodiscard]] List *add_list(u32 position) noexcept;

  void push(List &list, VkCommandBuffer cmd, nx::string_view name,
            bool in_render_pass);
  void pop(List &list, VkCommandBuffer cmd) noexcept;
  /// Closes whatever a command buffer left open, before it ends.
  void end_open(List &list, VkCommandBuffer cmd) noexcept;

  /// Times a copy on the copy queue, whoever records it. NO_COPY when it is not
  /// timed, which copy_end takes too.
  [[nodiscard]] u32 copy_begin(VkCommandBuffer cmd,
                               nx::string_view name) noexcept;
  void copy_end(VkCommandBuffer cmd, u32 copy) noexcept;

  class CopyScope {
  public:
    CopyScope(VulkanGpuMarkers &markers, const VkCommandBuffer cmd,
              const nx::string_view name) noexcept
        : m_markers(markers), m_cmd(cmd),
          m_copy(markers.copy_begin(cmd, name)) {}
    ~CopyScope() { m_markers.copy_end(m_cmd, m_copy); }
    CopyScope(const CopyScope &) = delete;
    CopyScope &operator=(const CopyScope &) = delete;

  private:
    VulkanGpuMarkers &m_markers;
    VkCommandBuffer m_cmd;
    u32 m_copy;
  };

  usize read(std::span<gpu_marker> out) noexcept;
  [[nodiscard]] gpu_marker_stats stats() const noexcept;

private:
  struct Copy {
    u8 name_size = 0;
    char name[GPU_MARKER_NAME_BYTES] = {};
  };

  void collect(Slot &slot);
  void collect_copies() noexcept;
  void calibrate() noexcept;
  [[nodiscard]] gpu_marker &next_ready() noexcept;
  void place(gpu_marker &marker, u64 begin_ticks, u64 end_ticks,
             u32 valid_bits) const noexcept;

  VkDevice m_device = VK_NULL_HANDLE;
  f64 m_period_ns = 0.0;
  u32 m_valid_bits = 64;
  u32 m_copy_bits = 0;
  PFN_vkGetCalibratedTimestampsEXT m_calibrated = nullptr;
  bool m_inside_passes = true;
  std::atomic<bool> m_wanted{false};
  nx::vector<nx::unique_ptr<Slot>> m_slots;
  /// The frame recording now, for lists added to it.
  std::atomic<Slot *> m_open{nullptr};
  nx::vector<gpu_marker> m_ready;
  usize m_ready_first = 0;
  usize m_ready_count = 0;
  /// A moment on both clocks. Times are differences from it, since a device's
  /// ticks can be too large for a double to keep to the nanosecond.
  u64 m_anchor_ticks = 0;
  u64 m_anchor_ns = 0;
  bool m_have_anchor = false;
  u64 m_calibrated_at_ns = 0;
  u32 m_frame_number = 0;
  u64 m_frames_read = 0;
  std::atomic<u64> m_groups_dropped{0};
  u64 m_markers_dropped = 0;
  nx::vector<u64> m_ticks;

  nx::mutex m_copy_lock;
  VkQueryPool m_copy_pool = VK_NULL_HANDLE;
  nx::vector<Copy> m_copies;
  u32 m_copies_first = 0;
  u32 m_copies_count = 0;
};

} // namespace nx::gfx::detail
