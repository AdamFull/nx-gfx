#pragma once

#include "nx/gfx/types.h"

#include <span>

namespace nx::gfx {

class device;

inline constexpr usize GPU_MARKER_NAME_BYTES = 48;
/// Groups inside this many others go untimed.
inline constexpr u32 GPU_MARKER_MOST_DEPTH = 32;
/// Finished markers kept for a read before the oldest go.
inline constexpr usize GPU_MARKERS_HELD = 8192;

enum class gpu_marker_queue : u8 { Graphics, Copy };

/// A debug group, timed. Its times are on nx::os::ticks_ns's clock.
struct gpu_marker {
  u64 begin_ns = 0;
  u64 end_ns = 0;
  /// Which frame timed it, a number that rises. A copy has none.
  u32 frame = 0;
  /// How many groups it is inside.
  u16 depth = 0;
  u8 name_size = 0;
  gpu_marker_queue queue = gpu_marker_queue::Graphics;
  /// A longer name is cut; not terminated.
  char name[GPU_MARKER_NAME_BYTES] = {};

  [[nodiscard]] nx::string_view text() const noexcept {
    return {name, name_size};
  }
};

struct gpu_marker_stats {
  /// Frames read back.
  u64 frames = 0;
  /// Groups past a frame's room, or nested too deep.
  u64 groups_dropped = 0;
  /// Finished groups that no read took in time.
  u64 markers_dropped = 0;
  /// Times are anchored to the host's clock by reading both at once. Otherwise
  /// each frame is placed by when it was submitted, which can be a frame early.
  bool calibrated = false;
};

// Whoever calls begin_frame. caps().timestamps says whether this does anything.
// Debug groups on the frame's lists are timed, split lists too, in their order
// on the graphics queue, and so is each copy the device makes on its copy queue
// where that queue can time. On a device with memoryless attachments, which
// means a tiling GPU, a group inside a render pass is not timed, since a
// timestamp there says little.

/// Starts or stops timing, from the next begin_frame. Off costs each debug
/// group one test.
void set_gpu_markers(device &dev, bool on) noexcept;

/// Moves finished markers to @p out, oldest first on each queue, and returns
/// how many. A group comes before the groups inside it, and a frame's markers
/// arrive together. Never waits: they arrive when the frame's slot comes round
/// again.
usize read_gpu_markers(device &dev, std::span<gpu_marker> out) noexcept;

[[nodiscard]] gpu_marker_stats gpu_markers_stats(const device &dev) noexcept;

} // namespace nx::gfx
