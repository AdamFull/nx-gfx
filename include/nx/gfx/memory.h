#pragma once

#include "nx/gfx/types.h"

#include <span>

namespace nx::gfx {

class device;

inline constexpr u32 GPU_MOST_HEAPS = 16;
inline constexpr usize GPU_RESOURCE_NAME_BYTES = 46;
/// Events kept for a read before new ones are counted and dropped.
inline constexpr usize GPU_RESOURCE_EVENTS_HELD = 4096;

struct gpu_heap_stats {
  /// What this process has allocated from it, and how much it may.
  u64 used = 0;
  u64 budget = 0;
  u64 size = 0;
  bool device_local = false;
};

struct gpu_resource_totals {
  u64 bytes = 0;
  u32 count = 0;
};

struct gpu_memory_stats {
  gpu_heap_stats heaps[GPU_MOST_HEAPS];
  u32 heap_count = 0;
  gpu_resource_totals buffers;
  /// A texture placed in a memory heap counts there, and has no bytes here.
  gpu_resource_totals textures;
  gpu_resource_totals memory_heaps;
};

/// From any thread, and cheap enough for each frame.
[[nodiscard]] gpu_memory_stats memory_stats(device &dev) noexcept;

enum class gpu_resource_kind : u8 { Buffer, Texture, MemoryHeap };

struct gpu_resource_event {
  /// On nx::os::ticks_ns's clock. A destruction comes when the GPU has finished
  /// with the resource, not when destroy_ was called.
  u64 at_ns = 0;
  u64 bytes = 0;
  /// The same for a resource's creation and destruction, and not used again
  /// while it lives.
  u64 id = 0;
  gpu_resource_kind kind = gpu_resource_kind::Buffer;
  bool created = false;
  u8 name_size = 0;
  /// A longer name is cut; not terminated.
  char name[GPU_RESOURCE_NAME_BYTES] = {};

  [[nodiscard]] nx::string_view text() const noexcept {
    return {name, name_size};
  }
};

/// Starts or stops keeping events, off to begin with. A resource made before it
/// was on is destroyed with no creation to match. Off costs one test each.
void set_resource_events(device &dev, bool on) noexcept;
/// Moves the oldest events to @p out and returns how many. Never waits on the
/// GPU.
usize read_resource_events(device &dev,
                           std::span<gpu_resource_event> out) noexcept;
/// Events that found no room, since the device was made.
[[nodiscard]] u64 resource_events_dropped(const device &dev) noexcept;

} // namespace nx::gfx
