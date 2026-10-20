#pragma once

#include "core/foundation/platform/native_window.h"
#include "nx/gfx/types.h"

namespace nx::gfx {

struct swapchain_desc {
  nx::NativeWindowHandle window;
  u32 width = 0;
  u32 height = 0;
  gfx::present_mode present_mode = gfx::present_mode::Fifo;
  format preferred_format = format::BGRA8_UNORM;
  /// Opaque, or Premultiplied for a see-through window. The surface may not
  /// offer it; swapchain_alpha() reports what was granted.
  surface_alpha alpha = surface_alpha::Opaque;
};

/// Command buffers split off a frame by split_frame. They are submitted in
/// index order, after what the frame recorded before the split and before what
/// it records after.
struct recording_split {
  u32 first = 0;
  u32 count = 0;
};

// The device's owner thread. Every swapchain shares one format, chosen by
// the first.
[[nodiscard]] swapchain_handle create_swapchain(device &dev,
                                                const swapchain_desc &desc);
/// A swapchain with no surface: images of its own that nothing presents.
[[nodiscard]] swapchain_handle
create_offscreen_swapchain(device &dev, u32 width, u32 height, format fmt);
[[nodiscard]] bool resize_swapchain(device &dev, swapchain_handle swapchain,
                                    u32 width, u32 height);
void destroy_swapchain(device &dev, swapchain_handle swapchain) noexcept;
[[nodiscard]] bool offscreen_swapchain(const device &dev,
                                       swapchain_handle swapchain);
[[nodiscard]] format swapchain_format(const device &dev) noexcept;
[[nodiscard]] extent2d swapchain_extent(const device &dev,
                                        swapchain_handle swapchain);
/// What the surface granted of swapchain_desc::alpha.
[[nodiscard]] surface_alpha swapchain_alpha(const device &dev,
                                            swapchain_handle swapchain);
/// The last present found the surface changed.
[[nodiscard]] bool swapchain_stale(const device &dev,
                                   swapchain_handle swapchain);

/// Begins the next frame: waits for its slot, frees what it can and flushes
/// uploads. The frame's list belongs to the device until end_frame; null when
/// no frame could begin. Swapchains are acquired separately, once each.
[[nodiscard]] command_list *begin_frame(device &dev);
[[nodiscard]] acquire_result acquire(device &dev, swapchain_handle swapchain);
[[nodiscard]] texture_handle
current_backbuffer(const device &dev, swapchain_handle swapchain) noexcept;
/// One submit; presents every acquired swapchain together.
[[nodiscard]] bool end_frame(device &dev);

/// A frame with no swapchain that waits for itself (tests, tools).
[[nodiscard]] command_list *begin_headless_frame(device &dev);
[[nodiscard]] bool end_headless_frame(device &dev);

/// Ends what @p frame recorded so far and reserves @p count lists submitted
/// in index order after it; @p frame continues once every one has ended.
/// None where the device cannot record in parallel.
[[nodiscard]] recording_split split_frame(device &dev, command_list &frame,
                                          u32 count);
/// One list of @p split, for one thread; null when it could not begin.
[[nodiscard]] command_list *begin_recording(device &dev, recording_split split,
                                            u32 index);
void end_recording(device &dev, command_list &list);

} // namespace nx::gfx
