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

} // namespace nx::gfx
