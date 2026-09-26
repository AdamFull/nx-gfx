#pragma once

#include "core/foundation/core/foundation.h"

namespace nxe::rhi {

/// Puts frames rendered by Vulkan on screen through a premultiplied
/// DirectComposition swap chain. It serves Win32 surfaces whose Vulkan driver
/// offers no premultiplied composite alpha (Intel's), so a transparent window
/// still composites by its alpha.
///
/// D3D11 owns the images Vulkan renders into and both fences: `rendered`,
/// which Vulkan signals when a frame is done, and `copied`, which this signals
/// once the frame is in the swap chain and its image may be written again.
/// Every call comes from whoever owns the Vulkan swapchain, one at a time.
class CompositionPresenter {
public:
  CompositionPresenter();
  ~CompositionPresenter();

  CompositionPresenter(const CompositionPresenter &) = delete;
  CompositionPresenter &operator=(const CompositionPresenter &) = delete;

  /// `luid` names the adapter the Vulkan device runs on; the images must live
  /// on the same one. `rgba` picks RGBA8 over BGRA8.
  [[nodiscard]] bool create(void *hwnd, const u8 (&luid)[8], u32 width,
                            u32 height, u32 image_count, bool rgba);
  /// Remakes the images at the new size; their handles must be imported
  /// again. The GPU must be done with the old ones.
  [[nodiscard]] bool resize(u32 width, u32 height);

  [[nodiscard]] u32 image_count() const noexcept;
  /// New NT handles, each owned by the caller: import, then close it.
  [[nodiscard]] void *image_handle(u32 index) const;
  [[nodiscard]] void *rendered_fence_handle() const;
  [[nodiscard]] void *copied_fence_handle() const;

  /// Queues, on the GPU, a wait for `rendered` to reach `value`, the copy of
  /// image `index` into the swap chain, and `copied` reaching `value`; then
  /// presents. May block for the display when `vsync` is set.
  [[nodiscard]] bool present(u32 index, u64 value, bool vsync);

private:
  struct State;
  nx::unique_ptr<State> m_state;
};

} // namespace nxe::rhi
