#pragma once

#include "core/foundation/containers/small_vector.h"
#include "nx/gfx/gfx.h"

#if defined(NX_RHI_VULKAN)
#include "src/vulkan/vk_device.h"

namespace nx::gfx::detail {
using BackendDevice = VulkanDevice;
using BackendCommandContext = VulkanCommandContext;
} // namespace nx::gfx::detail

#elif defined(NX_RHI_METAL)
#include "src/metal/mtl_device.h"

namespace nx::gfx::detail {
using BackendDevice = MetalDevice;
using BackendCommandContext = MetalCommandContext;
} // namespace nx::gfx::detail

#else
#error "No RHI backend selected; rhi/CMakeLists.txt should define one"
#endif

namespace nx::gfx {

namespace detail {
struct backend_access;
}

class command_list final : private detail::BackendCommandContext {
  friend struct detail::backend_access;
};

/// The backend's device, with the command lists it hands out: the frame's,
/// and one for each recording split off it, by its place in the frame's
/// submission order.
class device final : private detail::BackendDevice {
  friend struct detail::backend_access;

  command_list m_frame;
  nx::small_vector<nx::unique_ptr<command_list>, 4> m_recordings;
};

namespace detail {

/// How the free functions reach the backend: nothing else can.
struct backend_access {
  [[nodiscard]] static detail::BackendDevice &of(device &dev) noexcept {
    return dev;
  }
  [[nodiscard]] static const detail::BackendDevice &
  of(const device &dev) noexcept {
    return dev;
  }
  [[nodiscard]] static detail::BackendCommandContext &
  of(command_list &cmd) noexcept {
    return cmd;
  }
  [[nodiscard]] static command_list &frame(device &dev) noexcept {
    return dev.m_frame;
  }
  [[nodiscard]] static nx::small_vector<nx::unique_ptr<command_list>, 4> &
  recordings(device &dev) noexcept {
    return dev.m_recordings;
  }
};

} // namespace detail

} // namespace nx::gfx
