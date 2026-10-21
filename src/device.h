#pragma once

#include "core/foundation/containers/small_vector.h"
#include "nx/gfx/frame.h"
#include "rendering/rhi/descs.h"
#include "rendering/rhi/handles.h"
#include "rendering/rhi/types.h"

#if defined(NX_RHI_VULKAN)
#include "rendering/rhi/vulkan/vk_device.h"

namespace nxe::rhi {
using BackendDevice = VulkanDevice;
using BackendCommandContext = VulkanCommandContext;
using Uploader = VulkanUploader;
}

#elif defined(NX_RHI_METAL)
#include "rendering/rhi/metal/mtl_device.h"

namespace nxe::rhi {
using BackendDevice = MetalDevice;
using BackendCommandContext = MetalCommandContext;
using Uploader = MetalUploader;
}

#else
#error "No RHI backend selected; rhi/CMakeLists.txt should define one"
#endif

namespace nx::gfx {

namespace detail {
struct backend_access;
}

class command_list final : private nxe::rhi::BackendCommandContext {
  friend struct detail::backend_access;
};

/// The backend's device, with the command lists it hands out: the frame's,
/// and one for each recording split off it, by its place in the frame's
/// submission order.
class device final : private nxe::rhi::BackendDevice {
  friend struct detail::backend_access;

  command_list m_frame;
  nx::small_vector<nx::unique_ptr<command_list>, 4> m_recordings;
};

namespace detail {

/// How the free functions reach the backend: nothing else can.
struct backend_access {
  [[nodiscard]] static nxe::rhi::BackendDevice &of(device &dev) noexcept {
    return dev;
  }
  [[nodiscard]] static const nxe::rhi::BackendDevice &
  of(const device &dev) noexcept {
    return dev;
  }
  [[nodiscard]] static nxe::rhi::BackendCommandContext &
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

namespace nxe::rhi {
using Device = nx::gfx::device;
using CommandContext = nx::gfx::command_list;
} // namespace nxe::rhi
