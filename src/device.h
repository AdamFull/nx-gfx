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

class command_list final : public nxe::rhi::BackendCommandContext {};

/// The backend's device, with the command lists it hands out: the frame's,
/// and one for each recording split off it, by its place in the frame's
/// submission order.
class device final : public nxe::rhi::BackendDevice {
  friend command_list *begin_frame(device &dev);
  friend command_list *begin_headless_frame(device &dev);
  friend recording_split split_frame(device &dev, command_list &frame,
                                     u32 count);
  friend command_list *begin_recording(device &dev, recording_split split,
                                       u32 index);

  command_list m_frame;
  nx::small_vector<nx::unique_ptr<command_list>, 4> m_recordings;
};

} // namespace nx::gfx

namespace nxe::rhi {
using Device = nx::gfx::device;
using CommandContext = nx::gfx::command_list;
} // namespace nxe::rhi
