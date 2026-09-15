#pragma once

#include "rendering/rhi/descs.h"
#include "rendering/rhi/handles.h"
#include "rendering/rhi/types.h"

#if defined(NX_RHI_VULKAN)
#include "rendering/rhi/vulkan/vk_device.h"

namespace nxe::rhi {
using Device = VulkanDevice;
using CommandContext = VulkanCommandContext;
using Uploader = VulkanUploader;
}

#elif defined(NX_RHI_METAL)
#include "rendering/rhi/metal/mtl_device.h"

namespace nxe::rhi {
using Device = MetalDevice;
using CommandContext = MetalCommandContext;
using Uploader = MetalUploader;
}

#else
#error "No RHI backend selected; rhi/CMakeLists.txt should define one"
#endif
