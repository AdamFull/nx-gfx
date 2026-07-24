#pragma once

/**
 * @file device.h
 * @brief The portable names for the backend's device and command context.
 *
 */

#include "rendering/rhi/descs.h"
#include "rendering/rhi/handles.h"
#include "rendering/rhi/types.h"

#if defined(NX_RHI_VULKAN)
#include "rendering/rhi/vulkan/vk_device.h"

namespace nxe::rhi {
using Device = VulkanDevice;
using CommandContext = VulkanCommandContext;
} // namespace nxe::rhi

#elif defined(NX_RHI_METAL)
#error "Metal backend is not implemented yet"
#else
#error "No RHI backend selected; rhi/CMakeLists.txt should define one"
#endif
