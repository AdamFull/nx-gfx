#pragma once

#if !defined(__OBJC__)
#error "mtl_gpu_crash.h is Objective-C++ only"
#endif

#import <Metal/Metal.h>

#include "core/foundation/strings/utf8_string.h"
#include "core/rendering/rhi/gpu_crash.h"

namespace nxe::rhi {

struct MetalDeviceState;

void configure_gpu_crash_reporting(MetalDeviceState &state, GpuCrashMode mode,
                                   nx::string_view device_name);
[[nodiscard]] GpuDeviceStatus
gpu_device_status(const MetalDeviceState &state) noexcept;
[[nodiscard]] id<MTLCommandBuffer> make_command_buffer(MetalDeviceState &state,
                                                       nx::string_view label,
                                                       u64 frame = 0);

} // namespace nxe::rhi
