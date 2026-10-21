#pragma once

#if !defined(__OBJC__)
#error "mtl_gpu_crash.h is Objective-C++ only"
#endif

#import <Metal/Metal.h>

#include "core/foundation/strings/utf8_string.h"
#include "nx/gfx/crash.h"

namespace nx::gfx::detail {

struct MetalDeviceState;

void configure_gpu_crash_reporting(MetalDeviceState &state, gfx::gpu_crash_mode mode,
                                   nx::string_view device_name);
[[nodiscard]] gfx::gpu_device_status
gpu_device_status(const MetalDeviceState &state) noexcept;
[[nodiscard]] id<MTLCommandBuffer> make_command_buffer(MetalDeviceState &state,
                                                       nx::string_view label,
                                                       u64 frame = 0);

} // namespace nx::gfx::detail
