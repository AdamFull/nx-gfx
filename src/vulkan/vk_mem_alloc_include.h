#pragma once

// The one place this repo includes <vk_mem_alloc.h> - route every other
// inclusion through this instead of the raw header directly. On Linux it
// pulls in vulkan.h with VK_USE_PLATFORM_XLIB_KHR, which drags in
// X11/Xlib.h - a header that #defines None as 0L (and Bool/True/False/
// Status/Complex/Success/Above/Below as similarly raw macros) with no
// namespacing at all. Left alone, every later `SomeEnum::None` in the same
// translation unit silently becomes `SomeEnum::0L`; this is what broke
// gfx::texture_usage::None/gfx::buffer_usage::None/VulkanFaultRoute::None in
// vk_device.h, and later TileFlags::Above in tileset.h. Undefining them
// once here, immediately after the include that can pull X11 in, means no
// other header needs to know this is why. A no-op on Windows/macOS, where
// X11 was never pulled in: #undef on a name that isn't a macro does
// nothing, not an error.
#include <vk_mem_alloc.h>

#undef None
#undef Bool
#undef True
#undef False
#undef Status
#undef Complex
#undef Success
#undef Above
#undef Below
