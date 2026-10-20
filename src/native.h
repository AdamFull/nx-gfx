#pragma once

#include "nx/gfx/native.h"

namespace nxe::rhi {

// The names before nx::gfx, until every caller has moved to nx/gfx (G6).
using NativeApi = nx::gfx::native_api;
using NativeDevice = nx::gfx::native_device;
using NativeTexture = nx::gfx::native_texture;
using NativeNames = nx::gfx::native_names;
using NativeQueue = nx::gfx::native_queue;
using NativeDeviceClient = nx::gfx::native_device_client;

} // namespace nxe::rhi
