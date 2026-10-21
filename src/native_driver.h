#pragma once

#include "nx/gfx/native.h"

namespace nxe::rhi {

// The names before nx::gfx. Only the backend's own sources and projects made
// before nx::gfx (through app/engine.h) use them.
using NativeDriverResolve = nx::gfx::native_driver_resolve;
using NativeDriverView = nx::gfx::native_driver_view;

} // namespace nxe::rhi
