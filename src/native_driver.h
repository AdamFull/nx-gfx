#pragma once

#include "nx/gfx/native.h"

namespace nxe::rhi {

// The names before nx::gfx, until every caller has moved to nx/gfx (G6).
using NativeDriverResolve = nx::gfx::native_driver_resolve;
using NativeDriverView = nx::gfx::native_driver_view;

} // namespace nxe::rhi
