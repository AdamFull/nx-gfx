#pragma once

/**
 * @file rhi.h
 * @brief Umbrella header for the render hardware interface (namespace
 * nxe::rhi).
 *
 * Include this to get the whole RHI surface: the device, command recording,
 * resource handles, descriptors and enums. Nothing above rhi/ should include a
 * backend header or a graphics API header directly.
 *
 * See docs/design.md section 5.
 */

#include "rhi/bindless.h"
#include "rhi/descs.h"
#include "rhi/device.h"
#include "rhi/handles.h"
#include "rhi/types.h"
