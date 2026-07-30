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

#include "rendering/rhi/bindless.h"
#include "rendering/rhi/descs.h"
#include "rendering/rhi/device.h"
#include "rendering/rhi/handles.h"
#include "rendering/rhi/image/image.h"
#include "rendering/rhi/types.h"
#include "rendering/rhi/upload.h"
#include "rendering/rhi/upload_ring.h"
