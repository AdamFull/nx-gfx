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

#include "core/rendering/rhi/bindless.h"
#include "core/rendering/rhi/descs.h"
#include "core/rendering/rhi/device.h"
#include "core/rendering/rhi/handles.h"
#include "core/rendering/rhi/image/image.h"
#include "core/rendering/rhi/types.h"
#include "core/rendering/rhi/upload.h"
#include "core/rendering/rhi/upload_ring.h"
