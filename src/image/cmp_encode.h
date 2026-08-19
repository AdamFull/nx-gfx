#pragma once

/**
 * @file cmp_encode.h
 * @brief BC1-7 block encoding via Compressonator's CMP_Core, behind a clean
 * interface. The implementation TU keeps CMP_Core's headers out of image.cpp.
 */

#include "core/foundation/core/foundation.h"

namespace nxe::rhi {

enum class CmpBc { BC1, BC3, BC4, BC5, BC7 };

/// Encodes a mip of tightly-packed RGBA8 pixels to @p fmt blocks. @p out holds
/// ceil(w/4)*ceil(h/4)*block_bytes (8 for BC1/BC4, 16 otherwise). @p row_pitch
/// is in bytes.
void cmp_encode_bc(CmpBc fmt, const u8 *rgba8, u32 width, u32 height,
                   u32 row_pitch, u8 *out);

/// Encodes a mip of tightly-packed RGBA32F pixels to unsigned BC6H (16 bytes
/// per 4x4 block). @p row_pitch is in bytes.
void cmp_encode_bc6h(const f32 *rgba32f, u32 width, u32 height, u32 row_pitch,
                     u8 *out);

} // namespace nxe::rhi
