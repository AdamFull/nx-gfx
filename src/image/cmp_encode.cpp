#include "core/rendering/rhi/image/cmp_encode.h"

#include <cmp_core.h>

#include <algorithm>
#include <cstring>

namespace nxe::rhi {
namespace {

/// float -> IEEE half bit pattern (BC6H unsigned; negatives/NaN clamp to +0).
u16 to_half(const float f) {
  if (!(f > 0.f))
    return 0;
  u32 x;
  std::memcpy(&x, &f, 4);
  const i32 e = nx::cast<i32>((x >> 23) & 0xFF) - 112;
  const u32 mant = x & 0x007FFFFFu;
  if (((x >> 23) & 0xFF) == 0xFF || e >= 0x1F)
    return 0x7C00; // inf/nan/overflow -> +inf
  if (e <= 0) {
    if (e < -10)
      return 0;
    return nx::cast<u16>((mant | 0x00800000u) >> nx::cast<u32>(14 - e));
  }
  u16 h = nx::cast<u16>((nx::cast<u32>(e) << 10) | (mant >> 13));
  if (mant & 0x1000u)
    ++h;
  return h;
}

/// Gathers a 4x4 RGBA8 block from an image, clamping at the edges.
void gather8(const u8 *px, const u32 rp, const u32 w, const u32 h, const u32 bx,
             const u32 by, u8 out[64]) {
  for (u32 py = 0; py < 4; ++py) {
    const u32 sy = std::min(by * 4u + py, h - 1u);
    for (u32 sx0 = 0; sx0 < 4; ++sx0) {
      const u32 sx = std::min(bx * 4u + sx0, w - 1u);
      std::memcpy(out + (py * 4u + sx0) * 4u,
                  px + nx::cast<u64>(sy) * rp + nx::cast<u64>(sx) * 4u, 4u);
    }
  }
}

} // namespace

void cmp_encode_bc(const CmpBc fmt, const u8 *rgba8, const u32 width,
                   const u32 height, const u32 row_pitch, u8 *out) {
  const u32 blocks_x = (width + 3u) / 4u;
  const u32 blocks_y = (height + 3u) / 4u;
  u8 *dp = out;
  for (u32 by = 0; by < blocks_y; ++by)
    for (u32 bx = 0; bx < blocks_x; ++bx) {
      u8 block[64];
      gather8(rgba8, row_pitch, width, height, bx, by, block);
      switch (fmt) {
      case CmpBc::BC1:
        CompressBlockBC1(block, 16, dp, nullptr);
        dp += 8;
        break;
      case CmpBc::BC3:
        CompressBlockBC3(block, 16, dp, nullptr);
        dp += 16;
        break;
      case CmpBc::BC4: {
        u8 r[16];
        for (u32 i = 0; i < 16; ++i)
          r[i] = block[i * 4];
        CompressBlockBC4(r, 4, dp, nullptr);
        dp += 8;
        break;
      }
      case CmpBc::BC5: {
        u8 r[16], g[16];
        for (u32 i = 0; i < 16; ++i) {
          r[i] = block[i * 4];
          g[i] = block[i * 4 + 1];
        }
        CompressBlockBC5(r, 4, g, 4, dp, nullptr);
        dp += 16;
        break;
      }
      case CmpBc::BC7:
        CompressBlockBC7(block, 16, dp, nullptr);
        dp += 16;
        break;
      }
    }
}

void cmp_encode_bc6h(const f32 *rgba32f, const u32 width, const u32 height,
                     const u32 row_pitch, u8 *out) {
  void *options = nullptr;
  CreateOptionsBC6(&options);
  SetSignedBC6(options, false); // BC6H_RGB_UFLOAT

  const u32 blocks_x = (width + 3u) / 4u;
  const u32 blocks_y = (height + 3u) / 4u;
  u8 *dp = out;
  for (u32 by = 0; by < blocks_y; ++by)
    for (u32 bx = 0; bx < blocks_x; ++bx) {
      u16 block[48]; // 16 texels x RGB half (BC6 source is 3-channel)
      for (u32 py = 0; py < 4; ++py) {
        const u32 sy = std::min(by * 4u + py, height - 1u);
        for (u32 sx0 = 0; sx0 < 4; ++sx0) {
          const u32 sx = std::min(bx * 4u + sx0, width - 1u);
          const f32 *t = reinterpret_cast<const f32 *>(
              reinterpret_cast<const u8 *>(rgba32f) + nx::cast<u64>(sy) * row_pitch +
              nx::cast<u64>(sx) * 16u);
          u16 *o = block + (py * 4u + sx0) * 3u;
          o[0] = to_half(t[0]);
          o[1] = to_half(t[1]);
          o[2] = to_half(t[2]);
        }
      }
      CompressBlockBC6(block, 12, dp, options); // stride: 4 texels x 3 channels
      dp += 16;
    }
  DestroyOptionsBC6(options);
}

} // namespace nxe::rhi
