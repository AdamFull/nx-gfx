/**
 * @file texture_compress.cpp
 * @brief Cook-time compression of a decoded image to a GPU block format.
 *
 * BC1-BC7 (incl. BC6H) go through Compressonator's CMP_Core; ASTC through ARM's
 * astc-encoder. All of it is cook-only - the runtime uploads the cooked blocks
 * and never compresses - so on a build without the encoders (Android) every
 * target is refused.
 */

#include "core/rendering/rhi/image/image.h"

#include "core/rendering/rhi/types.h"

#if NX_TEXTURE_ENCODERS
#include "core/foundation/threading/sync.h"
#include "core/foundation/threading/thread.h"
#include "core/rendering/rhi/image/cmp_encode.h"

#include <astcenc.h>

#include <atomic>
#endif

namespace nxe::rhi {
#if NX_TEXTURE_ENCODERS
namespace {

/// Tightly-packed byte size of a full mip chain in @p target.
[[nodiscard]] u64 compressed_size(const Format target, const u32 w, const u32 h,
                                  const u32 mips) noexcept {
  u64 total = 0;
  for (u32 m = 0; m < mips; ++m)
    total += format_subresource_size(target, nx::max(1u, w >> m),
                                     nx::max(1u, h >> m), 1);
  return total;
}

/// Encodes each mip through @p encode(src, w, h, row_pitch, dst) -> ok, sizing
/// the output for @p target and rebuilding its subresource table. Commits into
/// @p image only if every mip succeeds.
template <class Encode>
[[nodiscard]] bool compress_mips(ImageData &image, const Format target,
                                 Encode &&encode) {
  const u32 mips = image.mip_levels;
  nx::blob<u8> dst(compressed_size(target, image.width, image.height, mips));
  const nx::vector<ImageSubresource> src = image.subresources;
  const u8 *const pixels = image.pixels.data();
  nx::vector<ImageSubresource> subs;
  subs.reserve(mips);

  u64 off = 0;
  for (u32 mip = 0; mip < mips; ++mip) {
    const ImageSubresource &s = src[mip];
    if (!encode(pixels + s.offset, s.width, s.height, s.row_pitch,
                dst.data() + off))
      return false;
    const u64 size = format_subresource_size(target, s.width, s.height, 1);
    subs.push_back(ImageSubresource{
        off, size, mip, 0, s.width, s.height, 1,
        nx::cast<u32>(format_row_pitch(target, s.width)), 0, 0, 0});
    off += size;
  }
  image.format = target;
  image.pixels = std::move(dst);
  image.subresources = std::move(subs);
  return true;
}

// --- BC (Compressonator CMP_Core) ------------------------------------------

[[nodiscard]] bool bc_target(const Format f, CmpBc &fmt) noexcept {
  switch (f) {
  case Format::BC1_RGB_UNORM:
  case Format::BC1_RGBA_UNORM:
    fmt = CmpBc::BC1;
    return true;
  case Format::BC3_RGBA_UNORM:
    fmt = CmpBc::BC3;
    return true;
  case Format::BC4_R_UNORM:
    fmt = CmpBc::BC4;
    return true;
  case Format::BC5_RG_UNORM:
    fmt = CmpBc::BC5;
    return true;
  case Format::BC7_RGBA_UNORM:
    fmt = CmpBc::BC7;
    return true;
  default:
    return false;
  }
}

// --- ASTC (astc-encoder) ---------------------------------------------------

[[nodiscard]] bool is_astc(const Format f) noexcept {
  return f == Format::ASTC_4x4_UNORM || f == Format::ASTC_6x6_UNORM ||
         f == Format::ASTC_8x8_UNORM;
}

[[nodiscard]] u32 astc_threads() noexcept {
  return nx::min(nx::max(nx::cpu::logical_core_count(), 1u), 16u);
}

/// One astcenc context per (block size, thread count), reused across textures:
/// allocating one builds the block tables and takes seconds, so paying it per
/// texture made a cook of a few hundred textures take minutes. The static frees
/// it at exit rather than leaking. Caller holds the lock.
[[nodiscard]] astcenc_context *shared_astc_context(const u32 bx, const u32 by,
                                                   const u32 threads) {
  struct Cache {
    astcenc_context *ctx = nullptr;
    u32 bx = 0, by = 0, threads = 0;
    ~Cache() {
      if (ctx)
        astcenc_context_free(ctx);
    }
  };
  static Cache cache;
  if (cache.ctx != nullptr && cache.bx == bx && cache.by == by &&
      cache.threads == threads)
    return cache.ctx;
  if (cache.ctx != nullptr) {
    astcenc_context_free(cache.ctx);
    cache.ctx = nullptr;
  }
  // FAST, not MEDIUM: MEDIUM is far slower and the cook has to be usable.
  // Quality can become a flag if a project needs it.
  astcenc_config config{};
  if (astcenc_config_init(ASTCENC_PRF_LDR, bx, by, 1, ASTCENC_PRE_FAST, 0,
                          &config) != ASTCENC_SUCCESS)
    return nullptr;
  if (astcenc_context_alloc(&config, threads, &cache.ctx, nullptr) !=
      ASTCENC_SUCCESS) {
    cache.ctx = nullptr;
    return nullptr;
  }
  cache.bx = bx;
  cache.by = by;
  cache.threads = threads;
  return cache.ctx;
}

/// Runs one image through astcenc across @p threads workers. astcenc divides the
/// blocks among the threads that call it, so all @p threads must enter - this
/// thread takes index 0 and spawns the rest.
[[nodiscard]] bool astc_encode_image(astcenc_context *ctx, astcenc_image *img,
                                     const astcenc_swizzle *swizzle, u8 *out,
                                     const u64 size, const u32 threads) {
  std::atomic<bool> failed{false};
  const auto run = [&](const u32 index) {
    if (astcenc_compress_image(ctx, img, swizzle, out, size, index) !=
        ASTCENC_SUCCESS)
      failed.store(true, std::memory_order_relaxed);
  };
  nx::vector<nx::thread> workers;
  workers.reserve(threads > 0 ? threads - 1 : 0);
  for (u32 t = 1; t < threads; ++t)
    workers.emplace_back([&run, t] { run(t); });
  run(0);
  for (nx::thread &w : workers)
    w.join();
  return !failed.load(std::memory_order_relaxed);
}

[[nodiscard]] bool compress_astc(ImageData &image, const Format target) {
  u32 bx = 4, by = 4;
  format_block_extent(target, bx, by);

  static nx::mutex astc_lock;
  const nx::scoped_lock<nx::mutex> guard(astc_lock);
  const u32 threads = astc_threads();
  astcenc_context *const ctx = shared_astc_context(bx, by, threads);
  if (ctx == nullptr)
    return false;
  astcenc_compress_reset(ctx);

  const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B,
                                ASTCENC_SWZ_A};
  return compress_mips(
      image, target,
      [&](const u8 *px, const u32 w, const u32 h, u32, u8 *out) {
        void *slice = const_cast<u8 *>(px);
        astcenc_image img{};
        img.dim_x = w;
        img.dim_y = h;
        img.dim_z = 1;
        img.data_type = ASTCENC_TYPE_U8;
        img.data = &slice;
        const bool ok = astc_encode_image(
            ctx, &img, &swizzle, out, format_subresource_size(target, w, h, 1),
            threads);
        astcenc_compress_reset(ctx);
        return ok;
      });
}

} // namespace
#endif

bool compress_texture(ImageData &image, const Format target) {
#if NX_TEXTURE_ENCODERS
  if (image.format == target)
    return true;
  if (image.dimension != TextureDimension::Tex2D || image.array_layers != 1 ||
      image.subresources.empty())
    return false;

  // BC6H is the HDR target: a float source, encoded from FP16.
  if (target == Format::BC6H_RGB_UFLOAT) {
    if (image.format != Format::RGBA32_FLOAT)
      return false;
    return compress_mips(image, target,
                         [](const u8 *px, u32 w, u32 h, u32 rp, u8 *out) {
                           cmp_encode_bc6h(reinterpret_cast<const f32 *>(px), w,
                                           h, rp, out);
                           return true;
                         });
  }

  // Every other target is LDR, from an RGBA8 decode.
  if (image.format != Format::RGBA8_UNORM)
    return false;
  if (is_astc(target))
    return compress_astc(image, target);
  CmpBc fmt;
  if (!bc_target(target, fmt))
    return false;
  return compress_mips(image, target,
                       [fmt](const u8 *px, u32 w, u32 h, u32 rp, u8 *out) {
                         cmp_encode_bc(fmt, px, w, h, rp, out);
                         return true;
                       });
#else
  (void)image;
  (void)target;
  return false;
#endif
}

} // namespace nxe::rhi
