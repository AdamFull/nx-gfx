
#include "rendering/rhi/metal/mtl_internal.h"

#include "core/foundation/core/foundation.h"

namespace nx::gfx::detail {

const nx::log::Category log_rhi = nx::log::category("rhi");

MTLPixelFormat to_mtl_format(const gfx::format format) noexcept {
  switch (format) {
  case gfx::format::R8_UNORM:
    return MTLPixelFormatR8Unorm;
  case gfx::format::R8_SNORM:
    return MTLPixelFormatR8Snorm;
  case gfx::format::R8_UINT:
    return MTLPixelFormatR8Uint;
  case gfx::format::R8_SINT:
    return MTLPixelFormatR8Sint;
  case gfx::format::RG8_UNORM:
    return MTLPixelFormatRG8Unorm;
  case gfx::format::RG8_UINT:
    return MTLPixelFormatRG8Uint;
  case gfx::format::RGBA8_UNORM:
    return MTLPixelFormatRGBA8Unorm;
  case gfx::format::RGBA8_SRGB:
    return MTLPixelFormatRGBA8Unorm_sRGB;
  case gfx::format::RGBA8_SNORM:
    return MTLPixelFormatRGBA8Snorm;
  case gfx::format::RGBA8_UINT:
    return MTLPixelFormatRGBA8Uint;
  case gfx::format::BGRA8_UNORM:
    return MTLPixelFormatBGRA8Unorm;
  case gfx::format::BGRA8_SRGB:
    return MTLPixelFormatBGRA8Unorm_sRGB;

  case gfx::format::R16_UNORM:
    return MTLPixelFormatR16Unorm;
  case gfx::format::R16_FLOAT:
    return MTLPixelFormatR16Float;
  case gfx::format::R16_UINT:
    return MTLPixelFormatR16Uint;
  case gfx::format::R16_SINT:
    return MTLPixelFormatR16Sint;
  case gfx::format::RG16_FLOAT:
    return MTLPixelFormatRG16Float;
  case gfx::format::RG16_UNORM:
    return MTLPixelFormatRG16Unorm;
  case gfx::format::RGBA16_FLOAT:
    return MTLPixelFormatRGBA16Float;
  case gfx::format::RGBA16_UNORM:
    return MTLPixelFormatRGBA16Unorm;

  case gfx::format::R32_FLOAT:
    return MTLPixelFormatR32Float;
  case gfx::format::R32_UINT:
    return MTLPixelFormatR32Uint;
  case gfx::format::R32_SINT:
    return MTLPixelFormatR32Sint;
  case gfx::format::RG32_FLOAT:
    return MTLPixelFormatRG32Float;
  case gfx::format::RG32_UINT:
    return MTLPixelFormatRG32Uint;
  case gfx::format::RGBA32_FLOAT:
    return MTLPixelFormatRGBA32Float;
  case gfx::format::RGBA32_UINT:
    return MTLPixelFormatRGBA32Uint;

  case gfx::format::RGB10A2_UNORM:
    return MTLPixelFormatRGB10A2Unorm;
  case gfx::format::RG11B10_FLOAT:
    return MTLPixelFormatRG11B10Float;

  case gfx::format::D16_UNORM:
    return MTLPixelFormatDepth16Unorm;
  case gfx::format::D32_FLOAT:
    return MTLPixelFormatDepth32Float;
  case gfx::format::D24_UNORM_S8_UINT:
    // Mac-only, and there only on some GPUs; the enumerator does not even
    // exist on iOS. Invalid rather than a quiet 32-bit substitution - callers
    // wanting a guarantee ask for D32_FLOAT_S8_UINT.
#if TARGET_OS_OSX || TARGET_OS_MACCATALYST
    return MTLPixelFormatDepth24Unorm_Stencil8;
#else
    return MTLPixelFormatInvalid;
#endif
  case gfx::format::D32_FLOAT_S8_UINT:
    return MTLPixelFormatDepth32Float_Stencil8;

#if TARGET_OS_OSX || TARGET_OS_MACCATALYST
  case gfx::format::BC1_RGB_UNORM:
  case gfx::format::BC1_RGBA_UNORM:
    return MTLPixelFormatBC1_RGBA;
  case gfx::format::BC1_RGB_SRGB:
  case gfx::format::BC1_RGBA_SRGB:
    return MTLPixelFormatBC1_RGBA_sRGB;
  case gfx::format::BC2_RGBA_UNORM:
    return MTLPixelFormatBC2_RGBA;
  case gfx::format::BC2_RGBA_SRGB:
    return MTLPixelFormatBC2_RGBA_sRGB;
  case gfx::format::BC3_RGBA_UNORM:
    return MTLPixelFormatBC3_RGBA;
  case gfx::format::BC3_RGBA_SRGB:
    return MTLPixelFormatBC3_RGBA_sRGB;
  case gfx::format::BC4_R_UNORM:
    return MTLPixelFormatBC4_RUnorm;
  case gfx::format::BC5_RG_UNORM:
    return MTLPixelFormatBC5_RGUnorm;
  case gfx::format::BC6H_RGB_UFLOAT:
    return MTLPixelFormatBC6H_RGBUfloat;
  case gfx::format::BC6H_RGB_SFLOAT:
    return MTLPixelFormatBC6H_RGBFloat;
  case gfx::format::BC7_RGBA_UNORM:
    return MTLPixelFormatBC7_RGBAUnorm;
  case gfx::format::BC7_RGBA_SRGB:
    return MTLPixelFormatBC7_RGBAUnorm_sRGB;
#endif

  case gfx::format::ASTC_4x4_UNORM:
    return MTLPixelFormatASTC_4x4_LDR;
  case gfx::format::ASTC_4x4_SRGB:
    return MTLPixelFormatASTC_4x4_sRGB;
  case gfx::format::ASTC_6x6_UNORM:
    return MTLPixelFormatASTC_6x6_LDR;
  case gfx::format::ASTC_6x6_SRGB:
    return MTLPixelFormatASTC_6x6_sRGB;
  case gfx::format::ASTC_8x8_UNORM:
    return MTLPixelFormatASTC_8x8_LDR;
  case gfx::format::ASTC_8x8_SRGB:
    return MTLPixelFormatASTC_8x8_sRGB;

  default:
    return MTLPixelFormatInvalid;
  }
}

gfx::format from_mtl_format(const MTLPixelFormat format) noexcept {
  switch (format) {
  case MTLPixelFormatBGRA8Unorm:
    return gfx::format::BGRA8_UNORM;
  case MTLPixelFormatBGRA8Unorm_sRGB:
    return gfx::format::BGRA8_SRGB;
  case MTLPixelFormatRGBA8Unorm:
    return gfx::format::RGBA8_UNORM;
  case MTLPixelFormatRGBA8Unorm_sRGB:
    return gfx::format::RGBA8_SRGB;
  case MTLPixelFormatRGBA16Float:
    return gfx::format::RGBA16_FLOAT;
  case MTLPixelFormatRGB10A2Unorm:
    return gfx::format::RGB10A2_UNORM;
  default:
    return gfx::format::Unknown;
  }
}

MTLTextureType to_mtl_texture_type(const gfx::texture_dimension dimension,
                                   const u32 layers,
                                   const u32 sample_count) noexcept {
  const bool multisampled = sample_count > 1;
  switch (dimension) {
  case gfx::texture_dimension::Tex2D:
    return multisampled ? MTLTextureType2DMultisample : MTLTextureType2D;
  case gfx::texture_dimension::Tex2DArray:
    return multisampled ? MTLTextureType2DMultisampleArray
                        : MTLTextureType2DArray;
  case gfx::texture_dimension::TexCube:
    return layers > 6 ? MTLTextureTypeCubeArray : MTLTextureTypeCube;
  case gfx::texture_dimension::Tex3D:
    return MTLTextureType3D;
  default:
    return MTLTextureType2D;
  }
}

MTLTextureUsage to_mtl_texture_usage(const gfx::texture_usage_flags usage) noexcept {
  MTLTextureUsage out = MTLTextureUsageUnknown;
  if (usage.has(gfx::texture_usage::Sampled))
    out |= MTLTextureUsageShaderRead;
  if (usage.has(gfx::texture_usage::Storage))
    out |= MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  if (usage.has(gfx::texture_usage::RenderTarget) ||
      usage.has(gfx::texture_usage::DepthStencil))
    out |= MTLTextureUsageRenderTarget;
  // CopySrc / CopyDst need no usage bit: blit encoders can touch any texture.
  return out;
}

MTLStorageMode to_mtl_storage_mode(const gfx::memory_usage memory,
                                   const bool transient) noexcept {
  if (transient) {
#if TARGET_OS_IPHONE
    return MTLStorageModeMemoryless;
#else
    return MTLStorageModePrivate;
#endif
  }

  switch (memory) {
  case gfx::memory_usage::Upload:
  case gfx::memory_usage::Readback:
    return MTLStorageModeShared;
  case gfx::memory_usage::DeviceLocal:
  default:
    return MTLStorageModePrivate;
  }
}

MTLSamplerMinMagFilter to_mtl_filter(const gfx::filter_mode filter) noexcept {
  return filter == gfx::filter_mode::Nearest ? MTLSamplerMinMagFilterNearest
                                       : MTLSamplerMinMagFilterLinear;
}

MTLSamplerMipFilter to_mtl_mip_filter(const gfx::filter_mode filter) noexcept {
  return filter == gfx::filter_mode::Nearest ? MTLSamplerMipFilterNearest
                                       : MTLSamplerMipFilterLinear;
}

MTLSamplerAddressMode to_mtl_address(const gfx::address_mode mode) noexcept {
  switch (mode) {
  case gfx::address_mode::Repeat:
    return MTLSamplerAddressModeRepeat;
  case gfx::address_mode::MirroredRepeat:
    return MTLSamplerAddressModeMirrorRepeat;
  case gfx::address_mode::ClampToBorder:
    return MTLSamplerAddressModeClampToBorderColor;
  case gfx::address_mode::ClampToEdge:
  default:
    return MTLSamplerAddressModeClampToEdge;
  }
}

MTLCompareFunction to_mtl_compare(const gfx::compare_op op) noexcept {
  switch (op) {
  case gfx::compare_op::Never:
    return MTLCompareFunctionNever;
  case gfx::compare_op::Less:
    return MTLCompareFunctionLess;
  case gfx::compare_op::Equal:
    return MTLCompareFunctionEqual;
  case gfx::compare_op::LessOrEqual:
    return MTLCompareFunctionLessEqual;
  case gfx::compare_op::Greater:
    return MTLCompareFunctionGreater;
  case gfx::compare_op::NotEqual:
    return MTLCompareFunctionNotEqual;
  case gfx::compare_op::GreaterOrEqual:
    return MTLCompareFunctionGreaterEqual;
  case gfx::compare_op::Always:
  default:
    return MTLCompareFunctionAlways;
  }
}

MTLLoadAction to_mtl_load_action(const gfx::load_op op) noexcept {
  switch (op) {
  case gfx::load_op::Load:
    return MTLLoadActionLoad;
  case gfx::load_op::Clear:
    return MTLLoadActionClear;
  case gfx::load_op::DontCare:
  default:
    return MTLLoadActionDontCare;
  }
}

MTLStoreAction to_mtl_store_action(const gfx::store_op op) noexcept {
  return op == gfx::store_op::Store ? MTLStoreActionStore : MTLStoreActionDontCare;
}

NSString *to_ns_string(const nx::string_view text) noexcept {
  if (text.empty())
    return @"";
  return [[NSString alloc] initWithBytes:text.data()
                                  length:nx::cast<NSUInteger>(text.size())
                                encoding:NSUTF8StringEncoding];
}

}
