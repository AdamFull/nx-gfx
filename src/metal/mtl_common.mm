
#include "core/rendering/rhi/metal/mtl_internal.h"

#include "core/foundation/core/foundation.h"

namespace nxe::rhi {

const nx::log::Category log_rhi = nx::log::category("rhi");

MTLPixelFormat to_mtl_format(const Format format) noexcept {
  switch (format) {
  case Format::R8_UNORM:
    return MTLPixelFormatR8Unorm;
  case Format::R8_SNORM:
    return MTLPixelFormatR8Snorm;
  case Format::R8_UINT:
    return MTLPixelFormatR8Uint;
  case Format::R8_SINT:
    return MTLPixelFormatR8Sint;
  case Format::RG8_UNORM:
    return MTLPixelFormatRG8Unorm;
  case Format::RG8_UINT:
    return MTLPixelFormatRG8Uint;
  case Format::RGBA8_UNORM:
    return MTLPixelFormatRGBA8Unorm;
  case Format::RGBA8_SRGB:
    return MTLPixelFormatRGBA8Unorm_sRGB;
  case Format::RGBA8_SNORM:
    return MTLPixelFormatRGBA8Snorm;
  case Format::RGBA8_UINT:
    return MTLPixelFormatRGBA8Uint;
  case Format::BGRA8_UNORM:
    return MTLPixelFormatBGRA8Unorm;
  case Format::BGRA8_SRGB:
    return MTLPixelFormatBGRA8Unorm_sRGB;

  case Format::R16_UNORM:
    return MTLPixelFormatR16Unorm;
  case Format::R16_FLOAT:
    return MTLPixelFormatR16Float;
  case Format::R16_UINT:
    return MTLPixelFormatR16Uint;
  case Format::R16_SINT:
    return MTLPixelFormatR16Sint;
  case Format::RG16_FLOAT:
    return MTLPixelFormatRG16Float;
  case Format::RG16_UNORM:
    return MTLPixelFormatRG16Unorm;
  case Format::RGBA16_FLOAT:
    return MTLPixelFormatRGBA16Float;
  case Format::RGBA16_UNORM:
    return MTLPixelFormatRGBA16Unorm;

  case Format::R32_FLOAT:
    return MTLPixelFormatR32Float;
  case Format::R32_UINT:
    return MTLPixelFormatR32Uint;
  case Format::R32_SINT:
    return MTLPixelFormatR32Sint;
  case Format::RG32_FLOAT:
    return MTLPixelFormatRG32Float;
  case Format::RG32_UINT:
    return MTLPixelFormatRG32Uint;
  case Format::RGBA32_FLOAT:
    return MTLPixelFormatRGBA32Float;
  case Format::RGBA32_UINT:
    return MTLPixelFormatRGBA32Uint;

  case Format::RGB10A2_UNORM:
    return MTLPixelFormatRGB10A2Unorm;
  case Format::RG11B10_FLOAT:
    return MTLPixelFormatRG11B10Float;

  case Format::D16_UNORM:
    return MTLPixelFormatDepth16Unorm;
  case Format::D32_FLOAT:
    return MTLPixelFormatDepth32Float;
  case Format::D24_UNORM_S8_UINT:
    // Mac-only, and there only on some GPUs; the enumerator does not even
    // exist on iOS. Invalid rather than a quiet 32-bit substitution - callers
    // wanting a guarantee ask for D32_FLOAT_S8_UINT.
#if TARGET_OS_OSX || TARGET_OS_MACCATALYST
    return MTLPixelFormatDepth24Unorm_Stencil8;
#else
    return MTLPixelFormatInvalid;
#endif
  case Format::D32_FLOAT_S8_UINT:
    return MTLPixelFormatDepth32Float_Stencil8;

#if TARGET_OS_OSX || TARGET_OS_MACCATALYST
  case Format::BC1_RGB_UNORM:
  case Format::BC1_RGBA_UNORM:
    return MTLPixelFormatBC1_RGBA;
  case Format::BC1_RGB_SRGB:
  case Format::BC1_RGBA_SRGB:
    return MTLPixelFormatBC1_RGBA_sRGB;
  case Format::BC2_RGBA_UNORM:
    return MTLPixelFormatBC2_RGBA;
  case Format::BC2_RGBA_SRGB:
    return MTLPixelFormatBC2_RGBA_sRGB;
  case Format::BC3_RGBA_UNORM:
    return MTLPixelFormatBC3_RGBA;
  case Format::BC3_RGBA_SRGB:
    return MTLPixelFormatBC3_RGBA_sRGB;
  case Format::BC4_R_UNORM:
    return MTLPixelFormatBC4_RUnorm;
  case Format::BC5_RG_UNORM:
    return MTLPixelFormatBC5_RGUnorm;
  case Format::BC6H_RGB_UFLOAT:
    return MTLPixelFormatBC6H_RGBUfloat;
  case Format::BC6H_RGB_SFLOAT:
    return MTLPixelFormatBC6H_RGBFloat;
  case Format::BC7_RGBA_UNORM:
    return MTLPixelFormatBC7_RGBAUnorm;
  case Format::BC7_RGBA_SRGB:
    return MTLPixelFormatBC7_RGBAUnorm_sRGB;
#endif

  case Format::ASTC_4x4_UNORM:
    return MTLPixelFormatASTC_4x4_LDR;
  case Format::ASTC_4x4_SRGB:
    return MTLPixelFormatASTC_4x4_sRGB;
  case Format::ASTC_6x6_UNORM:
    return MTLPixelFormatASTC_6x6_LDR;
  case Format::ASTC_6x6_SRGB:
    return MTLPixelFormatASTC_6x6_sRGB;
  case Format::ASTC_8x8_UNORM:
    return MTLPixelFormatASTC_8x8_LDR;
  case Format::ASTC_8x8_SRGB:
    return MTLPixelFormatASTC_8x8_sRGB;

  default:
    return MTLPixelFormatInvalid;
  }
}

Format from_mtl_format(const MTLPixelFormat format) noexcept {
  switch (format) {
  case MTLPixelFormatBGRA8Unorm:
    return Format::BGRA8_UNORM;
  case MTLPixelFormatBGRA8Unorm_sRGB:
    return Format::BGRA8_SRGB;
  case MTLPixelFormatRGBA8Unorm:
    return Format::RGBA8_UNORM;
  case MTLPixelFormatRGBA8Unorm_sRGB:
    return Format::RGBA8_SRGB;
  case MTLPixelFormatRGBA16Float:
    return Format::RGBA16_FLOAT;
  case MTLPixelFormatRGB10A2Unorm:
    return Format::RGB10A2_UNORM;
  default:
    return Format::Unknown;
  }
}

MTLTextureType to_mtl_texture_type(const TextureDimension dimension,
                                   const u32 layers,
                                   const u32 sample_count) noexcept {
  const bool multisampled = sample_count > 1;
  switch (dimension) {
  case TextureDimension::Tex2D:
    return multisampled ? MTLTextureType2DMultisample : MTLTextureType2D;
  case TextureDimension::Tex2DArray:
    return multisampled ? MTLTextureType2DMultisampleArray
                        : MTLTextureType2DArray;
  case TextureDimension::TexCube:
    return layers > 6 ? MTLTextureTypeCubeArray : MTLTextureTypeCube;
  case TextureDimension::Tex3D:
    return MTLTextureType3D;
  default:
    return MTLTextureType2D;
  }
}

MTLTextureUsage to_mtl_texture_usage(const TextureUsageFlags usage) noexcept {
  MTLTextureUsage out = MTLTextureUsageUnknown;
  if (usage.has(TextureUsage::Sampled))
    out |= MTLTextureUsageShaderRead;
  if (usage.has(TextureUsage::Storage))
    out |= MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  if (usage.has(TextureUsage::RenderTarget) ||
      usage.has(TextureUsage::DepthStencil))
    out |= MTLTextureUsageRenderTarget;
  // CopySrc / CopyDst need no usage bit: blit encoders can touch any texture.
  return out;
}

MTLStorageMode to_mtl_storage_mode(const MemoryUsage memory,
                                   const bool transient) noexcept {
  if (transient) {
#if TARGET_OS_IPHONE
    return MTLStorageModeMemoryless;
#else
    return MTLStorageModePrivate;
#endif
  }

  switch (memory) {
  case MemoryUsage::Upload:
  case MemoryUsage::Readback:
    return MTLStorageModeShared;
  case MemoryUsage::DeviceLocal:
  default:
    return MTLStorageModePrivate;
  }
}

MTLSamplerMinMagFilter to_mtl_filter(const FilterMode filter) noexcept {
  return filter == FilterMode::Nearest ? MTLSamplerMinMagFilterNearest
                                       : MTLSamplerMinMagFilterLinear;
}

MTLSamplerMipFilter to_mtl_mip_filter(const FilterMode filter) noexcept {
  return filter == FilterMode::Nearest ? MTLSamplerMipFilterNearest
                                       : MTLSamplerMipFilterLinear;
}

MTLSamplerAddressMode to_mtl_address(const AddressMode mode) noexcept {
  switch (mode) {
  case AddressMode::Repeat:
    return MTLSamplerAddressModeRepeat;
  case AddressMode::MirroredRepeat:
    return MTLSamplerAddressModeMirrorRepeat;
  case AddressMode::ClampToBorder:
    return MTLSamplerAddressModeClampToBorderColor;
  case AddressMode::ClampToEdge:
  default:
    return MTLSamplerAddressModeClampToEdge;
  }
}

MTLCompareFunction to_mtl_compare(const CompareOp op) noexcept {
  switch (op) {
  case CompareOp::Never:
    return MTLCompareFunctionNever;
  case CompareOp::Less:
    return MTLCompareFunctionLess;
  case CompareOp::Equal:
    return MTLCompareFunctionEqual;
  case CompareOp::LessOrEqual:
    return MTLCompareFunctionLessEqual;
  case CompareOp::Greater:
    return MTLCompareFunctionGreater;
  case CompareOp::NotEqual:
    return MTLCompareFunctionNotEqual;
  case CompareOp::GreaterOrEqual:
    return MTLCompareFunctionGreaterEqual;
  case CompareOp::Always:
  default:
    return MTLCompareFunctionAlways;
  }
}

MTLLoadAction to_mtl_load_action(const LoadOp op) noexcept {
  switch (op) {
  case LoadOp::Load:
    return MTLLoadActionLoad;
  case LoadOp::Clear:
    return MTLLoadActionClear;
  case LoadOp::DontCare:
  default:
    return MTLLoadActionDontCare;
  }
}

MTLStoreAction to_mtl_store_action(const StoreOp op) noexcept {
  return op == StoreOp::Store ? MTLStoreActionStore : MTLStoreActionDontCare;
}

NSString *to_ns_string(const nx::string_view text) noexcept {
  if (text.empty())
    return @"";
  return [[NSString alloc] initWithBytes:text.data()
                                  length:nx::cast<NSUInteger>(text.size())
                                encoding:NSUTF8StringEncoding];
}

}
