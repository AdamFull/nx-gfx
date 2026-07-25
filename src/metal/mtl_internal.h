#pragma once

/**
 * @file mtl_internal.h
 * @brief The backend's Objective-C state, shared between the Metal .mm files.
 *
 * Objective-C++ only: never include this from a .cpp. Everything the public
 * mtl_device.h hides behind an opaque MetalDeviceState pointer is spelled out
 * here.
 *
 * ARC owns every object in these structs. That is the reason the state is
 * heap-allocated with new/delete rather than being a member of MetalDevice:
 * MetalDevice lives in a plain C++ header, where a strong `id` member could not
 * be declared, let alone destroyed correctly.
 */

#if !defined(__OBJC__)
#error "mtl_internal.h is Objective-C++ only"
#endif

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include "foundation/containers/slot_map.h"
#include "foundation/diagnostics/log.h"
#include "rendering/rhi/metal/mtl_device.h"

namespace nxe::rhi {

extern const nx::log::Category log_rhi;

/// Format translation. Format::Unknown maps to MTLPixelFormatInvalid, as does
/// any format Metal has no equivalent for (the BC set on Apple silicon, ASTC on
/// Intel); callers must check before using the result.
[[nodiscard]] MTLPixelFormat to_mtl_format(Format format) noexcept;
[[nodiscard]] Format from_mtl_format(MTLPixelFormat format) noexcept;

[[nodiscard]] MTLTextureType to_mtl_texture_type(TextureDimension dimension,
                                                 u32 layers,
                                                 u32 sample_count) noexcept;
[[nodiscard]] MTLTextureUsage to_mtl_texture_usage(TextureUsageFlags usage) noexcept;
[[nodiscard]] MTLStorageMode to_mtl_storage_mode(MemoryUsage memory,
                                                 bool transient) noexcept;
[[nodiscard]] MTLSamplerMinMagFilter to_mtl_filter(FilterMode filter) noexcept;
[[nodiscard]] MTLSamplerMipFilter to_mtl_mip_filter(FilterMode filter) noexcept;
[[nodiscard]] MTLSamplerAddressMode to_mtl_address(AddressMode mode) noexcept;
[[nodiscard]] MTLCompareFunction to_mtl_compare(CompareOp op) noexcept;
[[nodiscard]] MTLLoadAction to_mtl_load_action(LoadOp op) noexcept;
[[nodiscard]] MTLStoreAction to_mtl_store_action(StoreOp op) noexcept;

/// nx::string_view is not NUL-terminated; Metal's labels want an NSString.
[[nodiscard]] NSString *to_ns_string(nx::string_view text) noexcept;

struct MetalTexture {
  id<MTLTexture> texture = nil;
  Format format = Format::Unknown;
  TextureDimension dimension = TextureDimension::Tex2D;
  u32 width = 0;
  u32 height = 0;
  u32 mip_levels = 1;
  u32 layers = 1;
  u32 sample_count = 1;
  TextureUsageFlags usage = TextureUsage::None;
  ResourceState state = ResourceState::Undefined;
  /// False for the backbuffer, whose MTLTexture belongs to the drawable and is
  /// swapped out from under the handle every frame.
  bool owns_texture = true;
};

struct MetalBuffer {
  id<MTLBuffer> buffer = nil;
  u64 size = 0;
  BufferUsageFlags usage = BufferUsage::None;
  void *mapped = nullptr;
};

struct MetalSampler {
  id<MTLSamplerState> sampler = nil;
};

struct MetalDeviceState {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;

  // Swapchain. The layer is owned by the window's view once attached; we keep a
  // strong reference so a resize can still reach it.
  CAMetalLayer *layer = nil;
  /// The NSView the layer was attached to, so destroy_swapchain can detach it
  /// again. Typed as `id` rather than NSView* to keep AppKit out of this
  /// header; nil on iOS, where the layer is handed to us already attached.
  id layer_host = nil;
  id<CAMetalDrawable> drawable = nil;
  Format swapchain_format = Format::Unknown;
  Extent2D extent = {};
  PresentMode present_mode = PresentMode::Fifo;
  /// One handle for the backbuffer, re-pointed at each frame's drawable
  /// texture. Metal hands out a different MTLTexture every frame, so there is
  /// no fixed set of backbuffer images to mirror the Vulkan backend with.
  TextureHandle backbuffer;

  // Frame pacing. Metal has no timeline semaphore; a counting semaphore
  // released from the command buffer's completion handler is the standard way
  // to cap how far the CPU may run ahead.
  dispatch_semaphore_t frame_semaphore = nil;
  u32 frames_in_flight = 2;
  u64 frames_submitted = 0;

  // Metal vends command buffers, drawables and encoders autoreleased, and this
  // process has no run loop to drain the thread's outermost pool - the frame
  // loop is our own while(). Every .mm entry point therefore wraps its body in
  // @autoreleasepool; assigning to these strong members is what keeps the
  // objects alive past the pool, across the begin_frame / end_frame boundary.
  id<MTLCommandBuffer> cmd = nil;
  id<MTLRenderCommandEncoder> encoder = nil;
  bool frame_active = false;

  nx::slot_map<MetalTexture, TextureHandle> textures;
  nx::slot_map<MetalBuffer, BufferHandle> buffers;
  nx::slot_map<MetalSampler, SamplerHandle> samplers;
};

} // namespace nxe::rhi
