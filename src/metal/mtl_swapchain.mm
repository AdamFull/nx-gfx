
#include "rendering/rhi/metal/mtl_internal.h"

#if TARGET_OS_OSX
#import <AppKit/AppKit.h>
#else
#import <UIKit/UIKit.h>
#endif

namespace nxe::rhi {
namespace {

/// AppKit is main-thread-only. The desktop frame loop already runs there
/// (the desktop platform requires it), so this is normally a straight call.
void on_main_thread(void (^block)(void)) {
  if ([NSThread isMainThread]) {
    block();
    return;
  }
  dispatch_sync(dispatch_get_main_queue(), block);
}

/// Metal only ever presents 8-bit BGRA or a wide-gamut float format, so a
/// preferred format it cannot show falls back rather than failing.
MTLPixelFormat pick_layer_format(const Format preferred) noexcept {
  switch (preferred) {
  case Format::BGRA8_UNORM:
  case Format::BGRA8_SRGB:
  case Format::RGBA16_FLOAT:
  case Format::RGB10A2_UNORM:
    return to_mtl_format(preferred);
  case Format::RGBA8_UNORM:
    return MTLPixelFormatBGRA8Unorm;
  case Format::RGBA8_SRGB:
    return MTLPixelFormatBGRA8Unorm_sRGB;
  default:
    return MTLPixelFormatBGRA8Unorm;
  }
}

}

SwapchainHandle MetalDevice::create_swapchain(const SwapchainDesc &desc) {
  if (m_state == nullptr) {
    nx::loge(log_rhi, "create_swapchain before init");
    return {};
  }
  if (desc.width == 0 || desc.height == 0) {
    nx::loge(log_rhi, "create_swapchain: zero extent");
    return {};
  }

  MetalDeviceState *s = m_state;
  // Pipelines are built for one format, so the swapchains after the first
  // take its format.
  const bool fixed = !s->swapchains.empty();
  const MTLPixelFormat pixel_format =
      pick_layer_format(fixed ? s->swapchain_format : desc.preferred_format);
  if (fixed && from_mtl_format(pixel_format) != s->swapchain_format) {
    nx::loge(log_rhi,
             "create_swapchain: a layer cannot present {}, the format the "
             "other swapchains have",
             to_string(s->swapchain_format));
    return {};
  }
  // Core Animation composites a non-opaque layer as premultiplied.
  const bool opaque = desc.alpha == SurfaceAlpha::Opaque;

  __block CAMetalLayer *layer = nil;
  __block id layer_host = nil;
  __block bool ok = false;

  @autoreleasepool {
    switch (desc.window.kind) {
#if TARGET_OS_OSX
    case nx::NativeWindowKind::Cocoa: {
      NSWindow *window = (__bridge NSWindow *)desc.window.window;
      if (window == nil) {
        nx::loge(log_rhi, "create_swapchain: null NSWindow");
        return {};
      }

      id<MTLDevice> device = s->device;
      on_main_thread(^{
        NSView *view = window.contentView;
        if (view == nil)
          return;

        layer = [CAMetalLayer layer];
        layer.device = device;
        layer.pixelFormat = pixel_format;
        // Final-frame capture uses a blit from the drawable before it is
        // presented. framebufferOnly drawables cannot participate in that
        // copy on Metal.
        layer.framebufferOnly = NO;
        layer.opaque = opaque;
        layer.contentsScale = window.backingScaleFactor;

        view.wantsLayer = YES;
        view.layer = layer;
        layer_host = view;
        ok = true;
      });
      break;
    }
#else
    case nx::NativeWindowKind::UIKit: {
      layer = (__bridge CAMetalLayer *)desc.window.window;
      if (layer == nil) {
        nx::loge(log_rhi, "create_swapchain: null CAMetalLayer");
        return {};
      }
      layer.device = s->device;
      layer.pixelFormat = pixel_format;
      layer.framebufferOnly = NO;
      layer.opaque = opaque;
      ok = true;
      break;
    }
#endif
    default:
      nx::loge(log_rhi, "create_swapchain: native window kind {} is not a "
                        "Metal surface",
               nx::cast<u32>(desc.window.kind));
      return {};
    }
  }

  if (!ok || layer == nil) {
    nx::loge(log_rhi, "create_swapchain: could not attach a CAMetalLayer");
    return {};
  }

  const u32 frames_in_flight = s->frames_in_flight;

  @autoreleasepool {
    layer.drawableSize = CGSizeMake(desc.width, desc.height);
    layer.maximumDrawableCount = nx::clamp<u32>(frames_in_flight + 1, 2, 3);
#if TARGET_OS_OSX
    layer.displaySyncEnabled = desc.present_mode != PresentMode::Immediate;
#endif
  }

  s->swapchain_format = from_mtl_format(pixel_format);
  MetalSwapchain sc;
  sc.layer = layer;
  sc.layer_host = layer_host;
  sc.extent = {desc.width, desc.height};
  sc.present_mode = desc.present_mode;
  sc.alpha = opaque ? SurfaceAlpha::Opaque : SurfaceAlpha::Premultiplied;

  // One handle for the backbuffer, valid for the swapchain's lifetime. Its
  // MTLTexture is filled in per frame from the drawable.
  MetalTexture backbuffer;
  backbuffer.format = s->swapchain_format;
  backbuffer.dimension = TextureDimension::Tex2D;
  backbuffer.width = desc.width;
  backbuffer.height = desc.height;
  backbuffer.usage = TextureUsage::RenderTarget | TextureUsage::CopySrc;
  backbuffer.state = ResourceState::Undefined;
  backbuffer.owns_texture = false;
  sc.backbuffer = s->textures.insert(std::move(backbuffer));

  nx::logd(log_rhi, "swapchain: {}x{} {} ({} frames in flight, {})", desc.width,
           desc.height, to_string(s->swapchain_format), frames_in_flight,
           desc.present_mode == PresentMode::Immediate ? "no vsync" : "vsync");
  return s->swapchains.insert(std::move(sc));
}

SwapchainHandle MetalDevice::create_offscreen_swapchain(const u32 width,
                                                       const u32 height,
                                                       const Format format) {
  if (m_state == nullptr || device_status() == GpuDeviceStatus::Lost)
    return {};
  if (width == 0 || height == 0 || format == Format::Unknown)
    return {};
  MetalDeviceState *s = m_state;
  if (!s->swapchains.empty() && format != s->swapchain_format) {
    nx::loge(log_rhi, "offscreen swapchain in {} while the other swapchains "
                      "are {}",
             to_string(format), to_string(s->swapchain_format));
    return {};
  }

  // CopySrc because a frame capture reads the backbuffer back, and Sampled
  // because a pass may sample what an earlier one wrote to it.
  const TextureHandle image = create_texture({
      .name = "offscreen backbuffer",
      .format = format,
      .width = width,
      .height = height,
      .usage = TextureUsage::RenderTarget | TextureUsage::Sampled |
               TextureUsage::CopySrc,
  });
  if (!image.valid())
    return {};

  s->swapchain_format = format;
  MetalSwapchain sc;
  sc.backbuffer = image;
  sc.extent = {width, height};
  sc.offscreen = true;
  nx::logd(log_rhi, "offscreen swapchain: {}x{} {}", width, height,
           to_string(format));
  return s->swapchains.insert(std::move(sc));
}

bool MetalDevice::offscreen_swapchain(const SwapchainHandle handle) const {
  const MetalSwapchain *const sc =
      m_state != nullptr ? m_state->swapchains.try_get(handle) : nullptr;
  return sc != nullptr && sc->offscreen;
}

bool MetalDevice::resize_swapchain(const SwapchainHandle handle,
                                   const u32 width, const u32 height) {
  if (m_state == nullptr || width == 0 || height == 0)
    return false;
  MetalDeviceState *s = m_state;
  MetalSwapchain *sc = s->swapchains.try_get(handle);
  if (sc == nullptr)
    return false;
  if (sc->acquired) {
    nx::loge(log_rhi, "resize_swapchain between acquire and end_frame");
    return false;
  }
  if (sc->extent.width == width && sc->extent.height == height)
    return true;

  wait_idle();
  if (sc->offscreen) {
    destroy_texture(sc->backbuffer);
    const TextureHandle image = create_texture({
        .name = "offscreen backbuffer",
        .format = s->swapchain_format,
        .width = width,
        .height = height,
        .usage = TextureUsage::RenderTarget | TextureUsage::Sampled |
                 TextureUsage::CopySrc,
    });
    sc->backbuffer = image;
    sc->extent = image.valid() ? Extent2D{width, height} : Extent2D{};
    return image.valid();
  }
  if (sc->layer == nil)
    return false;

  @autoreleasepool {
    sc->layer.drawableSize = CGSizeMake(width, height);
  }
  sc->extent = {width, height};

  if (MetalTexture *backbuffer = s->textures.try_get(sc->backbuffer)) {
    backbuffer->width = width;
    backbuffer->height = height;
  }
  return true;
}

void MetalDevice::destroy_swapchain(const SwapchainHandle handle) noexcept {
  if (m_state == nullptr)
    return;
  MetalDeviceState *s = m_state;
  MetalSwapchain *sc = s->swapchains.try_get(handle);
  if (sc == nullptr)
    return;
  if (sc->acquired) {
    nx::loge(log_rhi, "destroy_swapchain between acquire and end_frame");
    return;
  }
  wait_idle();

  if (sc->offscreen) {
    // A texture this device made, so it goes back the way any texture does
    // rather than being erased from under its allocation.
    destroy_texture(sc->backbuffer);
    s->swapchains.erase(handle);
    return;
  }

  @autoreleasepool {
    sc->drawable = nil;
    s->textures.erase(sc->backbuffer);

#if TARGET_OS_OSX
    // Drop the layer from the view as well, otherwise it outlives the device it
    // was created against and keeps presenting a dead surface.
    CAMetalLayer *layer = sc->layer;
    NSView *view = (NSView *)sc->layer_host;
    if (view != nil) {
      on_main_thread(^{
        if (view.layer == layer) {
          view.layer = nil;
          view.wantsLayer = NO;
        }
      });
    }
#endif
    sc->layer_host = nil;
    sc->layer = nil;
  }
  s->swapchains.erase(handle);
}

u32 MetalDevice::frames_in_flight() const noexcept {
  return m_state != nullptr ? m_state->frames_in_flight : 0u;
}

Format MetalDevice::swapchain_format() const noexcept {
  return m_state != nullptr ? m_state->swapchain_format : Format::Unknown;
}

Extent2D MetalDevice::swapchain_extent(const SwapchainHandle handle) const {
  const MetalSwapchain *const sc =
      m_state != nullptr ? m_state->swapchains.try_get(handle) : nullptr;
  return sc != nullptr ? sc->extent : Extent2D{};
}

SurfaceAlpha MetalDevice::swapchain_alpha(const SwapchainHandle handle) const {
  const MetalSwapchain *const sc =
      m_state != nullptr ? m_state->swapchains.try_get(handle) : nullptr;
  return sc != nullptr ? sc->alpha : SurfaceAlpha::Opaque;
}

}
