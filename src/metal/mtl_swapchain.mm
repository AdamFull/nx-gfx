
#include "rendering/rhi/metal/mtl_internal.h"

#if TARGET_OS_OSX
#import <AppKit/AppKit.h>
#else
#import <UIKit/UIKit.h>
#endif

namespace nx::gfx::detail {
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
MTLPixelFormat pick_layer_format(const gfx::format preferred) noexcept {
  switch (preferred) {
  case gfx::format::BGRA8_UNORM:
  case gfx::format::BGRA8_SRGB:
  case gfx::format::RGBA16_FLOAT:
  case gfx::format::RGB10A2_UNORM:
    return to_mtl_format(preferred);
  case gfx::format::RGBA8_UNORM:
    return MTLPixelFormatBGRA8Unorm;
  case gfx::format::RGBA8_SRGB:
    return MTLPixelFormatBGRA8Unorm_sRGB;
  default:
    return MTLPixelFormatBGRA8Unorm;
  }
}

}

gfx::swapchain_handle MetalDevice::create_swapchain(const gfx::swapchain_desc &desc) {
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
  const bool opaque = desc.alpha == gfx::surface_alpha::Opaque;

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
    layer.displaySyncEnabled = desc.present_mode != gfx::present_mode::Immediate;
#endif
  }

  s->swapchain_format = from_mtl_format(pixel_format);
  MetalSwapchain sc;
  sc.layer = layer;
  sc.layer_host = layer_host;
  sc.extent = {desc.width, desc.height};
  sc.present_mode = desc.present_mode;
  sc.alpha = opaque ? gfx::surface_alpha::Opaque : gfx::surface_alpha::Premultiplied;

  // One handle for the backbuffer, valid for the swapchain's lifetime. Its
  // MTLTexture is filled in per frame from the drawable.
  MetalTexture backbuffer;
  backbuffer.format = s->swapchain_format;
  backbuffer.dimension = gfx::texture_dimension::Tex2D;
  backbuffer.width = desc.width;
  backbuffer.height = desc.height;
  backbuffer.usage = gfx::texture_usage::RenderTarget | gfx::texture_usage::CopySrc;
  backbuffer.state = gfx::resource_state::Undefined;
  backbuffer.owns_texture = false;
  sc.backbuffer = s->textures.insert(std::move(backbuffer));

  nx::logd(log_rhi, "swapchain: {}x{} {} ({} frames in flight, {})", desc.width,
           desc.height, to_string(s->swapchain_format), frames_in_flight,
           desc.present_mode == gfx::present_mode::Immediate ? "no vsync" : "vsync");
  return s->swapchains.insert(std::move(sc));
}

gfx::swapchain_handle MetalDevice::create_offscreen_swapchain(const u32 width,
                                                       const u32 height,
                                                       const gfx::format format) {
  if (m_state == nullptr || device_status() == gfx::gpu_device_status::Lost)
    return {};
  if (width == 0 || height == 0 || format == gfx::format::Unknown)
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
  const gfx::texture_handle image = create_texture({
      .name = "offscreen backbuffer",
      .format = format,
      .width = width,
      .height = height,
      .usage = gfx::texture_usage::RenderTarget | gfx::texture_usage::Sampled |
               gfx::texture_usage::CopySrc,
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

bool MetalDevice::offscreen_swapchain(const gfx::swapchain_handle handle) const {
  const MetalSwapchain *const sc =
      m_state != nullptr ? m_state->swapchains.try_get(handle) : nullptr;
  return sc != nullptr && sc->offscreen;
}

bool MetalDevice::resize_swapchain(const gfx::swapchain_handle handle,
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
    const gfx::texture_handle image = create_texture({
        .name = "offscreen backbuffer",
        .format = s->swapchain_format,
        .width = width,
        .height = height,
        .usage = gfx::texture_usage::RenderTarget | gfx::texture_usage::Sampled |
                 gfx::texture_usage::CopySrc,
    });
    sc->backbuffer = image;
    sc->extent = image.valid() ? gfx::extent2d{width, height} : gfx::extent2d{};
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

void MetalDevice::destroy_swapchain(const gfx::swapchain_handle handle) noexcept {
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

gfx::format MetalDevice::swapchain_format() const noexcept {
  return m_state != nullptr ? m_state->swapchain_format : gfx::format::Unknown;
}

gfx::extent2d MetalDevice::swapchain_extent(const gfx::swapchain_handle handle) const {
  const MetalSwapchain *const sc =
      m_state != nullptr ? m_state->swapchains.try_get(handle) : nullptr;
  return sc != nullptr ? sc->extent : gfx::extent2d{};
}

gfx::surface_alpha MetalDevice::swapchain_alpha(const gfx::swapchain_handle handle) const {
  const MetalSwapchain *const sc =
      m_state != nullptr ? m_state->swapchains.try_get(handle) : nullptr;
  return sc != nullptr ? sc->alpha : gfx::surface_alpha::Opaque;
}

}
