
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

bool MetalDevice::create_swapchain(const SwapchainDesc &desc) {
  if (m_state == nullptr) {
    nx::loge(log_rhi, "create_swapchain before init");
    return false;
  }
  if (desc.width == 0 || desc.height == 0) {
    nx::loge(log_rhi, "create_swapchain: zero extent");
    return false;
  }

  destroy_swapchain();

  MetalDeviceState *s = m_state;
  const MTLPixelFormat pixel_format = pick_layer_format(desc.preferred_format);

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
        return false;
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
        return false;
      }
      layer.device = s->device;
      layer.pixelFormat = pixel_format;
      layer.framebufferOnly = NO;
      ok = true;
      break;
    }
#endif
    default:
      nx::loge(log_rhi, "create_swapchain: native window kind {} is not a "
                        "Metal surface",
               nx::cast<u32>(desc.window.kind));
      return false;
    }
  }

  if (!ok || layer == nil) {
    nx::loge(log_rhi, "create_swapchain: could not attach a CAMetalLayer");
    return false;
  }

  const u32 frames_in_flight = s->frames_in_flight;

  @autoreleasepool {
    layer.drawableSize = CGSizeMake(desc.width, desc.height);
    layer.maximumDrawableCount = nx::clamp<u32>(frames_in_flight + 1, 2, 3);
#if TARGET_OS_OSX
    layer.displaySyncEnabled = desc.present_mode != PresentMode::Immediate;
#endif
  }

  s->layer = layer;
  s->layer_host = layer_host;
  s->swapchain_format = from_mtl_format(pixel_format);
  s->extent = {desc.width, desc.height};
  s->present_mode = desc.present_mode;

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
  s->backbuffer = s->textures.insert(std::move(backbuffer));

  nx::logd(log_rhi, "swapchain: {}x{} {} ({} frames in flight, {})", desc.width,
           desc.height, to_string(s->swapchain_format), frames_in_flight,
           desc.present_mode == PresentMode::Immediate ? "no vsync" : "vsync");
  return true;
}

bool MetalDevice::create_offscreen_swapchain(const u32 width, const u32 height,
                                            const Format format) {
  if (m_state == nullptr || device_status() == GpuDeviceStatus::Lost)
    return false;
  if (width == 0 || height == 0 || format == Format::Unknown)
    return false;

  destroy_swapchain();

  MetalDeviceState *s = m_state;
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
    return false;

  s->backbuffer = image;
  s->swapchain_format = format;
  s->extent = {width, height};
  s->offscreen = true;

  nx::logd(log_rhi, "offscreen swapchain: {}x{} {}", width, height,
           to_string(format));
  return true;
}

bool MetalDevice::offscreen_swapchain() const noexcept {
  return m_state != nullptr && m_state->offscreen;
}

bool MetalDevice::resize_swapchain(const u32 width, const u32 height) {
  if (m_state == nullptr)
    return false;
  if (width == 0 || height == 0)
    return false;
  if (m_state->offscreen)
    return create_offscreen_swapchain(width, height,
                                      m_state->swapchain_format);
  if (m_state->layer == nil)
    return false;

  MetalDeviceState *s = m_state;
  if (s->extent.width == width && s->extent.height == height)
    return true;

  if (s->frame_active) {
    nx::loge(log_rhi, "resize_swapchain during a frame");
    return false;
  }

  wait_idle();

  @autoreleasepool {
    s->layer.drawableSize = CGSizeMake(width, height);
  }
  s->extent = {width, height};

  if (MetalTexture *backbuffer = s->textures.try_get(s->backbuffer)) {
    backbuffer->width = width;
    backbuffer->height = height;
  }
  return true;
}

void MetalDevice::destroy_swapchain() noexcept {
  if (m_state == nullptr)
    return;

  if (m_state->offscreen) {
    // A texture this device made, so it goes back the way any texture does
    // rather than being erased from under its allocation.
    MetalDeviceState *const state = m_state;
    wait_idle();
    destroy_texture(state->backbuffer);
    state->backbuffer = {};
    state->offscreen = false;
    state->frame_active = false;
    state->swapchain_format = Format::Unknown;
    state->extent = {};
    return;
  }

  if (m_state->layer == nil)
    return;

  MetalDeviceState *s = m_state;
  wait_idle();

  @autoreleasepool {
    for (MetalContextSlot &slot : s->contexts) {
      if (slot.encoder != nil) {
        [slot.encoder endEncoding];
        slot.encoder = nil;
      }
      if (slot.compute_encoder != nil) {
        [slot.compute_encoder endEncoding];
        slot.compute_encoder = nil;
      }
      slot.cmd = nil;
      slot.in_use = false;
    }
    s->used_contexts = 0;
    s->drawable = nil;
    s->frame_active = false;

    s->textures.erase(s->backbuffer);
    s->backbuffer = {};

#if TARGET_OS_OSX
    // Drop the layer from the view as well, otherwise it outlives the device it
    // was created against and keeps presenting a dead surface.
    CAMetalLayer *layer = s->layer;
    NSView *view = (NSView *)s->layer_host;
    if (view != nil) {
      on_main_thread(^{
        if (view.layer == layer) {
          view.layer = nil;
          view.wantsLayer = NO;
        }
      });
    }
#endif
    s->layer_host = nil;
    s->layer = nil;
  }

  // frame_semaphore is device-scoped (created once in init(), sized by
  // frames_in_flight) and outlives any number of swapchain destroy/recreate
  // cycles; niling it here left begin_frame() waiting on a nil semaphore
  // after the next create_swapchain(), which never recreated it.
  s->swapchain_format = Format::Unknown;
  s->extent = {};
}

u32 MetalDevice::frames_in_flight() const noexcept {
  return m_state != nullptr ? m_state->frames_in_flight : 0u;
}

Format MetalDevice::swapchain_format() const noexcept {
  return m_state != nullptr ? m_state->swapchain_format : Format::Unknown;
}

Extent2D MetalDevice::swapchain_extent() const noexcept {
  return m_state != nullptr ? m_state->extent : Extent2D{};
}

}
