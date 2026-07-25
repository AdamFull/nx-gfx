/**
 * @file mtl_device.mm
 * @brief Metal device creation, resources, the frame loop and command recording.
 */

#include "rendering/rhi/metal/mtl_internal.h"

#include "foundation/strings/format.h"

#include <cstdlib>

namespace nxe::rhi {
namespace {

/// Metal's validation layer is opt-in through the environment and is read when
/// the device is created, so setting it here still takes effect - as long as no
/// MTLDevice has been made yet, which init() guarantees.
void enable_metal_validation() noexcept {
  if (::getenv("METAL_DEVICE_WRAPPER_TYPE") != nullptr)
    return; // Respect an explicit setting (including a deliberate 0).
  ::setenv("METAL_DEVICE_WRAPPER_TYPE", "1", 0);
  ::setenv("METAL_ERROR_MODE", "3", 0); // assert on API misuse
}

/// MTLScissorRect is a bare struct with no Make helper, and its fields are
/// unsigned: a negative offset would wrap to an enormous rectangle and trip
/// Metal's validation, so clamp instead.
MTLScissorRect to_mtl_scissor(const Rect2D &rect) noexcept {
  return MTLScissorRect{
      nx::cast<NSUInteger>(nx::max<i32>(0, rect.offset.x)),
      nx::cast<NSUInteger>(nx::max<i32>(0, rect.offset.y)),
      nx::cast<NSUInteger>(rect.extent.width),
      nx::cast<NSUInteger>(rect.extent.height),
  };
}

u32 max_texture_dimension(id<MTLDevice> device) noexcept {
  // Mac2 and Apple3 raised the 2D limit to 16384; everything older is 8192.
  if ([device supportsFamily:MTLGPUFamilyMac2] ||
      [device supportsFamily:MTLGPUFamilyApple3])
    return 16384;
  return 8192;
}

} // namespace

MetalDevice::~MetalDevice() { shutdown(); }

bool MetalDevice::init(const DeviceDesc &desc) {
  if (m_state != nullptr)
    return true;

  @autoreleasepool {
    if (desc.enable_validation)
      enable_metal_validation();

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      nx::loge(log_rhi, "init: no Metal device on this system");
      return false;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      nx::loge(log_rhi, "init: newCommandQueue failed");
      return false;
    }
    queue.label = @"nx2d graphics";

    auto *state = new MetalDeviceState();
    state->device = device;
    state->queue = queue;

    const char *name = [[device name] UTF8String];
    m_device_name = nx::string(name != nullptr ? name : "unknown Metal device");

    const MTLArgumentBuffersTier tier = [device argumentBuffersSupport];
    const bool tier2 = tier >= MTLArgumentBuffersTier2;
    const bool metal3 = [device supportsFamily:MTLGPUFamilyMetal3];

    m_caps = {};
    m_caps.device_name = m_device_name.view();
    m_caps.api_version = metal3 ? 3u : 2u;
    m_caps.discrete = ![device hasUnifiedMemory];

    // Bindless needs argument-buffer Tier 2: Tier 1 permits neither arrays of
    // resources in an argument buffer nor dynamic indexing into one, which is
    // the entire binding model. There is deliberately no Tier 1 fallback.
    m_caps.descriptor_indexing = tier2;
    // Metal has always allowed a shader to hold a `device T*`; reading a
    // buffer's address back on the CPU (MTLBuffer.gpuAddress) is Metal 3.
    m_caps.buffer_device_address = metal3;
    // Metal has no render pass or pipeline barrier objects at all: attachments
    // are named at encoder creation and hazards are tracked by the driver.
    m_caps.dynamic_rendering = true;
    m_caps.synchronization2 = true;
    m_caps.timeline_semaphores = true; // MTLSharedEvent

    m_caps.async_compute = true; // any number of MTLCommandQueues
    m_caps.dedicated_transfer = false; // no transfer-only queue exists
    m_caps.draw_indirect_count = metal3;
    m_caps.memoryless_attachments = [device supportsFamily:MTLGPUFamilyApple1];

    m_caps.max_texture_2d = max_texture_dimension(device);
    // Tier 2 allows 500,000 textures and 2048 samplers in one argument buffer;
    // our own ceilings are lower, and the heap is sized to the smaller of the
    // two.
    m_caps.max_bindless_sampled_images = tier2 ? MAX_BINDLESS_TEXTURES : 0;
    m_caps.max_bindless_storage_images = tier2 ? MAX_BINDLESS_RW_TEXTURES : 0;
    m_caps.max_bindless_samplers =
        tier2 ? nx::min<u32>(MAX_BINDLESS_SAMPLERS, 2048) : 0;
    // setVertexBytes / setFragmentBytes take up to 4 KiB, well above the
    // 128-byte push block the RHI standardises on.
    m_caps.max_push_constant_size = 4096;
    if (@available(macOS 10.12, iOS 16.0, *))
      m_caps.device_local_memory = [device recommendedMaxWorkingSetSize];
    // Metal reports GPU timestamps in nanoseconds already.
    m_caps.timestamp_period_ns = 1.f;

    m_state = state;
    m_uploader.m_state = state;

    nx::logi(log_rhi, "metal device: {} ({}, unified memory: {})",
             m_device_name, metal3 ? "Metal 3" : "Metal 2",
             [device hasUnifiedMemory] ? "yes" : "no");
    nx::logi(log_rhi, "argument buffers: tier {}", tier2 ? 2 : 1);
    if (!tier2) {
      nx::logw(log_rhi,
               "argument buffer tier 1: bindless is unavailable on this GPU, "
               "so textured draws will be skipped");
    }
    if (desc.enable_validation)
      nx::logi(log_rhi, "metal api validation requested");
  }

  return true;
}

void MetalDevice::shutdown() noexcept {
  if (m_state == nullptr)
    return;

  wait_idle();
  destroy_swapchain();

  @autoreleasepool {
    // ARC releases every id member when the state is deleted, but the resource
    // maps hold objects too and their destructors run in the same delete.
    m_uploader.m_state = nullptr;
    delete m_state;
    m_state = nullptr;
  }

  m_caps = {};
  m_device_name.clear();
}

void MetalDevice::wait_idle() noexcept {
  if (m_state == nullptr || m_state->queue == nil)
    return;

  // Metal has no device-wide idle wait. A command queue runs in submission
  // order, so an empty buffer that has completed proves everything committed
  // before it has too.
  @autoreleasepool {
    id<MTLCommandBuffer> cmd = [m_state->queue commandBuffer];
    cmd.label = @"wait_idle";
    [cmd commit];
    [cmd waitUntilCompleted];
  }
}

// Frame

AcquireResult MetalDevice::begin_frame(MetalCommandContext &out_context) {
  if (m_state == nullptr || m_state->layer == nil)
    return AcquireResult::SurfaceLost;

  MetalDeviceState *s = m_state;
  if (s->frame_active) {
    nx::loge(log_rhi, "begin_frame called twice without an end_frame");
    return AcquireResult::SurfaceLost;
  }

  // Block until a frame slot frees up. This is the counting semaphore that
  // stands in for Vulkan's per-frame timeline wait.
  dispatch_semaphore_wait(s->frame_semaphore, DISPATCH_TIME_FOREVER);

  @autoreleasepool {
    s->drawable = [s->layer nextDrawable];
    if (s->drawable == nil) {
      // The layer is mid-resize, or every drawable is still held by the
      // compositor. Neither is an error; drop the frame and try again.
      dispatch_semaphore_signal(s->frame_semaphore);
      return AcquireResult::OutOfDate;
    }

    s->cmd = [s->queue commandBuffer];
    if (s->cmd == nil) {
      s->drawable = nil;
      dispatch_semaphore_signal(s->frame_semaphore);
      return AcquireResult::SurfaceLost;
    }
    s->cmd.label = @"nx2d frame";
  }

  // Re-point the backbuffer handle at this frame's drawable. The handle is
  // stable across frames; the MTLTexture behind it is not.
  if (MetalTexture *backbuffer = s->textures.try_get(s->backbuffer)) {
    backbuffer->texture = s->drawable.texture;
    backbuffer->width = s->extent.width;
    backbuffer->height = s->extent.height;
    // Every drawable comes back undefined: its previous contents belong to a
    // frame the compositor has already consumed.
    backbuffer->state = ResourceState::Undefined;
  }

  s->frame_active = true;
  out_context.m_state = s;
  out_context.m_in_render_pass = false;

  // A drawable whose size no longer matches the layer means the surface moved
  // under us; the frame is still renderable, so report it and let the caller
  // rebuild afterwards.
  const CGSize drawable_size = s->layer.drawableSize;
  if (nx::cast<u32>(drawable_size.width) != s->extent.width ||
      nx::cast<u32>(drawable_size.height) != s->extent.height)
    return AcquireResult::Suboptimal;

  return AcquireResult::Success;
}

bool MetalDevice::end_frame() {
  MetalDeviceState *s = m_state;
  if (s == nullptr || !s->frame_active)
    return false;

  @autoreleasepool {
    if (s->encoder != nil) {
      // A caller that forgot end_render_pass would otherwise hit an assertion
      // inside Metal on commit; close it and say so.
      nx::logw(log_rhi, "end_frame with a render pass still open");
      [s->encoder endEncoding];
      s->encoder = nil;
    }

    // Captured by the block rather than reaching through `s`, so the handler
    // stays valid even if the device is torn down before the GPU catches up.
    dispatch_semaphore_t semaphore = s->frame_semaphore;
    [s->cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
      dispatch_semaphore_signal(semaphore);
    }];

    if (s->drawable != nil)
      [s->cmd presentDrawable:s->drawable];
    [s->cmd commit];
  }

  s->cmd = nil;
  s->drawable = nil;
  s->frame_active = false;
  ++s->frames_submitted;

  // The drawable's texture is the compositor's again the moment it is
  // presented; leaving an alias in the slot map would outlive it.
  if (MetalTexture *backbuffer = s->textures.try_get(s->backbuffer))
    backbuffer->texture = nil;

  return true;
}

TextureHandle MetalDevice::current_backbuffer() const noexcept {
  return m_state != nullptr ? m_state->backbuffer : TextureHandle{};
}

// Resources

TextureHandle MetalDevice::create_texture(const TextureDesc &desc) {
  if (m_state == nullptr)
    return {};

  if (desc.format == Format::Unknown || desc.width == 0 || desc.height == 0) {
    nx::loge(log_rhi, "create_texture('{}'): invalid format or extent",
             desc.name);
    return {};
  }

  const MTLPixelFormat pixel_format = to_mtl_format(desc.format);
  if (pixel_format == MTLPixelFormatInvalid) {
    nx::loge(log_rhi, "create_texture('{}'): {} has no Metal equivalent",
             desc.name, to_string(desc.format));
    return {};
  }

  const bool is_3d = desc.dimension == TextureDimension::Tex3D;
  const u32 layers = is_3d ? 1 : nx::max<u32>(1, desc.depth_or_layers);
  const u32 sample_count = nx::max<u32>(1, desc.sample_count);

  MetalTexture texture;
  @autoreleasepool {
    MTLTextureDescriptor *info = [[MTLTextureDescriptor alloc] init];
    info.textureType =
        to_mtl_texture_type(desc.dimension, layers, sample_count);
    info.pixelFormat = pixel_format;
    info.width = desc.width;
    info.height = desc.height;
    info.depth = is_3d ? nx::max<u32>(1, desc.depth_or_layers) : 1;
    info.mipmapLevelCount = nx::max<u32>(1, desc.mip_levels);
    // A cube's six faces are its slices, not its array length; Metal counts
    // arrayLength in whole cubes.
    info.arrayLength = desc.dimension == TextureDimension::TexCube
                           ? nx::max<u32>(1, layers / 6)
                           : layers;
    info.sampleCount = sample_count;
    info.usage = to_mtl_texture_usage(desc.usage);
    info.storageMode =
        to_mtl_storage_mode(desc.memory, desc.usage.has(TextureUsage::Transient));

    texture.texture = [m_state->device newTextureWithDescriptor:info];
    if (texture.texture == nil) {
      nx::loge(log_rhi, "create_texture('{}'): allocation failed", desc.name);
      return {};
    }
    texture.texture.label = to_ns_string(desc.name);
  }

  texture.format = desc.format;
  texture.dimension = desc.dimension;
  texture.width = desc.width;
  texture.height = desc.height;
  texture.mip_levels = nx::max<u32>(1, desc.mip_levels);
  texture.layers = layers;
  texture.sample_count = sample_count;
  texture.usage = desc.usage;
  texture.state = desc.initial_state;
  texture.owns_texture = true;

  return m_state->textures.insert(std::move(texture));
}

void MetalDevice::destroy_texture(const TextureHandle handle) noexcept {
  if (m_state == nullptr || !handle.valid())
    return;
  // TODO(metal): defer to the end of the frame once there is GPU work that can
  // still be referencing it. Nothing is recorded against textures yet.
  m_state->textures.erase(handle);
}

BufferHandle MetalDevice::create_buffer(const BufferDesc &desc) {
  if (m_state == nullptr)
    return {};

  if (desc.size == 0) {
    nx::loge(log_rhi, "create_buffer('{}'): zero size", desc.name);
    return {};
  }

  MetalBuffer buffer;
  @autoreleasepool {
    MTLResourceOptions options = MTLResourceStorageModePrivate;
    switch (desc.memory) {
    case MemoryUsage::Upload:
      options = MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
      break;
    case MemoryUsage::Readback:
      options = MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;
      break;
    case MemoryUsage::DeviceLocal:
    default:
      options = MTLResourceStorageModePrivate;
      break;
    }

    buffer.buffer = [m_state->device newBufferWithLength:desc.size
                                                 options:options];
    if (buffer.buffer == nil) {
      nx::loge(log_rhi, "create_buffer('{}'): allocation of {} bytes failed",
               desc.name, desc.size);
      return {};
    }
    buffer.buffer.label = to_ns_string(desc.name);

    // A private buffer has no CPU address at all; asking for contents on one
    // returns null rather than failing, so gate it on the storage mode.
    if (desc.persistently_mapped && desc.memory != MemoryUsage::DeviceLocal)
      buffer.mapped = [buffer.buffer contents];
  }

  buffer.size = desc.size;
  buffer.usage = desc.usage;
  return m_state->buffers.insert(std::move(buffer));
}

void MetalDevice::destroy_buffer(const BufferHandle handle) noexcept {
  if (m_state == nullptr || !handle.valid())
    return;
  m_state->buffers.erase(handle);
}

SamplerHandle MetalDevice::create_sampler(const SamplerDesc &desc) {
  if (m_state == nullptr)
    return {};

  MetalSampler sampler;
  @autoreleasepool {
    MTLSamplerDescriptor *info = [[MTLSamplerDescriptor alloc] init];
    info.minFilter = to_mtl_filter(desc.min_filter);
    info.magFilter = to_mtl_filter(desc.mag_filter);
    info.mipFilter = to_mtl_mip_filter(desc.mip_filter);
    info.sAddressMode = to_mtl_address(desc.address_u);
    info.tAddressMode = to_mtl_address(desc.address_v);
    info.rAddressMode = to_mtl_address(desc.address_w);
    info.lodMinClamp = desc.min_lod;
    info.lodMaxClamp = desc.max_lod;
    info.maxAnisotropy =
        nx::cast<NSUInteger>(nx::max<f32>(1.f, desc.max_anisotropy));
    info.compareFunction = to_mtl_compare(desc.compare);
    info.label = to_ns_string(desc.name);
    // Required before a sampler may be encoded into an argument buffer, which
    // is where every sampler ends up once the bindless heap exists. Setting it
    // on a Tier 1 device is not allowed, hence the capability check.
    info.supportArgumentBuffers = m_caps.descriptor_indexing ? YES : NO;

    sampler.sampler = [m_state->device newSamplerStateWithDescriptor:info];
    if (sampler.sampler == nil) {
      nx::loge(log_rhi, "create_sampler('{}'): creation failed", desc.name);
      return {};
    }
  }

  return m_state->samplers.insert(std::move(sampler));
}

void MetalDevice::destroy_sampler(const SamplerHandle handle) noexcept {
  if (m_state == nullptr || !handle.valid())
    return;
  m_state->samplers.erase(handle);
}

// Shaders and pipelines - not implemented yet.
//
// These return null handles rather than asserting, so that a caller written
// against the Vulkan backend degrades to the clear-only path instead of
// crashing. Slang already compiles the engine's shaders to metallib; what is
// missing is the pipeline state cache and the bindless argument buffer the
// shaders bind against.

ShaderHandle MetalDevice::create_shader(const ShaderDesc &desc) {
  nx::logw(log_rhi, "create_shader('{}'): the Metal backend has no pipeline "
                    "support yet",
           desc.name);
  return {};
}

void MetalDevice::destroy_shader(ShaderHandle) noexcept {}

PipelineHandle
MetalDevice::create_graphics_pipeline(const GraphicsPipelineDesc &desc) {
  nx::logw(log_rhi, "create_graphics_pipeline('{}'): not implemented on Metal "
                    "yet",
           desc.name);
  return {};
}

PipelineHandle
MetalDevice::create_compute_pipeline(const ComputePipelineDesc &desc) {
  nx::logw(log_rhi, "create_compute_pipeline('{}'): not implemented on Metal "
                    "yet",
           desc.name);
  return {};
}

void MetalDevice::destroy_pipeline(PipelineHandle) noexcept {}

// Bindless indices. The heap itself needs argument-buffer Tier 2 and has not
// been built yet, so nothing has a slot.

u32 MetalDevice::texture_index(TextureHandle) const noexcept {
  return BINDLESS_INVALID_INDEX;
}

u32 MetalDevice::rw_texture_index(TextureHandle, u32) const noexcept {
  return BINDLESS_INVALID_INDEX;
}

u32 MetalDevice::sampler_index(SamplerHandle) const noexcept {
  return BINDLESS_INVALID_INDEX;
}

u64 MetalDevice::buffer_address(const BufferHandle handle) const noexcept {
  if (m_state == nullptr)
    return 0;
  const MetalBuffer *buffer = m_state->buffers.try_get(handle);
  if (buffer == nullptr || buffer->buffer == nil)
    return 0;
  if (@available(macOS 13.0, iOS 16.0, *))
    return [buffer->buffer gpuAddress];
  return 0;
}

// MetalCommandContext

void MetalCommandContext::begin_render_pass(const RenderPassDesc &desc) {
  if (m_state == nullptr || m_state->cmd == nil)
    return;
  if (m_in_render_pass) {
    nx::loge(log_rhi, "begin_render_pass('{}') inside another render pass",
             desc.name);
    return;
  }

  @autoreleasepool {
    MTLRenderPassDescriptor *pass =
        [MTLRenderPassDescriptor renderPassDescriptor];

    u32 attached = 0;
    for (u32 i = 0; i < desc.color_count && i < MAX_COLOR_ATTACHMENTS; ++i) {
      const RenderAttachment &attachment = desc.color[i];
      const MetalTexture *texture =
          m_state->textures.try_get(attachment.texture);
      if (texture == nullptr || texture->texture == nil) {
        nx::loge(log_rhi, "begin_render_pass('{}'): colour attachment {} is not "
                          "a live texture",
                 desc.name, i);
        return;
      }

      MTLRenderPassColorAttachmentDescriptor *color = pass.colorAttachments[i];
      color.texture = texture->texture;
      color.level = attachment.mip_level;
      color.slice = attachment.base_layer;
      color.loadAction = to_mtl_load_action(attachment.load);
      color.storeAction = to_mtl_store_action(attachment.store);
      color.clearColor = MTLClearColorMake(
          attachment.clear.color[0], attachment.clear.color[1],
          attachment.clear.color[2], attachment.clear.color[3]);

      if (const MetalTexture *resolve =
              m_state->textures.try_get(attachment.resolve_target);
          resolve != nullptr && resolve->texture != nil) {
        color.resolveTexture = resolve->texture;
        color.storeAction = attachment.store == StoreOp::Store
                                ? MTLStoreActionStoreAndMultisampleResolve
                                : MTLStoreActionMultisampleResolve;
      }
      ++attached;
    }

    if (desc.has_depth_stencil) {
      const RenderAttachment &attachment = desc.depth_stencil;
      const MetalTexture *texture =
          m_state->textures.try_get(attachment.texture);
      if (texture == nullptr || texture->texture == nil) {
        nx::loge(log_rhi,
                 "begin_render_pass('{}'): depth attachment is not a live "
                 "texture",
                 desc.name);
        return;
      }

      pass.depthAttachment.texture = texture->texture;
      pass.depthAttachment.level = attachment.mip_level;
      pass.depthAttachment.slice = attachment.base_layer;
      pass.depthAttachment.loadAction = to_mtl_load_action(attachment.load);
      pass.depthAttachment.storeAction = to_mtl_store_action(attachment.store);
      pass.depthAttachment.clearDepth = attachment.clear.depth_stencil.depth;

      if (format_is_stencil(texture->format)) {
        pass.stencilAttachment.texture = texture->texture;
        pass.stencilAttachment.level = attachment.mip_level;
        pass.stencilAttachment.slice = attachment.base_layer;
        pass.stencilAttachment.loadAction = to_mtl_load_action(attachment.load);
        pass.stencilAttachment.storeAction =
            to_mtl_store_action(attachment.store);
        pass.stencilAttachment.clearStencil =
            attachment.clear.depth_stencil.stencil;
      }
      ++attached;
    }

    if (attached == 0) {
      nx::loge(log_rhi, "begin_render_pass('{}'): no attachments", desc.name);
      return;
    }

    m_state->encoder = [m_state->cmd renderCommandEncoderWithDescriptor:pass];
    if (m_state->encoder == nil) {
      nx::loge(log_rhi, "begin_render_pass('{}'): encoder creation failed",
               desc.name);
      return;
    }
    m_state->encoder.label = to_ns_string(desc.name);

    // Metal has no render area on the pass itself; it is a scissor rectangle.
    if (desc.render_area.extent.width != 0 &&
        desc.render_area.extent.height != 0)
      [m_state->encoder setScissorRect:to_mtl_scissor(desc.render_area)];
  }

  m_in_render_pass = true;
}

void MetalCommandContext::end_render_pass() {
  if (m_state == nullptr || !m_in_render_pass)
    return;

  if (m_state->encoder != nil) {
    [m_state->encoder endEncoding];
    m_state->encoder = nil;
  }
  m_in_render_pass = false;
}

void MetalCommandContext::set_viewport(const Viewport &viewport) {
  if (m_state == nullptr || m_state->encoder == nil)
    return;

  const MTLViewport mtl{viewport.x,         viewport.y,
                        viewport.width,     viewport.height,
                        viewport.min_depth, viewport.max_depth};
  [m_state->encoder setViewport:mtl];
}

void MetalCommandContext::set_scissor(const Rect2D &rect) {
  if (m_state == nullptr || m_state->encoder == nil)
    return;

  [m_state->encoder setScissorRect:to_mtl_scissor(rect)];
}

// Draw recording is inert until pipelines exist. create_graphics_pipeline
// never hands out a valid handle, so a caller cannot reach these with anything
// to draw with; they are silent rather than noisy because they would otherwise
// log once per draw per frame.

void MetalCommandContext::bind_pipeline(PipelineHandle) {}
void MetalCommandContext::push_constants(const void *, u32, u32) {}
void MetalCommandContext::draw(u32, u32, u32, u32) {}
void MetalCommandContext::draw_indexed(u32, u32, u32, i32, u32) {}
void MetalCommandContext::dispatch(u32, u32, u32) {}

void MetalCommandContext::barrier(const TextureBarrier *textures,
                                  const u32 texture_count,
                                  const BufferBarrier *buffers,
                                  const u32 buffer_count) {
  if (m_state == nullptr)
    return;

  // Metal tracks read-after-write hazards between encoders itself, so there is
  // no barrier to record. What the RHI still needs is the bookkeeping: a
  // texture's ResourceState decides the load action a later pass picks, and
  // whether the swapchain considers the backbuffer written.
  for (u32 i = 0; i < texture_count; ++i) {
    if (MetalTexture *texture = m_state->textures.try_get(textures[i].texture))
      texture->state = textures[i].to;
  }

  // Buffers carry no tracked state in this backend.
  (void)buffers;
  (void)buffer_count;
}

void MetalCommandContext::push_debug_group(const nx::string_view name,
                                           const f32 r, const f32 g,
                                           const f32 b) {
  if (m_state == nullptr)
    return;

  (void)r;
  (void)g;
  (void)b; // Metal debug groups carry no colour.

  @autoreleasepool {
    NSString *label = to_ns_string(name);
    if (m_state->encoder != nil)
      [m_state->encoder pushDebugGroup:label];
    else if (m_state->cmd != nil)
      [m_state->cmd pushDebugGroup:label];
  }
}

void MetalCommandContext::pop_debug_group() {
  if (m_state == nullptr)
    return;

  if (m_state->encoder != nil)
    [m_state->encoder popDebugGroup];
  else if (m_state->cmd != nil)
    [m_state->cmd popDebugGroup];
}

} // namespace nxe::rhi
