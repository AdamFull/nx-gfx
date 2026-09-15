
#include "rendering/rhi/metal/mtl_internal.h"
#include "rendering/rhi/metal/mtl_gpu_crash.h"

#include "core/foundation/strings/format.h"

#include <cstdlib>

namespace nxe::rhi {
namespace {

void enable_metal_validation() noexcept {
  if (::getenv("METAL_DEVICE_WRAPPER_TYPE") != nullptr)
    return;
  ::setenv("METAL_DEVICE_WRAPPER_TYPE", "1", 0);
  ::setenv("METAL_ERROR_MODE", "3", 0);
}

MTLScissorRect to_mtl_scissor(const Rect2D &rect) noexcept {
  return MTLScissorRect{
      nx::cast<NSUInteger>(nx::max<i32>(0, rect.offset.x)),
      nx::cast<NSUInteger>(nx::max<i32>(0, rect.offset.y)),
      nx::cast<NSUInteger>(rect.extent.width),
      nx::cast<NSUInteger>(rect.extent.height),
  };
}

MTLViewport to_mtl_viewport(const f32 x, const f32 y, const f32 width,
                            const f32 height, const f32 min_depth,
                            const f32 max_depth) noexcept {
  return MTLViewport{x, y, width, height, min_depth, max_depth};
}

void sweep_empty_heaps(MetalDeviceState &state) noexcept {
  BindlessHeap &bindless = state.bindless;
  if (bindless.heaps.empty())
    return;

  usize kept = 0;
  for (usize i = 0; i < bindless.heaps.size(); ++i) {
    id<MTLHeap> heap = bindless.heaps[i];
    // usedSize counts live sub-allocations; zero means every texture and
    // buffer that lived here has been released.
    if (heap != nil && [heap usedSize] == 0) {
      const u64 size = [heap size];
      bindless.heap_bytes -= nx::min<u64>(size, bindless.heap_bytes);
      nx::logd(log_rhi, "bindless: heap released ({} MiB, {} MiB total)",
               size >> 20, bindless.heap_bytes >> 20);
      continue;
    }
    if (kept != i)
      bindless.heaps[kept] = bindless.heaps[i];
    ++kept;
  }

  while (bindless.heaps.size() > kept)
    bindless.heaps.pop_back();
}

u32 max_texture_dimension(id<MTLDevice> device) noexcept {
  if ([device supportsFamily:MTLGPUFamilyMac2] ||
      [device supportsFamily:MTLGPUFamilyApple3])
    return 16384;
  return 8192;
}

}

/// Completion is derived from the frame semaphore: it admits at most
/// frames_in_flight outstanding frames, so frame T is done by the time
/// T + frames_in_flight is recorded.
void retire(MetalDeviceState &state,
            const MetalDeviceState::PendingDelete &pending) {
  MetalDeviceState::PendingDelete entry = pending;
  entry.frame = state.frames_submitted;
  state.pending_deletes.push_back(entry);
}

void collect_garbage(MetalDeviceState &state, const u64 completed_frame) {
  if (state.pending_deletes.empty())
    return;

  usize kept = 0;
  for (usize i = 0; i < state.pending_deletes.size(); ++i) {
    const MetalDeviceState::PendingDelete &pending = state.pending_deletes[i];
    if (pending.frame > completed_frame) {
      if (kept != i)
        state.pending_deletes[kept] = pending;
      ++kept;
      continue;
    }

    if (MetalTexture *texture = state.textures.try_get(pending.texture)) {
      if (texture->bindless_slot != BINDLESS_INVALID_INDEX)
        write_texture_descriptor(state, texture->bindless_slot, nil);
      for (const u32 slot : texture->uav_slots)
        free_rw_slot(state, slot);
      if (!texture->heap_backed)
        forget_resident(state, texture->texture);
      state.textures.erase(pending.texture);
    }

    if (const MetalBuffer *buffer = state.buffers.try_get(pending.buffer)) {
      if (!buffer->heap_backed && buffer->bindless_visible)
        forget_resident(state, buffer->buffer);
      state.buffers.erase(pending.buffer);
    }

    if (const MetalSampler *sampler = state.samplers.try_get(pending.sampler);
        sampler != nullptr) {
      if (sampler->bindless_slot != BINDLESS_INVALID_INDEX)
        write_sampler_descriptor(state, sampler->bindless_slot, nil);
      state.samplers.erase(pending.sampler);
    }

    state.pipelines.erase(pending.pipeline);
    state.shaders.erase(pending.shader);
  }

  while (state.pending_deletes.size() > kept)
    state.pending_deletes.pop_back();

  sweep_empty_heaps(state);
}

MetalDevice::~MetalDevice() { shutdown(); }

bool MetalDevice::init(const DeviceDesc &desc) {
  if (m_state != nullptr)
    return true;

  m_init_error.clear();
  m_init_report = {.failure = DeviceInitFailure::Failed};

  @autoreleasepool {
    if (desc.enable_validation)
      enable_metal_validation();

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      m_init_report.failure = DeviceInitFailure::Unsupported;
      m_init_error = "No Metal graphics device was found.";
      nx::loge(log_rhi, "init: no Metal device on this system");
      return false;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      m_init_error = "The Metal command queue could not be created.";
      nx::loge(log_rhi, "init: newCommandQueue failed");
      return false;
    }
    queue.label = @"nx2d graphics";

    auto *state = new MetalDeviceState();
    state->device = device;
    state->queue = queue;

    const char *name = [[device name] UTF8String];
    m_device_name = nx::string(name != nullptr ? name : "unknown Metal device");
    configure_gpu_crash_reporting(*state, desc.gpu_crash_mode, m_device_name);

    const MTLArgumentBuffersTier tier = [device argumentBuffersSupport];
    const bool tier2 = tier >= MTLArgumentBuffersTier2;
    const bool metal3 = [device supportsFamily:MTLGPUFamilyMetal3];

    m_caps = {};
    m_caps.device_name = m_device_name.view();
    m_caps.api_version = metal3 ? 3u : 2u;
    m_caps.discrete = ![device hasUnifiedMemory];
    m_caps.vendor_id = nx::cast<u32>([device registryID] & 0xFFFFFFFFu);
    m_caps.driver_version = m_caps.api_version;
    m_init_report.vendor_id = m_caps.vendor_id;

    m_caps.descriptor_indexing = tier2;
    m_caps.buffer_device_address = metal3;
    // Metal has no render pass or pipeline barrier objects at all: attachments
    // are named at encoder creation and hazards are tracked by the driver.
    m_caps.dynamic_rendering = true;
    m_caps.synchronization2 = true;
    m_caps.timeline_semaphores = true;

    // Unlike the three above, this isn't backed by an architectural argument
    // the way Vulkan's queue-family probe is - whether compute genuinely
    // overlaps render on a given GPU's scheduler is unverified here.
    m_caps.async_compute = true;
    m_caps.dedicated_transfer = false;
    m_caps.draw_indirect_count = metal3;
    m_caps.memoryless_attachments = [device supportsFamily:MTLGPUFamilyApple1];

    m_caps.max_texture_2d = max_texture_dimension(device);
    m_caps.max_buffer_size = nx::cast<u64>([device maxBufferLength]);
    // Metal accepts an NSUInteger-sized grid and exposes no smaller per-axis
    // device limit. The frame format stores u32 counts, so those are the
    // effective limits on both 32- and 64-bit Apple platforms.
    m_caps.max_compute_work_groups = {~0u, ~0u, ~0u};
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

    nx::logi(log_rhi, "metal device: {} ({}, unified memory: {})",
             m_device_name, metal3 ? "Metal 3" : "Metal 2",
             [device hasUnifiedMemory] ? "yes" : "no");
    nx::logd(log_rhi, "argument buffers: tier {}", tier2 ? 2 : 1);

    if (!metal3) {
      m_init_report.failure = DeviceInitFailure::Unsupported;
      m_init_error = nx::format(
          "{} does not support Metal 3. NX2D requires macOS 13 or iOS 16 "
          "and a Metal 3-capable graphics device.",
          m_device_name);
      nx::loge(log_rhi, "the Metal backend needs Metal 3 (macOS 13 / iOS 16) "
                        "for resource IDs in argument buffers");
      delete state;
      m_caps = {};
      m_device_name.clear();
      return false;
    }

    if (!create_bindless_heap(*state, m_caps)) {
      m_init_error = "The Metal bindless resource heap could not be created.";
      delete state;
      m_caps = {};
      m_device_name.clear();
      return false;
    }

    state->pass_fence = [device newFence];
    state->upload_event = [device newEvent];
    if (state->pass_fence == nil || state->upload_event == nil) {
      m_init_error =
          "The Metal synchronization resources could not be created.";
      nx::loge(log_rhi, "init: could not create the heap synchronisation "
                        "primitives");
      delete state;
      m_caps = {};
      m_device_name.clear();
      return false;
    }
    state->pass_fence.label = @"nx2d pass";
    state->upload_event.label = @"nx2d uploads";

    create_pipeline_archive(*state, desc.pipeline_cache_path);

    state->frames_in_flight = nx::clamp<u32>(desc.frames_in_flight, 1, 3);
    state->frame_semaphore =
        dispatch_semaphore_create(state->frames_in_flight);
    if (state->frame_semaphore == nil) {
      m_init_error = "The Metal frame pacing resources could not be created.";
      nx::loge(log_rhi, "init: could not create the frame pacing semaphore");
      delete state;
      m_caps = {};
      m_device_name.clear();
      return false;
    }

    state->owner = this;
    m_state = state;
    m_uploader.m_state = state;

    if (!uploader_init(*state)) {
      m_init_error = "The Metal upload service could not be initialized.";
      m_uploader.m_state = nullptr;
      m_state = nullptr;
      delete state;
      m_caps = {};
      m_device_name.clear();
      return false;
    }

    if (desc.enable_validation)
      nx::logi(log_rhi, "metal api validation requested");
  }

  m_init_report.failure = DeviceInitFailure::None;
  return true;
}

void MetalDevice::shutdown() noexcept {
  if (m_state == nullptr)
    return;

  save_pipeline_archive(*m_state);

  uploader_shutdown(*m_state);

  wait_idle();
  destroy_swapchain();
  collect_garbage(*m_state, ~0ull);

  @autoreleasepool {
    m_uploader.m_state = nullptr;
    delete m_state;
    m_state = nullptr;
  }

  m_caps = {};
  m_device_name.clear();
}

void MetalDevice::wait_idle() noexcept {
  if (m_state == nullptr || m_state->queue == nil ||
      device_status() == GpuDeviceStatus::Lost)
    return;

  @autoreleasepool {
    id<MTLCommandBuffer> cmd =
        make_command_buffer(*m_state, "wait_idle", m_state->frames_submitted);
    [cmd commit];
    [cmd waitUntilCompleted];
  }
}

GpuDeviceStatus MetalDevice::device_status() const noexcept {
  return m_state != nullptr ? gpu_device_status(*m_state)
                            : GpuDeviceStatus::Ready;
}

void MetalDevice::flush_uploads() noexcept {
  if (m_state == nullptr || m_state->frame_active)
    return;
  m_uploader.pump();
  m_uploader.flush_batch();
}

AcquireResult MetalDevice::begin_frame(MetalCommandContext &out_context) {
  if (m_state == nullptr || m_state->layer == nil ||
      device_status() == GpuDeviceStatus::Lost)
    return AcquireResult::SurfaceLost;

  MetalDeviceState *s = m_state;
  if (s->frame_active) {
    nx::loge(log_rhi, "begin_frame called twice without an end_frame");
    return AcquireResult::SurfaceLost;
  }

  dispatch_semaphore_wait(s->frame_semaphore, DISPATCH_TIME_FOREVER);
  if (device_status() == GpuDeviceStatus::Lost)
    return AcquireResult::SurfaceLost;

  collect_garbage(*s, s->frames_submitted >= s->frames_in_flight
                          ? s->frames_submitted - s->frames_in_flight
                          : 0);
  uploader_reclaim(*s);
  uploader_pump(*s);

  @autoreleasepool {
    s->drawable = [s->layer nextDrawable];
    if (s->drawable == nil) {
      dispatch_semaphore_signal(s->frame_semaphore);
      return AcquireResult::OutOfDate;
    }

    s->contexts.resize(MetalDeviceState::MAX_CONTEXTS);
    for (MetalContextSlot &slot : s->contexts)
      slot.in_use = false;
    s->used_contexts = 1;

    MetalContextSlot &slot = s->contexts[0];
    slot = MetalContextSlot{};
    slot.cmd = make_command_buffer(*s, "nx2d frame", s->frames_submitted);
    if (slot.cmd == nil) {
      s->drawable = nil;
      dispatch_semaphore_signal(s->frame_semaphore);
      return AcquireResult::SurfaceLost;
    }
    slot.in_use = true;

    begin_command_buffer(slot);
    if (s->upload.submitted > 0) {
      [slot.cmd encodeWaitForEvent:s->upload_event value:s->upload.submitted];
    }
  }

  // Re-point the backbuffer handle at this frame's drawable. The handle is
  // stable across frames; the MTLTexture behind it is not.
  if (MetalTexture *backbuffer = s->textures.try_get(s->backbuffer)) {
    backbuffer->texture = s->drawable.texture;
    backbuffer->width = s->extent.width;
    backbuffer->height = s->extent.height;
    backbuffer->state = ResourceState::Undefined;
  }

  s->frame_active = true;
  out_context.m_state = s;
  out_context.m_slot = 0;
  out_context.m_in_render_pass = false;

  const CGSize drawable_size = s->layer.drawableSize;
  if (nx::cast<u32>(drawable_size.width) != s->extent.width ||
      nx::cast<u32>(drawable_size.height) != s->extent.height)
    return AcquireResult::Suboptimal;

  return AcquireResult::Success;
}

bool MetalDevice::end_frame() {
  MetalDeviceState *s = m_state;
  if (s == nullptr || !s->frame_active ||
      device_status() == GpuDeviceStatus::Lost)
    return false;
  if (s->headless_frame) {
    nx::loge(log_rhi, "end_frame on a headless frame; use end_headless_frame");
    return false;
  }

  @autoreleasepool {
    // A caller that forgot to end a pass would otherwise hit an assertion
    // inside Metal on commit; close it and say so.
    for (usize i = 0; i < s->used_contexts; ++i) {
      MetalContextSlot &slot = s->contexts[i];
      if (slot.encoder != nil) {
        nx::logw(log_rhi, "end_frame with a render pass still open");
        encoder_update_fence(*s, slot, slot.encoder);
        [slot.encoder endEncoding];
        slot.encoder = nil;
      }
      if (slot.compute_encoder != nil) {
        nx::logw(log_rhi, "end_frame with a compute pass still open");
        encoder_update_fence(*s, slot, slot.compute_encoder);
        [slot.compute_encoder endEncoding];
        slot.compute_encoder = nil;
      }
    }

    const usize last = s->used_contexts - 1;
    for (usize i = 0; i < s->used_contexts; ++i) {
      MetalContextSlot &slot = s->contexts[i];
      if (slot.cmd == nil)
        continue;

      if (i == last) {
        dispatch_semaphore_t semaphore = s->frame_semaphore;
        [slot.cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
          dispatch_semaphore_signal(semaphore);
        }];
        if (s->drawable != nil)
          [slot.cmd presentDrawable:s->drawable];
      }
      [slot.cmd commit];
      slot.cmd = nil;
      slot.in_use = false;
    }
  }

  s->used_contexts = 0;
  s->drawable = nil;
  s->frame_active = false;
  ++s->frames_submitted;

  if (MetalTexture *backbuffer = s->textures.try_get(s->backbuffer))
    backbuffer->texture = nil;

  return true;
}

bool MetalDevice::begin_headless_frame(MetalCommandContext &out_context) {
  if (m_state == nullptr || m_state->queue == nil ||
      device_status() == GpuDeviceStatus::Lost)
    return false;

  MetalDeviceState *s = m_state;
  if (s->frame_active) {
    nx::loge(log_rhi, "begin_headless_frame with a frame already open");
    return false;
  }

  if (s->frame_semaphore == nil)
    return false;

  dispatch_semaphore_wait(s->frame_semaphore, DISPATCH_TIME_FOREVER);
  if (device_status() == GpuDeviceStatus::Lost)
    return false;

  collect_garbage(*s, s->frames_submitted >= s->frames_in_flight
                          ? s->frames_submitted - s->frames_in_flight
                          : 0);
  uploader_reclaim(*s);
  uploader_pump(*s);

  @autoreleasepool {
    s->contexts.resize(MetalDeviceState::MAX_CONTEXTS);
    for (MetalContextSlot &slot : s->contexts)
      slot.in_use = false;
    s->used_contexts = 1;

    MetalContextSlot &slot = s->contexts[0];
    slot = MetalContextSlot{};
    slot.cmd =
        make_command_buffer(*s, "nx2d headless frame", s->frames_submitted);
    if (slot.cmd == nil) {
      dispatch_semaphore_signal(s->frame_semaphore);
      return false;
    }
    slot.in_use = true;

    begin_command_buffer(slot);
    if (s->upload.submitted > 0) {
      [slot.cmd encodeWaitForEvent:s->upload_event value:s->upload.submitted];
    }
  }

  s->frame_active = true;
  s->headless_frame = true;
  out_context.m_state = s;
  out_context.m_slot = 0;
  out_context.m_in_render_pass = false;
  return true;
}

bool MetalDevice::end_headless_frame() {
  MetalDeviceState *s = m_state;
  if (s == nullptr || !s->frame_active || !s->headless_frame ||
      device_status() == GpuDeviceStatus::Lost)
    return false;

  id<MTLCommandBuffer> last_cmd = nil;

  @autoreleasepool {
    for (usize i = 0; i < s->used_contexts; ++i) {
      MetalContextSlot &slot = s->contexts[i];
      if (slot.encoder != nil) {
        nx::logw(log_rhi, "end_headless_frame with a render pass still open");
        encoder_update_fence(*s, slot, slot.encoder);
        [slot.encoder endEncoding];
        slot.encoder = nil;
      }
      if (slot.compute_encoder != nil) {
        nx::logw(log_rhi, "end_headless_frame with a compute pass still open");
        encoder_update_fence(*s, slot, slot.compute_encoder);
        [slot.compute_encoder endEncoding];
        slot.compute_encoder = nil;
      }
    }

    const usize last = s->used_contexts - 1;
    for (usize i = 0; i < s->used_contexts; ++i) {
      MetalContextSlot &slot = s->contexts[i];
      if (slot.cmd == nil)
        continue;

      if (i == last) {
        dispatch_semaphore_t semaphore = s->frame_semaphore;
        [slot.cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
          dispatch_semaphore_signal(semaphore);
        }];
        last_cmd = slot.cmd;
      }
      [slot.cmd commit];
      slot.cmd = nil;
      slot.in_use = false;
    }
  }

  if (last_cmd != nil)
    [last_cmd waitUntilCompleted];

  s->used_contexts = 0;
  s->frame_active = false;
  s->headless_frame = false;
  ++s->frames_submitted;
  return true;
}

TextureHandle MetalDevice::current_backbuffer() const noexcept {
  return m_state != nullptr ? m_state->backbuffer : TextureHandle{};
}

TextureUsageFlags
MetalDevice::texture_usage(const TextureHandle handle) const noexcept {
  if (m_state == nullptr)
    return TextureUsage::None;
  const MetalTexture *texture = m_state->textures.try_get(handle);
  return texture != nullptr ? texture->usage : TextureUsage::None;
}

namespace {

/// Shared by the owning and the heap-placed paths, so the two cannot drift - which would make the
/// size and alignment measured for one wrong for the other.
MTLTextureDescriptor *make_texture_descriptor(const TextureDesc &desc) {
  const bool is_3d = desc.dimension == TextureDimension::Tex3D;
  const u32 layers = is_3d ? 1 : nx::max<u32>(1, desc.depth_or_layers);
  const u32 sample_count = nx::max<u32>(1, desc.sample_count);

  MTLTextureDescriptor *info = [[MTLTextureDescriptor alloc] init];
  info.textureType = to_mtl_texture_type(desc.dimension, layers, sample_count);
  info.pixelFormat = to_mtl_format(desc.format);
  info.width = desc.width;
  info.height = desc.height;
  info.depth = is_3d ? nx::max<u32>(1, desc.depth_or_layers) : 1;
  info.mipmapLevelCount = nx::max<u32>(1, desc.mip_levels);
  info.arrayLength = desc.dimension == TextureDimension::TexCube
                         ? nx::max<u32>(1, layers / 6)
                         : layers;
  info.sampleCount = sample_count;
  info.usage = to_mtl_texture_usage(desc.usage);
  info.storageMode =
      to_mtl_storage_mode(desc.memory, desc.usage.has(TextureUsage::Transient));
  return info;
}

}

bool MetalDevice::supports(const Format format,
                           const TextureUsageFlags usage) const {
  if (m_state == nullptr || m_state->device == nil ||
      format == Format::Unknown ||
      to_mtl_format(format) == MTLPixelFormatInvalid)
    return false;

  TextureDesc desc;
  desc.name = "format support probe";
  desc.format = format;
  desc.width = 8;
  desc.height = 8;
  desc.usage = usage;

  @autoreleasepool {
    MTLTextureDescriptor *info = make_texture_descriptor(desc);
    id<MTLTexture> texture = [m_state->device newTextureWithDescriptor:info];
    return texture != nil;
  }
}

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
    MTLTextureDescriptor *info = make_texture_descriptor(desc);

    const bool bindless_visible = desc.usage.has(TextureUsage::Sampled) ||
                                  desc.usage.has(TextureUsage::Storage);
    if (bindless_visible) {
      texture.texture = alloc_bindless_texture(*m_state, info);
      texture.heap_backed = texture.texture != nil;
    }

    if (texture.texture == nil)
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

  if (desc.usage.has(TextureUsage::Storage)) {
    @autoreleasepool {
      for (u32 mip = 0; mip < texture.mip_levels; ++mip) {
        id<MTLTexture> view = [texture.texture
            newTextureViewWithPixelFormat:pixel_format
                              textureType:texture.texture.textureType
                                   levels:NSMakeRange(mip, 1)
                                   slices:NSMakeRange(0, layers)];
        const u32 slot = view != nil ? alloc_rw_slot(*m_state)
                                     : BINDLESS_INVALID_INDEX;
        texture.uavs.push_back(view);
        texture.uav_slots.push_back(slot);
        if (slot != BINDLESS_INVALID_INDEX)
          write_rw_texture_descriptor(*m_state, slot, view);
      }
    }
  }

  const TextureHandle handle = m_state->textures.insert(std::move(texture));
  if (!handle.valid())
    return {};

  MetalTexture &stored = m_state->textures[handle];
  if (desc.usage.has(TextureUsage::Sampled)) {
    stored.bindless_slot = bindless_index(handle);
    write_texture_descriptor(*m_state, stored.bindless_slot, stored.texture);
  }

  if (!stored.heap_backed && (desc.usage.has(TextureUsage::Sampled) ||
                              desc.usage.has(TextureUsage::Storage)))
    track_resident(*m_state, stored.texture);

  return handle;
}

// Memory aliasing MTLHeap with placement type is Metal's equivalent of a hand-suballocated
// VkDeviceMemory: the caller picks the offsets, and two textures placed on overlapping bytes alias.

MemoryRequirements
MetalDevice::texture_requirements(const TextureDesc &desc) const {
  if (m_state == nullptr)
    return {};

  MemoryRequirements out;
  @autoreleasepool {
    MTLTextureDescriptor *info = make_texture_descriptor(desc);
    const MTLSizeAndAlign sa =
        [m_state->device heapTextureSizeAndAlignWithDescriptor:info];
    out.size = sa.size;
    out.alignment = nx::max<u64>(1, sa.align);
  }
  return out;
}

HeapHandle MetalDevice::create_heap(const HeapDesc &desc) {
  if (m_state == nullptr || desc.size == 0) {
    nx::loge(log_rhi, "create_heap('{}'): zero size", desc.name);
    return {};
  }

  MetalHeap heap;
  @autoreleasepool {
    MTLHeapDescriptor *info = [[MTLHeapDescriptor alloc] init];
    info.size = desc.size;
    info.type = MTLHeapTypePlacement;
    info.hazardTrackingMode = MTLHazardTrackingModeTracked;
    info.storageMode = desc.memory == MemoryUsage::DeviceLocal
                           ? MTLStorageModePrivate
                           : MTLStorageModeShared;
    heap.heap = [m_state->device newHeapWithDescriptor:info];
    if (heap.heap == nil) {
      nx::loge(log_rhi, "create_heap('{}'): {} KiB allocation failed", desc.name,
               desc.size / 1024);
      return {};
    }
    heap.heap.label = to_ns_string(desc.name);
  }
  heap.size = desc.size;
  return m_state->heaps.insert(std::move(heap));
}

void MetalDevice::destroy_heap(const HeapHandle handle) noexcept {
  if (m_state == nullptr || !handle.valid())
    return;
  if (MetalHeap *heap = m_state->heaps.try_get(handle)) {
    heap->heap = nil;
    m_state->heaps.erase(handle);
  }
}

TextureHandle MetalDevice::create_texture_in_heap(const TextureDesc &desc,
                                                  const HeapHandle heap_handle,
                                                  const u64 offset) {
  if (m_state == nullptr)
    return {};
  MetalHeap *heap = m_state->heaps.try_get(heap_handle);
  if (heap == nullptr || heap->heap == nil ||
      desc.format == Format::Unknown || desc.width == 0 || desc.height == 0) {
    nx::loge(log_rhi, "create_texture_in_heap('{}'): invalid heap or extent",
             desc.name);
    return {};
  }

  const bool is_3d = desc.dimension == TextureDimension::Tex3D;
  const u32 layers = is_3d ? 1 : nx::max<u32>(1, desc.depth_or_layers);
  const u32 sample_count = nx::max<u32>(1, desc.sample_count);
  const MTLPixelFormat pixel_format = to_mtl_format(desc.format);

  MetalTexture texture;
  @autoreleasepool {
    MTLTextureDescriptor *info = make_texture_descriptor(desc);
    texture.texture = [heap->heap newTextureWithDescriptor:info offset:offset];
    if (texture.texture == nil) {
      nx::loge(log_rhi,
               "create_texture_in_heap('{}'): placement at offset {} failed",
               desc.name, offset);
      return {};
    }
    texture.texture.label = to_ns_string(desc.name);
  }
  texture.heap_backed = true;

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

  if (desc.usage.has(TextureUsage::Storage)) {
    @autoreleasepool {
      for (u32 mip = 0; mip < texture.mip_levels; ++mip) {
        id<MTLTexture> view = [texture.texture
            newTextureViewWithPixelFormat:pixel_format
                              textureType:texture.texture.textureType
                                   levels:NSMakeRange(mip, 1)
                                   slices:NSMakeRange(0, layers)];
        const u32 slot =
            view != nil ? alloc_rw_slot(*m_state) : BINDLESS_INVALID_INDEX;
        texture.uavs.push_back(view);
        texture.uav_slots.push_back(slot);
        if (slot != BINDLESS_INVALID_INDEX)
          write_rw_texture_descriptor(*m_state, slot, view);
      }
    }
  }

  const TextureHandle handle = m_state->textures.insert(std::move(texture));
  if (!handle.valid())
    return {};

  MetalTexture &stored = m_state->textures[handle];
  if (desc.usage.has(TextureUsage::Sampled)) {
    stored.bindless_slot = bindless_index(handle);
    write_texture_descriptor(*m_state, stored.bindless_slot, stored.texture);
  }
  return handle;
}

void MetalDevice::destroy_texture(const TextureHandle handle) noexcept {
  if (m_state == nullptr || !handle.valid())
    return;
  retire(*m_state, {.texture = handle});
}

bool MetalDevice::replace_texture(const TextureHandle target,
                                  const TextureHandle replacement) {
  if (m_state == nullptr || !target.valid() || !replacement.valid() ||
      target == replacement)
    return false;

  MetalTexture *const held = m_state->textures.try_get(target);
  MetalTexture *const fresh = m_state->textures.try_get(replacement);
  if (held == nullptr || fresh == nullptr || held->usage != fresh->usage ||
      held->usage.has(TextureUsage::Storage))
    return false;

  // Per-mip storage slots may be cached outside the device. Restrict adoption
  // to resources whose complete external identity is the stable texture
  // handle (and sampled slot), which is the editor asset-reload use case.
  wait_idle();
  std::swap(*held, *fresh);
  const u32 target_slot = bindless_index(target);
  const u32 replacement_slot = bindless_index(replacement);
  if (held->usage.has(TextureUsage::Sampled)) {
    held->bindless_slot = target_slot;
    write_texture_descriptor(*m_state, target_slot, held->texture);
  } else {
    held->bindless_slot = BINDLESS_INVALID_INDEX;
    write_texture_descriptor(*m_state, target_slot, nil);
  }

  // The old target now sits behind the unpublished replacement handle. Do not
  // let its retirement clear the target slot we just rebound.
  fresh->bindless_slot = BINDLESS_INVALID_INDEX;
  write_texture_descriptor(*m_state, replacement_slot, nil);
  retire(*m_state, {.texture = replacement});
  return true;
}

bool MetalDevice::set_texture_view_base_mip(const TextureHandle, u32) {
  // Not yet implemented on Metal: a base-mip SRV would need a retained
  // newTextureView bound into the sampled slot. Returning false makes the
  // engine upload the full mip chain (correct, just no streaming benefit here).
  return false;
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

    buffer.bindless_visible = desc.usage.has(BufferUsage::DeviceAddress);
    if (buffer.bindless_visible) {
      buffer.buffer = alloc_bindless_buffer(*m_state, desc.size, options);
      buffer.heap_backed = buffer.buffer != nil;
    }

    if (buffer.buffer == nil)
      buffer.buffer = [m_state->device newBufferWithLength:desc.size
                                                   options:options];

    if (buffer.buffer == nil) {
      nx::loge(log_rhi, "create_buffer('{}'): allocation of {} bytes failed",
               desc.name, desc.size);
      return {};
    }
    buffer.buffer.label = to_ns_string(desc.name);

    if (desc.persistently_mapped && desc.memory != MemoryUsage::DeviceLocal)
      buffer.mapped = [buffer.buffer contents];
  }

  buffer.size = desc.size;
  buffer.usage = desc.usage;

  const BufferHandle handle = m_state->buffers.insert(std::move(buffer));
  if (!handle.valid())
    return {};

  MetalBuffer &stored = m_state->buffers[handle];
  if (stored.bindless_visible && !stored.heap_backed)
    track_resident(*m_state, stored.buffer);

  return handle;
}

void MetalDevice::destroy_buffer(const BufferHandle handle) noexcept {
  if (m_state == nullptr || !handle.valid())
    return;
  retire(*m_state, {.buffer = handle});
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
    info.supportArgumentBuffers = m_caps.descriptor_indexing ? YES : NO;

    sampler.sampler = [m_state->device newSamplerStateWithDescriptor:info];
    if (sampler.sampler == nil) {
      nx::loge(log_rhi, "create_sampler('{}'): creation failed", desc.name);
      return {};
    }
  }

  const SamplerHandle handle = m_state->samplers.insert(std::move(sampler));
  if (!handle.valid())
    return {};

  MetalSampler &stored = m_state->samplers[handle];
  stored.bindless_slot = bindless_index(handle);
  write_sampler_descriptor(*m_state, stored.bindless_slot, stored.sampler);
  return handle;
}

void MetalDevice::destroy_sampler(const SamplerHandle handle) noexcept {
  if (m_state == nullptr || !handle.valid())
    return;
  retire(*m_state, {.sampler = handle});
}

void MetalDevice::set_texture_state(const TextureHandle handle,
                                    const ResourceState state) noexcept {
  if (m_state == nullptr)
    return;
  if (MetalTexture *texture = m_state->textures.try_get(handle))
    texture->state = state;
}

ResourceState
MetalDevice::texture_state(const TextureHandle handle) const noexcept {
  if (m_state == nullptr)
    return ResourceState::Undefined;
  const MetalTexture *texture = m_state->textures.try_get(handle);
  return texture != nullptr ? texture->state : ResourceState::Undefined;
}

u32 MetalDevice::texture_index(const TextureHandle handle) const noexcept {
  if (m_state == nullptr)
    return BINDLESS_INVALID_INDEX;
  const MetalTexture *texture = m_state->textures.try_get(handle);
  return texture != nullptr ? texture->bindless_slot : BINDLESS_INVALID_INDEX;
}

u32 MetalDevice::rw_texture_index(const TextureHandle handle,
                                  const u32 mip) const noexcept {
  if (m_state == nullptr)
    return BINDLESS_INVALID_INDEX;
  const MetalTexture *texture = m_state->textures.try_get(handle);
  if (texture == nullptr || mip >= texture->uav_slots.size())
    return BINDLESS_INVALID_INDEX;
  return texture->uav_slots[mip];
}

u32 MetalDevice::sampler_index(const SamplerHandle handle) const noexcept {
  if (m_state == nullptr)
    return BINDLESS_INVALID_INDEX;
  const MetalSampler *sampler = m_state->samplers.try_get(handle);
  return sampler != nullptr ? sampler->bindless_slot : BINDLESS_INVALID_INDEX;
}

void *MetalDevice::buffer_mapped(const BufferHandle handle) const noexcept {
  if (m_state == nullptr)
    return nullptr;
  const MetalBuffer *buffer = m_state->buffers.try_get(handle);
  return buffer != nullptr ? buffer->mapped : nullptr;
}

bool MetalCommandContext::copy_texture_to_buffer(
    const TextureHandle texture_handle, const BufferHandle buffer_handle,
    const u64 row_pitch, const u32 mip, const u32 layer) {
  if (m_state == nullptr || m_in_render_pass)
    return false;
  MetalContextSlot &slot = ctx_slot();
  if (slot.cmd == nil || slot.encoder != nil || slot.compute_encoder != nil)
    return false;
  const MetalTexture *texture = m_state->textures.try_get(texture_handle);
  const MetalBuffer *buffer = m_state->buffers.try_get(buffer_handle);
  if (texture == nullptr || texture->texture == nil || buffer == nullptr ||
      buffer->buffer == nil || !(texture->usage & TextureUsage::CopySrc) ||
      !(buffer->usage & BufferUsage::CopyDst) ||
      texture->state != ResourceState::CopySrc ||
      format_is_depth(texture->format) || format_is_compressed(texture->format) ||
      mip >= texture->mip_levels || layer >= texture->layers)
    return false;

  const u32 width = nx::max<u32>(1, texture->width >> mip);
  const u32 height = nx::max<u32>(1, texture->height >> mip);
  const u64 tight = format_row_pitch(texture->format, width);
  if (row_pitch < tight || row_pitch > buffer->size / height)
    return false;

  @autoreleasepool {
    id<MTLBlitCommandEncoder> blit = [slot.cmd blitCommandEncoder];
    if (blit == nil)
      return false;
    blit.label = @"nx2d frame capture";
    [blit copyFromTexture:texture->texture
              sourceSlice:layer
              sourceLevel:mip
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(width, height, 1)
                 toBuffer:buffer->buffer
        destinationOffset:0
   destinationBytesPerRow:row_pitch
 destinationBytesPerImage:row_pitch * height];
    [blit endEncoding];
  }
  return true;
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

MetalContextSlot &MetalCommandContext::ctx_slot() const noexcept {
  return m_state->contexts[m_slot];
}

void MetalCommandContext::begin_render_pass(const RenderPassDesc &desc) {
  if (m_state == nullptr || ctx_slot().cmd == nil)
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
    u32 pass_width = 0;
    u32 pass_height = 0;

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

      if (attached == 0) {
        pass_width = nx::max<u32>(1, texture->width >> attachment.mip_level);
        pass_height = nx::max<u32>(1, texture->height >> attachment.mip_level);
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

      if (attached == 0) {
        pass_width = nx::max<u32>(1, texture->width >> attachment.mip_level);
        pass_height = nx::max<u32>(1, texture->height >> attachment.mip_level);
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

    ctx_slot().encoder = [ctx_slot().cmd renderCommandEncoderWithDescriptor:pass];
    if (ctx_slot().encoder == nil) {
      nx::loge(log_rhi, "begin_render_pass('{}'): encoder creation failed",
               desc.name);
      return;
    }
    ctx_slot().encoder.label = to_ns_string(desc.name);

    encoder_wait_fence(*m_state, ctx_slot(), ctx_slot().encoder);

    // The bindless argument buffer is bound once per encoder and never rebound, matching how the
    // Vulkan backend binds its descriptor set once per frame.
    bind_bindless(*m_state, ctx_slot().encoder);
    ctx_slot().has_bound_pipeline = false;

    // Metal's default viewport is not flipped, so one has to be set even when
    // the caller never touches it - otherwise every draw comes out mirrored.
    // The size is the first attachment's, at the mip the pass renders to.
    [ctx_slot().encoder
        setViewport:to_mtl_viewport(0.f, 0.f, nx::cast<f32>(pass_width),
                                    nx::cast<f32>(pass_height), 0.f, 1.f)];

    if (desc.render_area.extent.width != 0 &&
        desc.render_area.extent.height != 0)
      [ctx_slot().encoder setScissorRect:to_mtl_scissor(desc.render_area)];
  }

  m_in_render_pass = true;
}

void MetalCommandContext::end_render_pass() {
  if (m_state == nullptr || !m_in_render_pass)
    return;

  if (ctx_slot().encoder != nil) {
    encoder_update_fence(*m_state, ctx_slot(), ctx_slot().encoder);
    [ctx_slot().encoder endEncoding];
    ctx_slot().encoder = nil;
  }
  ctx_slot().has_bound_pipeline = false;
  m_in_render_pass = false;
}

void MetalCommandContext::begin_compute_pass(const nx::string_view name) {
  if (m_state == nullptr || ctx_slot().cmd == nil)
    return;
  if (m_in_render_pass || ctx_slot().encoder != nil) {
    nx::loge(log_rhi, "begin_compute_pass('{}') inside a render pass", name);
    return;
  }
  if (ctx_slot().compute_encoder != nil) {
    nx::loge(log_rhi, "begin_compute_pass('{}') inside another compute pass",
             name);
    return;
  }

  @autoreleasepool {
    ctx_slot().compute_encoder = [ctx_slot().cmd
        computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    if (ctx_slot().compute_encoder == nil) {
      nx::loge(log_rhi, "begin_compute_pass('{}'): encoder creation failed",
               name);
      return;
    }
    ctx_slot().compute_encoder.label = to_ns_string(name);

    encoder_wait_fence(*m_state, ctx_slot(), ctx_slot().compute_encoder);
    bind_bindless_compute(*m_state, ctx_slot().compute_encoder);
  }
  ctx_slot().has_bound_pipeline = false;
}

void MetalCommandContext::end_compute_pass() {
  if (m_state == nullptr || ctx_slot().compute_encoder == nil)
    return;

  encoder_update_fence(*m_state, ctx_slot(), ctx_slot().compute_encoder);
  [ctx_slot().compute_encoder endEncoding];
  ctx_slot().compute_encoder = nil;
  ctx_slot().has_bound_pipeline = false;
}

void MetalCommandContext::set_viewport(const Viewport &viewport) {
  if (m_state == nullptr || ctx_slot().encoder == nil)
    return;

  [ctx_slot().encoder setViewport:to_mtl_viewport(viewport.x, viewport.y,
                                                viewport.width, viewport.height,
                                                viewport.min_depth,
                                                viewport.max_depth)];
}

void MetalCommandContext::set_scissor(const Rect2D &rect) {
  if (m_state == nullptr || ctx_slot().encoder == nil)
    return;

  [ctx_slot().encoder setScissorRect:to_mtl_scissor(rect)];
}

void MetalCommandContext::barrier(const TextureBarrier *textures,
                                  const u32 texture_count,
                                  const BufferBarrier *buffers,
                                  const u32 buffer_count,
                                  const StateTracking tracking) {
  if (m_state == nullptr || tracking == StateTracking::Skip)
    return;

  // Metal tracks read-after-write hazards between encoders itself, so there is no barrier to
  // record.
  for (u32 i = 0; i < texture_count; ++i) {
    if (MetalTexture *texture = m_state->textures.try_get(textures[i].texture))
      texture->state = textures[i].to;
  }

  (void)buffers;
  (void)buffer_count;
}

void MetalCommandContext::push_debug_group(const nx::string_view name,
                                           const f32 r, const f32 g,
                                           const f32 b) {
  if (m_state == nullptr)
    return;

  // pushDebugGroup: takes a label only - Metal has no per-group colour, unlike
  // Vulkan's debug-utils labels. Kept as parameters so callers stay portable.
  (void)r;
  (void)g;
  (void)b;

  @autoreleasepool {
    NSString *label = to_ns_string(name);
    if (ctx_slot().encoder != nil)
      [ctx_slot().encoder pushDebugGroup:label];
    else if (ctx_slot().cmd != nil)
      [ctx_slot().cmd pushDebugGroup:label];
  }
}

void MetalCommandContext::pop_debug_group() {
  if (m_state == nullptr)
    return;

  if (ctx_slot().encoder != nil)
    [ctx_slot().encoder popDebugGroup];
  else if (ctx_slot().cmd != nil)
    [ctx_slot().cmd popDebugGroup];
}

}
