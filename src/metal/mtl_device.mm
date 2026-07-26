/**
 * @file mtl_device.mm
 * @brief Metal device creation, resources, the frame loop and command recording.
 */

#include "rendering/rhi/metal/mtl_internal.h"

#include "foundation/strings/format.h"

#include <cstdlib>

namespace nxe::rhi {
namespace {

/// Metal reads these when the device is created, so setting them here works as
/// long as no MTLDevice exists yet - which init() guarantees.
void enable_metal_validation() noexcept {
  if (::getenv("METAL_DEVICE_WRAPPER_TYPE") != nullptr)
    return; // Respect an explicit setting (including a deliberate 0).
  ::setenv("METAL_DEVICE_WRAPPER_TYPE", "1", 0);
  ::setenv("METAL_ERROR_MODE", "3", 0); // assert on API misuse
}

/// MTLScissorRect has no Make helper and unsigned fields, so a negative offset
/// would wrap to an enormous rectangle. Clamp instead.
MTLScissorRect to_mtl_scissor(const Rect2D &rect) noexcept {
  return MTLScissorRect{
      nx::cast<NSUInteger>(nx::max<i32>(0, rect.offset.x)),
      nx::cast<NSUInteger>(nx::max<i32>(0, rect.offset.y)),
      nx::cast<NSUInteger>(rect.extent.width),
      nx::cast<NSUInteger>(rect.extent.height),
  };
}

/// No Y flip on purpose. The RHI uses the Metal/D3D convention, which Vulkan
/// reaches via a negative-height viewport and Metal has natively. Flipping here
/// too would mirror Metal against Vulkan - invisible when rendering straight to
/// the backbuffer, obvious the moment a pass samples an offscreen target.
MTLViewport to_mtl_viewport(const f32 x, const f32 y, const f32 width,
                            const f32 height, const f32 min_depth,
                            const f32 max_depth) noexcept {
  return MTLViewport{x, y, width, height, min_depth, max_depth};
}

/// Releases bindless heaps that no longer hold anything.
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
      nx::logi(log_rhi, "bindless: heap released ({} MiB, {} MiB total)",
               size >> 20, bindless.heap_bytes >> 20);
      continue; // dropped: ARC frees it once the vector shrinks
    }
    if (kept != i)
      bindless.heaps[kept] = bindless.heaps[i];
    ++kept;
  }

  while (bindless.heaps.size() > kept)
    bindless.heaps.pop_back();
}

u32 max_texture_dimension(id<MTLDevice> device) noexcept {
  // Mac2 and Apple3 raised the 2D limit to 16384; everything older is 8192.
  if ([device supportsFamily:MTLGPUFamilyMac2] ||
      [device supportsFamily:MTLGPUFamilyApple3])
    return 16384;
  return 8192;
}

} // namespace

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
      // Heap-backed resources were never listed; the erase below returns their
      // memory to the heap.
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

  // Give emptied heaps back instead of holding 64 MiB each forever. Safe even
  // while an older frame still names one in useHeaps:, because command buffers
  // retain their references until they complete.
  sweep_empty_heaps(state);
}

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
    // Metal exposes no vendor or driver version. registryID is the closest
    // stable identifier - unique per GPU per boot - which is enough to tell two
    // adapters apart in a log, and better than leaving the field zero.
    m_caps.vendor_id = nx::cast<u32>([device registryID] & 0xFFFFFFFFu);
    m_caps.driver_version = m_caps.api_version;

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

    nx::logi(log_rhi, "metal device: {} ({}, unified memory: {})",
             m_device_name, metal3 ? "Metal 3" : "Metal 2",
             [device hasUnifiedMemory] ? "yes" : "no");
    nx::logi(log_rhi, "argument buffers: tier {}", tier2 ? 2 : 1);

    // The binding model is bindless or nothing: everything above the RHI
    // addresses resources by slot, and there is no non-bindless path to fall
    // back to. Refusing here is clearer than failing at the first draw.
    if (!create_bindless_heap(*state, m_caps)) {
      delete state;
      m_caps = {};
      m_device_name.clear();
      return false;
    }

    // Metal 3 writes argument buffers through MTLResourceID, which is how the
    // bindless heap fills its slots.
    if (!metal3) {
      nx::loge(log_rhi, "the Metal backend needs Metal 3 (macOS 13 / iOS 16) "
                        "for resource IDs in argument buffers");
      delete state;
      m_caps = {};
      m_device_name.clear();
      return false;
    }

    // Synchronisation for the untracked bindless heaps.
    state->pass_fence = [device newFence];
    state->upload_event = [device newEvent];
    state->frame_event = [device newEvent];
    if (state->pass_fence == nil || state->upload_event == nil ||
        state->frame_event == nil) {
      nx::loge(log_rhi, "init: could not create the heap synchronisation "
                        "primitives");
      delete state;
      m_caps = {};
      m_device_name.clear();
      return false;
    }
    state->pass_fence.label = @"nx2d pass";
    state->upload_event.label = @"nx2d uploads";
    state->frame_event.label = @"nx2d frame groups";

    create_pipeline_archive(*state, desc.pipeline_cache_path);

    state->owner = this;
    m_state = state;
    m_uploader.m_state = state;

    if (!uploader_init(*state)) {
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

  return true;
}

void MetalDevice::shutdown() noexcept {
  if (m_state == nullptr)
    return;

  // Written before teardown, while the archive still holds this run's
  // pipelines.
  save_pipeline_archive(*m_state);

  // Drain the decode worker and any batch still in flight before anything it
  // could be touching is destroyed.
  uploader_shutdown(*m_state);

  wait_idle();
  destroy_swapchain();
  // The GPU is idle, so every deferred delete is safe to run now regardless of
  // which frame queued it.
  collect_garbage(*m_state, ~0ull);

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

  // Return staging the GPU has finished with, then turn finished decodes into
  // textures and commit the upload batch. Committing here - before the frame's
  // own command buffer - is the whole synchronisation story on Metal: the
  // queue runs them in order, so anything uploaded above is ready below.
  // The semaphore just let us through, so at most frames_in_flight frames are
  // outstanding: everything older than that has run to completion.
  collect_garbage(*s, s->frames_submitted >= s->frames_in_flight
                          ? s->frames_submitted - s->frames_in_flight
                          : 0);
  uploader_reclaim(*s);
  uploader_pump(*s);

  @autoreleasepool {
    s->drawable = [s->layer nextDrawable];
    if (s->drawable == nil) {
      // The layer is mid-resize, or every drawable is still held by the
      // compositor. Neither is an error; drop the frame and try again.
      dispatch_semaphore_signal(s->frame_semaphore);
      return AcquireResult::OutOfDate;
    }

    // Every slot is released at end_frame; slot 0 is this context's.
    s->contexts.resize(MetalDeviceState::MAX_CONTEXTS);
    for (MetalContextSlot &slot : s->contexts)
      slot.in_use = false;
    s->used_contexts = 1;

    MetalContextSlot &slot = s->contexts[0];
    slot = MetalContextSlot{};
    slot.cmd = [s->queue commandBuffer];
    if (slot.cmd == nil) {
      s->drawable = nil;
      dispatch_semaphore_signal(s->frame_semaphore);
      return AcquireResult::SurfaceLost;
    }
    slot.cmd.label = @"nx2d frame";
    slot.in_use = true;

    // Fences carry nothing across command buffers; the uploader's writes are
    // ordered by this event instead. Encoded here, before any encoder exists -
    // a wait only orders the commands that follow it.
    begin_command_buffer(slot);
    if (s->upload.submitted > 0) {
      [slot.cmd encodeWaitForEvent:s->upload_event value:s->upload.submitted];
    }
    // Slot 0 opens the chain and waits for nothing, so frames stay free to
    // overlap; open_contexts() encodes the waits for the later groups.
    s->frame_event_base = s->frame_event_value;
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
  out_context.m_slot = 0;
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

    // Commit in slot order, and make the GPU honour that order with an event
    // chain. Commit order alone is not enough: buffers on a queue may overlap
    // execution, so a later group could read transients an earlier one had not
    // finished writing. Present and the completion signal go on the *last*
    // buffer - on slot 0 they would fire before the later groups ran.
    const usize last = s->used_contexts - 1;
    const u64 base = s->frame_event_base;
    s->frame_event_value = base + s->used_contexts;
    for (usize i = 0; i < s->used_contexts; ++i) {
      MetalContextSlot &slot = s->contexts[i];
      if (slot.cmd == nil)
        continue;

      // Signal only - the matching wait was encoded when the buffer was
      // created, the only position from which it can order anything.
      [slot.cmd encodeSignalEvent:s->frame_event value:base + i + 1];

      if (i == last) {
        // Captured by the block rather than reaching through `s`, so the
        // handler stays valid even if the device is torn down before the GPU
        // catches up.
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

namespace {

/// The MTLTextureDescriptor a TextureDesc maps to. Shared by the owning and the
/// heap-placed paths, so the two cannot drift - which would make the size and
/// alignment measured for one wrong for the other. Autoreleased: call inside an
/// @autoreleasepool.
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
  // A cube's six faces are its slices, not its array length; Metal counts
  // arrayLength in whole cubes.
  info.arrayLength = desc.dimension == TextureDimension::TexCube
                         ? nx::max<u32>(1, layers / 6)
                         : layers;
  info.sampleCount = sample_count;
  info.usage = to_mtl_texture_usage(desc.usage);
  info.storageMode =
      to_mtl_storage_mode(desc.memory, desc.usage.has(TextureUsage::Transient));
  return info;
}

} // namespace

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

    // Anything a shader reaches through the bindless tables comes from a heap
    // where possible, so that one useHeaps: per encoder covers it instead of
    // an entry in the per-resource residency list. A pure render target is
    // named by the render pass descriptor and needs neither.
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

  // A storage image is addressed one mip at a time, so each mip needs its own
  // view and its own slot in the rwTexture table.
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

  // The sampled slot *is* the handle index - that is the whole point of the
  // handle layout - so the descriptor can only be written once the slot map
  // has assigned one.
  MetalTexture &stored = m_state->textures[handle];
  if (desc.usage.has(TextureUsage::Sampled)) {
    stored.bindless_slot = bindless_index(handle);
    write_texture_descriptor(*m_state, stored.bindless_slot, stored.texture);
  }

  // Heap-backed resources are already covered by useHeaps:; only the ones that
  // fell back to a device allocation need naming individually.
  if (!stored.heap_backed && (desc.usage.has(TextureUsage::Sampled) ||
                              desc.usage.has(TextureUsage::Storage)))
    track_resident(*m_state, stored.texture);

  return handle;
}

// Memory aliasing
//
// MTLHeap with placement type is Metal's equivalent of a hand-suballocated
// VkDeviceMemory: the caller picks the offsets, and two textures placed on
// overlapping bytes alias. Metal has no memory-type constraint, so
// memory_type_bits stays all-ones.

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
    // Placement, not automatic: automatic heaps sub-allocate themselves and
    // would ignore the offsets the graph computed.
    info.type = MTLHeapTypePlacement;
    // Explicit: a heap defaults to untracked, and these hold every transient
    // the graph aliases. See the note in mtl_bindless.mm's grow_heaps.
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
    heap->heap = nil; // ARC releases once every placed texture has gone
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
  // Placed in a heap, so one useHeaps: covers residency - same as the
  // bindless-heap path in create_texture.
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
  // Deferred: the texture keeps its bindless slot and its slot_map entry until
  // the GPU has passed the frame being recorded. Freeing the slot now would
  // let the next create_texture hand the same index - hence the same
  // descriptor - to a different texture mid-flight.
  retire(*m_state, {.texture = handle});
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

    // A buffer a shader reaches by GPU address needs residency exactly as a
    // bindless texture does - naming it in a push block does not make its
    // memory reachable - so it goes in a heap for the same reason.
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

    // A private buffer has no CPU address at all; asking for contents on one
    // returns null rather than failing, so gate it on the storage mode.
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

// Bindless indices, for the push block. A texture's sampled slot and a
// sampler's slot are their handle indices; storage-image slots come from a
// free list because a texture owns one per mip.

bool MetalDevice::open_contexts(const std::span<MetalCommandContext> out) {
  if (m_state == nullptr || !m_state->frame_active || out.empty())
    return out.empty();

  MetalDeviceState *s = m_state;
  if (s->used_contexts + out.size() > MetalDeviceState::MAX_CONTEXTS) {
    nx::logw(log_rhi, "open_contexts: {} more would exceed the {} slot limit; "
                      "recording serially instead",
             out.size(), MetalDeviceState::MAX_CONTEXTS);
    return false;
  }

  const usize first = s->used_contexts;
  @autoreleasepool {
    for (usize i = 0; i < out.size(); ++i) {
      MetalContextSlot &slot = s->contexts[first + i];
      slot = MetalContextSlot{};
      slot.cmd = [s->queue commandBuffer];
      if (slot.cmd == nil) {
        // Keep whatever opened cleanly; end_frame still commits those in order.
        s->used_contexts = first + i;
        return false;
      }
      slot.cmd.label = @"nx2d frame (parallel)";
      slot.in_use = true;
      begin_command_buffer(slot);

      // Encoded here, not at commit time: a wait only orders the commands that
      // follow it in the buffer, so one encoded after the encoders orders
      // nothing. This is what makes group i run after group i-1.
      [slot.cmd encodeWaitForEvent:s->frame_event
                             value:s->frame_event_base + first + i];

      out[i].m_state = s;
      out[i].m_slot = nx::cast<u32>(first + i);
      out[i].m_in_render_pass = false;
    }
  }

  s->used_contexts = first + out.size();
  return true;
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

/// The recording slot this context owns. Contexts are handed out by
/// begin_frame and open_contexts and never outlive the frame, so the index is
/// always in range while one is valid.
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
    // Extent of the first attachment at its mip, which is what the default
    // viewport and scissor cover.
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

    // Heap resources carry no automatic tracking, so order against any earlier
    // encoder in this command buffer.
    encoder_wait_fence(*m_state, ctx_slot(), ctx_slot().encoder);

    // The bindless argument buffer is bound once per encoder and never
    // rebound, matching how the Vulkan backend binds its descriptor set once
    // per frame. Encoder state does not survive endEncoding, so "once per
    // frame" has to become "once per pass" here.
    bind_bindless(*m_state, ctx_slot().encoder);
    ctx_slot().has_bound_pipeline = false;

    // Metal's default viewport is not flipped, so one has to be set even when
    // the caller never touches it - otherwise every draw comes out mirrored.
    // The size is the first attachment's, at the mip the pass renders to.
    [ctx_slot().encoder
        setViewport:to_mtl_viewport(0.f, 0.f, nx::cast<f32>(pass_width),
                                    nx::cast<f32>(pass_height), 0.f, 1.f)];

    // Metal has no render area on the pass itself; it is a scissor rectangle.
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
    // Concurrent dispatch would let independent dispatches overlap, but it
    // also drops the automatic ordering between them. Serial matches what a
    // caller coming from vkCmdDispatch expects.
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

} // namespace nxe::rhi
