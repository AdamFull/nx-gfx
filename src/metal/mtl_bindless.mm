/**
 * @file mtl_bindless.mm
 * @brief The single bindless argument buffer.
 *
 * Slang lowers ParameterBlock<BindlessTables> to one argument buffer holding
 * textures, rw_textures and samplers back to back, 8 bytes per entry, so the
 * CPU writes it directly - the layout comes from the array bounds in
 * shaders/bindless.slang.
 *
 * Two things Vulkan does not need: Tier 2 argument buffers (Tier 1 has neither
 * resource arrays nor dynamic indexing, so init() refuses the device), and
 * residency - naming a resource here does not make it reachable, which is what
 * the heaps and `resident` list are for.
 */

#include "core/rendering/rhi/metal/mtl_internal.h"

namespace nxe::rhi {
namespace {

/// Every resource handle in a Tier 2 argument buffer is one MTLResourceID.
/// Asserted against the shader's own view of the layout below.
constexpr u64 SLOT_SIZE = sizeof(MTLResourceID);
static_assert(SLOT_SIZE == 8, "argument buffer slot size changed");

void write_slot(const MetalDeviceState &state, const u64 offset, const u32 slot,
                const u32 max_slots, const MTLResourceID id) noexcept {
  if (slot >= max_slots || state.bindless.argument_buffer == nil)
    return;
  auto *base = static_cast<u8 *>([state.bindless.argument_buffer contents]);
  std::memcpy(base + offset + nx::cast<u64>(slot) * SLOT_SIZE, &id,
              sizeof(id));
}

/// New heap size unless one resource needs more: big enough for a 2D game's
/// atlas set, small enough not to reserve absurd address space up front.
constexpr u64 DEFAULT_HEAP_BYTES = 64ull << 20;

/// Adds a heap big enough for @p required bytes and returns it, or nil.
id<MTLHeap> grow_heaps(MetalDeviceState &state, const u64 required) noexcept {
  MTLHeapDescriptor *info = [[MTLHeapDescriptor alloc] init];
  info.size = nx::max<u64>(DEFAULT_HEAP_BYTES, required);
  info.storageMode = MTLStorageModePrivate;
  info.type = MTLHeapTypeAutomatic;
  // Tracked, deliberately. Untracked would need the backend to place every
  // cross-encoder dependency itself, and getting that wrong left heap resources
  // with no synchronisation at all - survivable on an immediate-mode desktop
  // GPU, but it corrupts on a tile-based one.
  info.hazardTrackingMode = MTLHazardTrackingModeTracked;

  id<MTLHeap> heap = [state.device newHeapWithDescriptor:info];
  if (heap == nil) {
    nx::logw(log_rhi, "bindless: could not allocate a {} MiB heap; falling "
                      "back to per-resource residency",
             info.size >> 20);
    return nil;
  }
  heap.label = @"nx2d bindless";

  state.bindless.heaps.push_back(heap);
  state.bindless.heap_bytes += info.size;
  nx::logi(log_rhi, "bindless: heap {} added ({} MiB, {} MiB total)",
           state.bindless.heaps.size(), info.size >> 20,
           state.bindless.heap_bytes >> 20);
  return heap;
}

void clear_slot(const MetalDeviceState &state, const u64 offset, const u32 slot,
                const u32 max_slots) noexcept {
  if (slot >= max_slots || state.bindless.argument_buffer == nil)
    return;
  // A zeroed MTLResourceID is the null resource. Sampling through it returns
  // zero rather than faulting, which is what a freed slot should do.
  auto *base = static_cast<u8 *>([state.bindless.argument_buffer contents]);
  std::memset(base + offset + nx::cast<u64>(slot) * SLOT_SIZE, 0, SLOT_SIZE);
}

} // namespace

bool create_bindless_heap(MetalDeviceState &state, const DeviceCaps &caps) {
  if (!caps.descriptor_indexing) {
    nx::loge(log_rhi,
             "bindless: this GPU only supports argument buffer tier 1, which "
             "has neither resource arrays nor dynamic indexing");
    return false;
  }

  BindlessHeap &heap = state.bindless;

  // These must match the array bounds in shaders/bindless.slang: the shader's
  // struct layout is what fixes the offsets, and nothing at run time can
  // discover them.
  heap.max_textures = MAX_BINDLESS_TEXTURES;
  heap.max_rw_textures = MAX_BINDLESS_RW_TEXTURES;
  heap.max_samplers = MAX_BINDLESS_SAMPLERS;

  // Order must match the declaration order of BindlessTables.
  heap.textures_offset = 0;
  heap.rw_textures_offset =
      heap.textures_offset + nx::cast<u64>(heap.max_textures) * SLOT_SIZE;
  heap.samplers_offset =
      heap.rw_textures_offset + nx::cast<u64>(heap.max_rw_textures) * SLOT_SIZE;
  const u64 total =
      heap.samplers_offset + nx::cast<u64>(heap.max_samplers) * SLOT_SIZE;

  @autoreleasepool {
    // Shared so the slot writes above are plain stores. The buffer is written
    // far more often than it is read, and it is small enough that keeping it
    // in shared memory costs nothing measurable.
    heap.argument_buffer =
        [state.device newBufferWithLength:total
                                  options:MTLResourceStorageModeShared];
    if (heap.argument_buffer == nil) {
      nx::loge(log_rhi, "bindless: could not allocate a {} byte argument buffer",
               total);
      return false;
    }
    heap.argument_buffer.label = @"nx2d bindless";
    std::memset([heap.argument_buffer contents], 0, nx::cast<usize>(total));
  }

  nx::logi(log_rhi,
           "bindless heap: {} textures / {} rw textures / {} samplers "
           "({} KiB)",
           heap.max_textures, heap.max_rw_textures, heap.max_samplers,
           total / 1024);
  return true;
}

void write_texture_descriptor(MetalDeviceState &state, const u32 slot,
                              id<MTLTexture> texture) noexcept {
  const BindlessHeap &heap = state.bindless;
  if (texture == nil) {
    clear_slot(state, heap.textures_offset, slot, heap.max_textures);
    return;
  }
  write_slot(state, heap.textures_offset, slot, heap.max_textures,
             texture.gpuResourceID);
}

void write_rw_texture_descriptor(MetalDeviceState &state, const u32 slot,
                                 id<MTLTexture> texture) noexcept {
  const BindlessHeap &heap = state.bindless;
  if (texture == nil) {
    clear_slot(state, heap.rw_textures_offset, slot, heap.max_rw_textures);
    return;
  }
  write_slot(state, heap.rw_textures_offset, slot, heap.max_rw_textures,
             texture.gpuResourceID);
}

void write_sampler_descriptor(MetalDeviceState &state, const u32 slot,
                              id<MTLSamplerState> sampler) noexcept {
  const BindlessHeap &heap = state.bindless;
  if (sampler == nil) {
    clear_slot(state, heap.samplers_offset, slot, heap.max_samplers);
    return;
  }
  // Only valid when the sampler was created with supportArgumentBuffers set,
  // which create_sampler does whenever bindless is available.
  write_slot(state, heap.samplers_offset, slot, heap.max_samplers,
             sampler.gpuResourceID);
}

u32 alloc_rw_slot(MetalDeviceState &state) noexcept {
  BindlessHeap &heap = state.bindless;
  if (!heap.free_rw_slots.empty()) {
    const u32 slot = heap.free_rw_slots.back();
    heap.free_rw_slots.pop_back();
    return slot;
  }
  if (heap.rw_high_water >= heap.max_rw_textures) {
    nx::loge(log_rhi, "bindless: out of storage-image slots ({} in use)",
             heap.max_rw_textures);
    return BINDLESS_INVALID_INDEX;
  }
  return heap.rw_high_water++;
}

void free_rw_slot(MetalDeviceState &state, const u32 slot) noexcept {
  if (slot == BINDLESS_INVALID_INDEX)
    return;
  write_rw_texture_descriptor(state, slot, nil);
  state.bindless.free_rw_slots.push_back(slot);
}

id<MTLTexture>
alloc_bindless_texture(MetalDeviceState &state,
                       MTLTextureDescriptor *descriptor) noexcept {
  // The heaps are private-storage. A shared or memoryless texture would need a
  // heap of its own kind, which is not worth it for the handful that exist.
  if (descriptor.storageMode != MTLStorageModePrivate)
    return nil;

  for (id<MTLHeap> heap : state.bindless.heaps) {
    if (id<MTLTexture> texture = [heap newTextureWithDescriptor:descriptor])
      return texture;
  }

  // Every heap is full or too fragmented. Size the new one so this allocation
  // certainly fits, alignment included.
  const MTLSizeAndAlign need =
      [state.device heapTextureSizeAndAlignWithDescriptor:descriptor];
  id<MTLHeap> heap = grow_heaps(state, need.size + need.align);
  return heap != nil ? [heap newTextureWithDescriptor:descriptor] : nil;
}

id<MTLBuffer> alloc_bindless_buffer(MetalDeviceState &state, const u64 length,
                                    const MTLResourceOptions options) noexcept {
  if ((options & MTLResourceStorageModeMask) != MTLResourceStorageModePrivate)
    return nil;

  for (id<MTLHeap> heap : state.bindless.heaps) {
    if (id<MTLBuffer> buffer = [heap newBufferWithLength:length
                                                 options:options])
      return buffer;
  }

  const MTLSizeAndAlign need =
      [state.device heapBufferSizeAndAlignWithLength:length options:options];
  id<MTLHeap> heap = grow_heaps(state, need.size + need.align);
  return heap != nil ? [heap newBufferWithLength:length options:options] : nil;
}

// Fences

void begin_command_buffer(MetalContextSlot &slot) noexcept {
  slot.fence_written = false;
}

void encoder_wait_fence(MetalDeviceState &state, MetalContextSlot &slot,
                        id<MTLRenderCommandEncoder> encoder) noexcept {
  if (encoder == nil || state.pass_fence == nil || !slot.fence_written)
    return;
  // Before the vertex stage: a vertex shader may pull from a heap buffer, so
  // waiting only before fragment would be too late.
  [encoder waitForFence:state.pass_fence beforeStages:MTLRenderStageVertex];
}

void encoder_wait_fence(MetalDeviceState &state, MetalContextSlot &slot,
                        id<MTLComputeCommandEncoder> encoder) noexcept {
  if (encoder == nil || state.pass_fence == nil || !slot.fence_written)
    return;
  [encoder waitForFence:state.pass_fence];
}

void encoder_wait_fence(MetalDeviceState &state, MetalContextSlot &slot,
                        id<MTLBlitCommandEncoder> encoder) noexcept {
  if (encoder == nil || state.pass_fence == nil || !slot.fence_written)
    return;
  [encoder waitForFence:state.pass_fence];
}

void encoder_update_fence(MetalDeviceState &state, MetalContextSlot &slot,
                          id<MTLRenderCommandEncoder> encoder) noexcept {
  if (encoder == nil || state.pass_fence == nil)
    return;
  [encoder updateFence:state.pass_fence afterStages:MTLRenderStageFragment];
  slot.fence_written = true;
}

void encoder_update_fence(MetalDeviceState &state, MetalContextSlot &slot,
                          id<MTLComputeCommandEncoder> encoder) noexcept {
  if (encoder == nil || state.pass_fence == nil)
    return;
  [encoder updateFence:state.pass_fence];
  slot.fence_written = true;
}

void encoder_update_fence(MetalDeviceState &state, MetalContextSlot &slot,
                          id<MTLBlitCommandEncoder> encoder) noexcept {
  if (encoder == nil || state.pass_fence == nil)
    return;
  [encoder updateFence:state.pass_fence];
  slot.fence_written = true;
}

void track_resident(MetalDeviceState &state,
                    id<MTLResource> resource) noexcept {
  if (resource == nil)
    return;
  for (id<MTLResource> known : state.bindless.resident)
    if (known == resource)
      return;
  state.bindless.resident.push_back(resource);
}

void forget_resident(MetalDeviceState &state,
                     id<MTLResource> resource) noexcept {
  if (resource == nil)
    return;
  auto &resident = state.bindless.resident;
  for (usize i = 0; i < resident.size(); ++i) {
    if (resident[i] == resource) {
      resident[i] = resident.back();
      resident.pop_back();
      return;
    }
  }
}

void bind_bindless(MetalDeviceState &state,
                   id<MTLRenderCommandEncoder> encoder) noexcept {
  if (encoder == nil || state.bindless.argument_buffer == nil)
    return;

  [encoder setVertexBuffer:state.bindless.argument_buffer
                    offset:0
                   atIndex:BINDLESS_BUFFER_INDEX];
  [encoder setFragmentBuffer:state.bindless.argument_buffer
                      offset:0
                     atIndex:BINDLESS_BUFFER_INDEX];

  // One call per heap whatever it holds - the reason for sub-allocating.
  if (!state.bindless.heaps.empty()) {
    [encoder useHeaps:state.bindless.heaps.data()
                count:state.bindless.heaps.size()
               stages:MTLRenderStageVertex | MTLRenderStageFragment];
  }

  // Whatever could not live in a heap. Short by construction: shared and
  // memoryless resources only.
  if (!state.bindless.resident.empty()) {
    [encoder useResources:state.bindless.resident.data()
                    count:state.bindless.resident.size()
                    usage:MTLResourceUsageRead | MTLResourceUsageWrite
                   stages:MTLRenderStageVertex | MTLRenderStageFragment];
  }
}

void bind_bindless_compute(MetalDeviceState &state,
                           id<MTLComputeCommandEncoder> encoder) noexcept {
  if (encoder == nil || state.bindless.argument_buffer == nil)
    return;

  [encoder setBuffer:state.bindless.argument_buffer
              offset:0
             atIndex:BINDLESS_BUFFER_INDEX];

  if (!state.bindless.heaps.empty()) {
    [encoder useHeaps:state.bindless.heaps.data()
                count:state.bindless.heaps.size()];
  }

  if (!state.bindless.resident.empty()) {
    [encoder useResources:state.bindless.resident.data()
                    count:state.bindless.resident.size()
                    usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  }
}

} // namespace nxe::rhi
