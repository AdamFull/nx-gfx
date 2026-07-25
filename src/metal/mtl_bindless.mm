/**
 * @file mtl_bindless.mm
 * @brief The single bindless argument buffer.
 *
 * Slang lowers the shader's ParameterBlock<BindlessTables> to one Metal
 * argument buffer holding three back-to-back arrays: textures, then
 * rw_textures, then samplers. On Tier 2 hardware every entry is one 8-byte
 * MTLResourceID, so the buffer is written straight from the CPU with no
 * MTLArgumentEncoder - the layout is fully determined by the array bounds
 * declared in shaders/bindless.slang.
 *
 * The two things Metal needs that Vulkan does not:
 *
 *  - Tier 2 argument buffers. Tier 1 allows neither arrays of resources nor
 *    dynamic indexing into them, which is the entire binding model. There is
 *    no fallback; init() refuses the device.
 *
 *  - Residency. An argument buffer is just memory: naming a texture in it does
 *    not make the texture reachable. Every encoder must be told, which is what
 *    the `resident` list and bind_bindless() are for.
 */

#include "rendering/rhi/metal/mtl_internal.h"

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

/// New heaps are this big unless a single resource needs more. Large enough
/// that a 2D game's whole atlas set lands in one or two, small enough that the
/// first allocation does not reserve an absurd amount of address space.
constexpr u64 DEFAULT_HEAP_BYTES = 64ull << 20;

/// Adds a heap big enough for @p required bytes and returns it, or nil.
id<MTLHeap> grow_heaps(MetalDeviceState &state, const u64 required) noexcept {
  MTLHeapDescriptor *info = [[MTLHeapDescriptor alloc] init];
  info.size = nx::max<u64>(DEFAULT_HEAP_BYTES, required);
  info.storageMode = MTLStorageModePrivate;
  info.type = MTLHeapTypeAutomatic;
  // Untracked: Metal inserts no dependency between an encoder that writes a
  // resource in this heap and one that later reads it. The backend places that
  // synchronisation itself - see pass_fence and upload_event in
  // MetalDeviceState - which is what lets the driver stop tracking every
  // bindless resource individually at each encoder boundary.
  //
  // Only heap resources are untracked. Everything else keeps Metal's automatic
  // tracking, so the hand-placed synchronisation covers exactly the bindless
  // set and nothing more.
  info.hazardTrackingMode = MTLHazardTrackingModeUntracked;

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

void begin_command_buffer(MetalDeviceState &state) noexcept {
  state.fence_written = false;
}

void encoder_wait_fence(MetalDeviceState &state,
                        id<MTLRenderCommandEncoder> encoder) noexcept {
  if (encoder == nil || state.pass_fence == nil || !state.fence_written)
    return;
  // Before the vertex stage: a vertex shader may pull from a heap buffer, so
  // waiting only before fragment would be too late.
  [encoder waitForFence:state.pass_fence beforeStages:MTLRenderStageVertex];
}

void encoder_wait_fence(MetalDeviceState &state,
                        id<MTLComputeCommandEncoder> encoder) noexcept {
  if (encoder == nil || state.pass_fence == nil || !state.fence_written)
    return;
  [encoder waitForFence:state.pass_fence];
}

void encoder_wait_fence(MetalDeviceState &state,
                        id<MTLBlitCommandEncoder> encoder) noexcept {
  if (encoder == nil || state.pass_fence == nil || !state.fence_written)
    return;
  [encoder waitForFence:state.pass_fence];
}

void encoder_update_fence(MetalDeviceState &state,
                          id<MTLRenderCommandEncoder> encoder) noexcept {
  if (encoder == nil || state.pass_fence == nil)
    return;
  [encoder updateFence:state.pass_fence afterStages:MTLRenderStageFragment];
  state.fence_written = true;
}

void encoder_update_fence(MetalDeviceState &state,
                          id<MTLComputeCommandEncoder> encoder) noexcept {
  if (encoder == nil || state.pass_fence == nil)
    return;
  [encoder updateFence:state.pass_fence];
  state.fence_written = true;
}

void encoder_update_fence(MetalDeviceState &state,
                          id<MTLBlitCommandEncoder> encoder) noexcept {
  if (encoder == nil || state.pass_fence == nil)
    return;
  [encoder updateFence:state.pass_fence];
  state.fence_written = true;
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

  // One call per heap, whatever it holds - that is the whole reason bindless
  // resources are sub-allocated rather than allocated individually.
  if (!state.bindless.heaps.empty()) {
    [encoder useHeaps:state.bindless.heaps.data()
                count:state.bindless.heaps.size()
               stages:MTLRenderStageVertex | MTLRenderStageFragment];
  }

  // Whatever could not live in a heap still has to be named. This list stays
  // short by construction: it only takes shared or memoryless resources.
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
