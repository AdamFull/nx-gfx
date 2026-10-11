#pragma once

#if !defined(__OBJC__)
#error "mtl_internal.h is Objective-C++ only"
#endif

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include "core/foundation/containers/slot_map.h"
#include "core/foundation/containers/small_vector.h"
#include "core/foundation/containers/sync_slot_map.h"
#include "core/foundation/diagnostics/log.h"
#include "core/foundation/threading/sync.h"
#include "core/foundation/threading/thread.h"
#include "rendering/rhi/metal/mtl_device.h"

#include <atomic>

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

[[nodiscard]] NSString *to_ns_string(nx::string_view text) noexcept;

struct MetalTexture {
  id<MTLTexture> texture = nil;
  /// One view per mip for the storage-image table; empty unless Storage. nx::vector, not
  /// small_vector: ARC cannot cast the latter's inline std::byte storage to a __strong id*. Same
  /// for every container of `id` here.
  nx::vector<id<MTLTexture>> uavs;
  nx::small_vector<u32, 12> uav_slots;
  Format format = Format::Unknown;
  TextureDimension dimension = TextureDimension::Tex2D;
  u32 width = 0;
  u32 height = 0;
  u32 mip_levels = 1;
  u32 layers = 1;
  u32 sample_count = 1;
  TextureUsageFlags usage = TextureUsage::None;
  ResourceState state = ResourceState::Undefined;
  u32 bindless_slot = BINDLESS_INVALID_INDEX;
  bool heap_backed = false;
  bool owns_texture = true;
};

struct MetalBuffer {
  id<MTLBuffer> buffer = nil;
  u64 size = 0;
  BufferUsageFlags usage = BufferUsage::None;
  void *mapped = nullptr;
  bool heap_backed = false;
  bool bindless_visible = false;
};

struct MetalSampler {
  id<MTLSamplerState> sampler = nil;
  u32 bindless_slot = BINDLESS_INVALID_INDEX;
};

struct MetalHeap {
  id<MTLHeap> heap = nil;
  u64 size = 0;
};

struct MetalShader {
  id<MTLLibrary> library = nil;
  ShaderReflection reflection;
};

struct MetalPipeline {
  id<MTLRenderPipelineState> render = nil;
  id<MTLComputePipelineState> compute = nil;
  id<MTLDepthStencilState> depth_stencil = nil;
  MTLPrimitiveType topology = MTLPrimitiveTypeTriangle;
  MTLCullMode cull = MTLCullModeNone;
  MTLWinding winding = MTLWindingCounterClockwise;
  MTLSize threads_per_group = {1, 1, 1};
  bool is_compute = false;
};

/// The bindless argument buffer: texture[], rwTexture[] and sampler[] back to
/// back, 8 bytes per entry. Texture and sampler slots are handle indices;
/// rwTexture slots come from a free list, since a texture owns one per mip.
struct BindlessHeap {
  id<MTLBuffer> argument_buffer = nil;
  /// Byte offsets of each table within the buffer, in the declaration order of
  /// BindlessTables in shaders/core.slang.
  u64 textures_offset = 0;
  u64 uint_textures_offset = 0;
  u64 rw_textures_offset = 0;
  u64 rw_uint_textures_offset = 0;
  u64 samplers_offset = 0;
  u32 max_textures = 0;
  u32 max_rw_textures = 0;
  u32 max_samplers = 0;
  nx::small_vector<u32, 64> free_rw_slots;
  u32 rw_high_water = 0;

  // Residency: naming a resource in an argument buffer does not make it
  // reachable, so every encoder must be told. Per-resource is O(live) per pass,
  // hence heaps - one useHeaps: covers everything inside.

  nx::vector<id<MTLHeap>> heaps;
  /// Resources that could not come from a heap (shared or memoryless storage,
  /// or an allocation the heaps refused) and must be named individually.
  nx::vector<id<MTLResource>> resident;
  u64 heap_bytes = 0;
};

/// Everything one recording thread owns while a frame is open. Sized for one
/// context today (see MetalDeviceState::MAX_CONTEXTS) - not per-device state,
/// so encoder and bound state stay clear of whichever thread records.
struct MetalContextSlot {
  id<MTLCommandBuffer> cmd = nil;
  id<MTLRenderCommandEncoder> encoder = nil;
  id<MTLComputeCommandEncoder> compute_encoder = nil;

  /// Metal takes the index buffer on the draw call rather than from a bind, so
  /// bind_index_buffer only records what the next draw_indexed should use.
  id<MTLBuffer> index_buffer = nil;
  MTLIndexType index_type = MTLIndexTypeUInt16;
  u64 index_buffer_offset = 0;
  /// Size of one index in bytes, for turning first_index into a byte offset.
  u32 index_stride = 2;

  /// Shadow copy of the push block. Metal's setBytes replaces the whole
  /// binding, so a partial push_constants() call has to re-send the rest.
  alignas(16) u8 push_constants[PUSH_CONSTANT_SIZE] = {};
  MTLPrimitiveType bound_topology = MTLPrimitiveTypeTriangle;
  MTLSize bound_threads_per_group = {1, 1, 1};
  bool has_bound_pipeline = false;

  /// A fence may only be waited on after it has been updated in the *same*
  /// command buffer, so this resets with each new one.
  bool fence_written = false;
  /// Set while the slot holds a command buffer that end_frame must commit.
  bool in_use = false;
};

/// Much smaller than the Vulkan uploader because uploads share the frame's queue and are committed
/// before it: in-order execution replaces the semaphores, queue-ownership transfers and acquire
/// barriers vk_upload.cpp needs.
struct MetalUploadState {
  /// Pooled by size rather than sub-allocated from a ring: no wraparound to
  /// reason about, and a returned buffer is reused verbatim.
  struct Staging {
    id<MTLBuffer> buffer = nil;
    u64 capacity = 0;
    /// Readback uses the default cache mode; upload staging is write-combined,
    /// where a CPU read costs an uncached fetch per byte. Hence separate pools.
    bool host_read = false;
  };

  /// One committed batch and the staging it borrowed, held until the GPU is
  /// done so the buffers can go back to the pool.
  struct InFlight {
    u64 value = 0;
    id<MTLCommandBuffer> cmd = nil;
    nx::vector<Staging> staging;
  };

  id<MTLCommandBuffer> batch = nil;
  id<MTLBlitCommandEncoder> blit = nil;
  /// The batch is its own command buffer, so it needs its own fence
  /// bookkeeping - a fence may only be waited on in the buffer that updated it.
  MetalContextSlot batch_slot;
  /// Staging borrowed by the batch that is still open.
  nx::vector<Staging> batch_staging;
  nx::vector<Staging> free_staging;
  nx::vector<InFlight> in_flight;

  /// Ticket timeline. `submitted` is bumped when a batch is committed;
  /// `completed` is bumped from the command buffer's completion handler, which
  /// runs on an arbitrary thread - hence the atomic.
  u64 submitted = 0;
  std::atomic<u64> completed{0};
  u64 bytes_staged = 0;
};

/// One window's layer, or a texture of the device's own that nothing presents.
struct MetalSwapchain {
  // The layer is owned by the window's view once attached; we keep a strong
  // reference so a resize can still reach it.
  CAMetalLayer *layer = nil;
  /// The NSView the layer was attached to, so destroy_swapchain can detach it
  /// again. Typed as `id` rather than NSView* to keep AppKit out of this
  /// header; nil on iOS, where the layer is handed to us already attached.
  id layer_host = nil;
  id<CAMetalDrawable> drawable = nil;
  Extent2D extent = {};
  PresentMode present_mode = PresentMode::Fifo;
  SurfaceAlpha alpha = SurfaceAlpha::Opaque;
  /// One handle for the swapchain's lifetime; the MTLTexture behind it is the
  /// frame's drawable, or the device's own texture offscreen.
  TextureHandle backbuffer;
  /// The backbuffer is a texture this device made rather than a layer's
  /// drawable, so there is nothing to acquire and nothing to present.
  bool offscreen = false;
  /// Acquired by the frame being recorded.
  bool acquired = false;
};

struct MetalDeviceState {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id gpu_crash_observer = nil;
  bool gpu_crash_detailed = false;

  nx::slot_map<MetalSwapchain, SwapchainHandle> swapchains;
  /// The swapchains the frame being recorded acquired, in acquire order.
  nx::small_vector<SwapchainHandle, 4> acquired;
  /// Every swapchain's; see MetalDevice::swapchain_format.
  Format swapchain_format = Format::Unknown;

  dispatch_semaphore_t frame_semaphore = nil;
  u32 frames_in_flight = 2;
  u64 frames_submitted = 0;

  bool frame_active = false;
  bool headless_frame = false;

  static constexpr usize MAX_CONTEXTS = 1;
  nx::vector<MetalContextSlot> contexts;
  usize used_contexts = 0;

  BindlessHeap bindless;
  id<MTLFence> pass_fence = nil;
  id<MTLEvent> upload_event = nil;

  id<MTLBinaryArchive> pipeline_archive = nil;
  nx::string pipeline_archive_path;
  /// Pipeline compilation and archive mutation may run on worker threads.
  nx::mutex pipeline_archive_lock;

  nx::sync_slot_map<MetalTexture, TextureHandle> textures;
  nx::sync_slot_map<MetalBuffer, BufferHandle> buffers;
  nx::slot_map<MetalSampler, SamplerHandle> samplers;
  nx::slot_map<MetalHeap, HeapHandle> heaps;
  nx::slot_map<MetalShader, ShaderHandle> shaders;
  nx::sync_slot_map<MetalPipeline, PipelineHandle> pipelines;

  /// A resource freed during frame N stays alive - handle and bindless slot
  /// included - until the GPU passes N, so a recycled index cannot stomp a slot
  /// still being read.
  struct PendingDelete {
    u64 frame = 0;
    TextureHandle texture;
    BufferHandle buffer;
    SamplerHandle sampler;
    PipelineHandle pipeline;
    ShaderHandle shader;
    HeapHandle heap;
    /// The held frame work it waits on first; 0 once only frames are left.
    u64 hold = 0;
  };
  /// The producer holds and the render job releases, so the list and the hold
  /// are shared between them.
  nx::mutex retire_lock;
  nx::vector<PendingDelete> pending_deletes;
  u64 retire_hold = 0;
  u64 retire_released = 0;
  struct PendingReplace {
    TextureHandle target;
    TextureHandle replacement;
    /// The last held frame work that may name the target's old image.
    u64 after = 0;
  };
  nx::vector<PendingReplace> pending_replaces;

  MetalUploadState upload;
  /// The device owns the uploader's lifetime but not its code; these live in
  /// mtl_upload.mm.
  MetalDevice *owner = nullptr;
};

inline constexpr u32 PUSH_CONSTANT_BUFFER_INDEX = 0;
inline constexpr u32 BINDLESS_BUFFER_INDEX = 1;

[[nodiscard]] bool create_bindless_heap(MetalDeviceState &state,
                                        const DeviceCaps &caps);
void write_texture_descriptor(MetalDeviceState &state, u32 slot,
                              id<MTLTexture> texture) noexcept;
void write_rw_texture_descriptor(MetalDeviceState &state, u32 slot,
                                 id<MTLTexture> texture) noexcept;
void write_sampler_descriptor(MetalDeviceState &state, u32 slot,
                              id<MTLSamplerState> sampler) noexcept;
[[nodiscard]] u32 alloc_rw_slot(MetalDeviceState &state) noexcept;
void free_rw_slot(MetalDeviceState &state, u32 slot) noexcept;
void track_resident(MetalDeviceState &state, id<MTLResource> resource) noexcept;
void forget_resident(MetalDeviceState &state, id<MTLResource> resource) noexcept;

/// Allocates a bindless-visible resource from a heap, growing the heap set if none has room.
/// Returns nil when the resource cannot live in a heap at all, in which case the caller allocates
/// from the device and tracks it individually.
[[nodiscard]] id<MTLTexture>
alloc_bindless_texture(MetalDeviceState &state,
                       MTLTextureDescriptor *descriptor) noexcept;
[[nodiscard]] id<MTLBuffer>
alloc_bindless_buffer(MetalDeviceState &state, u64 length,
                      MTLResourceOptions options) noexcept;
void bind_bindless(MetalDeviceState &state,
                   id<MTLRenderCommandEncoder> encoder) noexcept;
void bind_bindless_compute(MetalDeviceState &state,
                           id<MTLComputeCommandEncoder> encoder) noexcept;

void create_pipeline_archive(MetalDeviceState &state, nx::string_view path);
void save_pipeline_archive(MetalDeviceState &state) noexcept;

// Pass-granularity ordering within one command buffer (mtl_bindless.mm). A fence cannot reach
// across command buffers. Cross-encoder hazards are left to Metal's own tracking, which is why the
// heaps are created tracked.

void encoder_wait_fence(MetalDeviceState &state, MetalContextSlot &slot,
                        id<MTLRenderCommandEncoder> encoder) noexcept;
void encoder_wait_fence(MetalDeviceState &state, MetalContextSlot &slot,
                        id<MTLComputeCommandEncoder> encoder) noexcept;
void encoder_wait_fence(MetalDeviceState &state, MetalContextSlot &slot,
                        id<MTLBlitCommandEncoder> encoder) noexcept;

void encoder_update_fence(MetalDeviceState &state, MetalContextSlot &slot,
                          id<MTLRenderCommandEncoder> encoder) noexcept;
void encoder_update_fence(MetalDeviceState &state, MetalContextSlot &slot,
                          id<MTLComputeCommandEncoder> encoder) noexcept;
void encoder_update_fence(MetalDeviceState &state, MetalContextSlot &slot,
                          id<MTLBlitCommandEncoder> encoder) noexcept;

void begin_command_buffer(MetalContextSlot &slot) noexcept;

void retire(MetalDeviceState &state,
            const MetalDeviceState::PendingDelete &pending);
void collect_garbage(MetalDeviceState &state, u64 completed_frame);

void uploader_shutdown(MetalDeviceState &state) noexcept;
/// Returns staging borrowed by batches the GPU has finished with.
void uploader_reclaim(MetalDeviceState &state) noexcept;

}
