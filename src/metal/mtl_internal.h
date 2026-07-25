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

#include "foundation/containers/concurrent_queue.h"
#include "foundation/containers/slot_map.h"
#include "foundation/containers/small_vector.h"
#include "foundation/diagnostics/log.h"
#include "foundation/threading/sync.h"
#include "foundation/threading/thread.h"
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

/// nx::string_view is not NUL-terminated; Metal's labels want an NSString.
[[nodiscard]] NSString *to_ns_string(nx::string_view text) noexcept;

struct MetalTexture {
  id<MTLTexture> texture = nil;
  /// One view per mip for the storage-image table; empty unless Storage.
  /// nx::vector rather than small_vector: ARC refuses to work through the
  /// latter's inline std::byte storage, which it cannot cast to a __strong id*.
  /// Every container holding an Objective-C object here is nx::vector for that
  /// reason.
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
  /// Slot in the bindless argument buffer's texture table; INVALID unless the
  /// texture is Sampled.
  u32 bindless_slot = BINDLESS_INVALID_INDEX;
  /// Sub-allocated from a bindless heap, so useHeaps: already covers it and it
  /// is not in the per-resource residency list.
  bool heap_backed = false;
  /// False for the backbuffer, whose MTLTexture belongs to the drawable and is
  /// swapped out from under the handle every frame.
  bool owns_texture = true;
};

struct MetalBuffer {
  id<MTLBuffer> buffer = nil;
  u64 size = 0;
  BufferUsageFlags usage = BufferUsage::None;
  void *mapped = nullptr;
  /// See MetalTexture::heap_backed.
  bool heap_backed = false;
  /// Reachable by GPU address from a shader, so it needs residency like a
  /// bindless texture does.
  bool bindless_visible = false;
};

struct MetalSampler {
  id<MTLSamplerState> sampler = nil;
  /// Slot in the bindless argument buffer's sampler table.
  u32 bindless_slot = BINDLESS_INVALID_INDEX;
};

struct MetalShader {
  id<MTLLibrary> library = nil;
};

/// Metal splits what Vulkan calls a pipeline across three places: the render
/// pipeline state, a separate depth/stencil state object, and encoder calls
/// for cull mode and winding. All of it is captured here so bind_pipeline can
/// apply the lot in one go.
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

/// The single bindless argument buffer: texture[], rwTexture[] and sampler[]
/// back to back, each entry one 8-byte MTLResourceID. Texture and sampler
/// slots are handle indices, so they need no allocation; rwTexture slots come
/// from a free list because one texture owns a slot per mip.
struct BindlessHeap {
  id<MTLBuffer> argument_buffer = nil;
  /// Byte offsets of each table within the buffer.
  u64 textures_offset = 0;
  u64 rw_textures_offset = 0;
  u64 samplers_offset = 0;
  u32 max_textures = 0;
  u32 max_rw_textures = 0;
  u32 max_samplers = 0;
  nx::small_vector<u32, 64> free_rw_slots;
  u32 rw_high_water = 0;

  // Residency. An argument buffer is just memory: naming a resource in it does
  // not make the resource reachable, so every encoder must be told what it may
  // touch. Doing that per resource is O(live resources) per pass, so anything
  // that can be is sub-allocated from a heap instead - one useHeaps: call
  // covers everything inside, however many resources that is.

  /// Heaps bindless-visible resources are sub-allocated from.
  nx::vector<id<MTLHeap>> heaps;
  /// Resources that could not come from a heap (shared or memoryless storage,
  /// or an allocation the heaps refused) and must be named individually.
  nx::vector<id<MTLResource>> resident;
  u64 heap_bytes = 0;
};

/**
 * @brief Staging and streaming state, owned by the device.
 *
 * Simpler than the Vulkan uploader in one important way: uploads are recorded
 * into a command buffer on the *same* MTLCommandQueue as the frame and
 * committed before it. A Metal queue runs its command buffers in submission
 * order, so a texture uploaded before a frame is guaranteed ready for that
 * frame with no semaphores, no queue-ownership transfer and no acquire
 * barriers - the three things that make vk_upload.cpp what it is.
 */
struct MetalUploadState {
  /// Staging buffers are pooled by size rather than sub-allocated from a ring:
  /// with no ring there is no wraparound to reason about, and a returned
  /// buffer is reused verbatim by the next upload that fits in it.
  struct Staging {
    id<MTLBuffer> buffer = nil;
    u64 capacity = 0;
    /// Readback staging is allocated with the default CPU cache mode so the
    /// CPU can actually read it; upload staging is write-combined, where a
    /// read costs an uncached fetch per byte. The pools are therefore kept
    /// apart - an upload must never be handed a readback buffer or vice versa.
    bool host_read = false;
  };

  /// One committed batch and the staging it borrowed, held until the GPU is
  /// done so the buffers can go back to the pool.
  struct InFlight {
    u64 value = 0;
    /// Kept so wait() can block on the GPU rather than spinning.
    id<MTLCommandBuffer> cmd = nil;
    nx::vector<Staging> staging;
  };

  id<MTLCommandBuffer> batch = nil;
  id<MTLBlitCommandEncoder> blit = nil;
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

  // Async loading. One worker does the slow, thread-safe half - reading bytes
  // and decoding them - and never touches the device; pump() on the render
  // thread turns finished decodes into textures.

  enum class Source : u8 { Path, Memory };

  struct LoadJob {
    u32 id = 0;
    u32 generation = 0;
    Source source = Source::Path;
    TextureUsageFlags usage = TextureUsage::Sampled | TextureUsage::CopyDst;
  };

  /// Slot in the request table. `status` is the only field both threads touch
  /// while a job is live, and it is the handshake: the worker publishes
  /// Decoding -> Ready-to-create, the render thread takes it from there.
  struct LoadSlot {
    std::atomic<LoadStatus> status{LoadStatus::None};
    u32 generation = 0;
    nx::string path;
    nx::blob<u8> encoded;
    ImageData decoded;
    TextureHandle texture;
    TextureUsageFlags usage = TextureUsage::Sampled | TextureUsage::CopyDst;
    bool in_use = false;
  };

  static constexpr usize MAX_LOADS = 256;

  nx::thread worker;
  nx::mpmc::Queue<LoadJob, 256> job_queue;
  /// Jobs the worker has finished decoding, waiting for pump() to make them
  /// into textures.
  nx::mpmc::Queue<LoadJob, 256> done_queue;
  nx::event job_signal{false, false};
  std::atomic<bool> worker_running{false};
  /// Guards allocation of slots in `loads` only; the payloads are handed
  /// between threads by the status handshake, not by this lock.
  nx::mutex load_mutex;
  LoadSlot loads[MAX_LOADS];
  u32 next_load_id = 1;
  u64 loads_ready = 0;
  u64 loads_failed = 0;
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
  /// Open only between begin_compute_pass and end_compute_pass; Metal permits
  /// exactly one encoder at a time, so this and `encoder` are never both set.
  id<MTLComputeCommandEncoder> compute_encoder = nil;
  bool frame_active = false;

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

  /// Bound at buffer index 1 on every encoder, matching where Slang puts the
  /// ParameterBlock. Index 0 is the push block.
  BindlessHeap bindless;
  id<MTLBinaryArchive> pipeline_archive = nil;
  nx::string pipeline_archive_path;

  nx::slot_map<MetalTexture, TextureHandle> textures;
  nx::slot_map<MetalBuffer, BufferHandle> buffers;
  nx::slot_map<MetalSampler, SamplerHandle> samplers;
  nx::slot_map<MetalShader, ShaderHandle> shaders;
  nx::slot_map<MetalPipeline, PipelineHandle> pipelines;

  /// A resource freed during frame N stays alive - handle, bindless slot and
  /// all - until the GPU has passed N. Deferring the slot_map erase, not just
  /// the Objective-C release, is what stops a recycled handle index from
  /// stomping a bindless slot the GPU is still reading through.
  struct PendingDelete {
    u64 frame = 0;
    TextureHandle texture;
    BufferHandle buffer;
    SamplerHandle sampler;
    PipelineHandle pipeline;
    ShaderHandle shader;
  };
  nx::vector<PendingDelete> pending_deletes;

  MetalUploadState upload;
  /// The device owns the uploader's lifetime but not its code; these live in
  /// mtl_upload.mm.
  MetalDevice *owner = nullptr;
};

/// Buffer index the push block occupies, matching Slang's [[vk::push_constant]]
/// lowering. Metal takes it through setVertexBytes / setFragmentBytes.
inline constexpr u32 PUSH_CONSTANT_BUFFER_INDEX = 0;
/// Buffer index of the bindless argument buffer (the ParameterBlock).
inline constexpr u32 BINDLESS_BUFFER_INDEX = 1;

/// Creates the argument buffer and its slot bookkeeping (mtl_bindless.mm).
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

/// Allocates a bindless-visible resource from a heap, growing the heap set if
/// none has room. Returns nil when the resource cannot live in a heap at all,
/// in which case the caller allocates from the device and tracks it
/// individually.
[[nodiscard]] id<MTLTexture>
alloc_bindless_texture(MetalDeviceState &state,
                       MTLTextureDescriptor *descriptor) noexcept;
[[nodiscard]] id<MTLBuffer>
alloc_bindless_buffer(MetalDeviceState &state, u64 length,
                      MTLResourceOptions options) noexcept;
/// Binds the argument buffer and declares every bindless resource resident.
void bind_bindless(MetalDeviceState &state,
                   id<MTLRenderCommandEncoder> encoder) noexcept;
void bind_bindless_compute(MetalDeviceState &state,
                           id<MTLComputeCommandEncoder> encoder) noexcept;

// Pipeline binary archive (mtl_pipeline.mm), Metal's answer to
// VkPipelineCache. Both are best-effort: a miss just means a compile.

/// Loads the archive at @p path, or starts an empty one. Never fails the
/// device: an unreadable or stale archive is only a cold cache.
void create_pipeline_archive(MetalDeviceState &state, nx::string_view path);
void save_pipeline_archive(MetalDeviceState &state) noexcept;

/// Queues a resource for destruction once the GPU has passed the frame being
/// recorded (mtl_device.mm). Exactly one handle field should be set.
void retire(MetalDeviceState &state,
            const MetalDeviceState::PendingDelete &pending);
/// Really destroys everything retire() queued for a frame the GPU has passed.
/// Pass ~0ull to force the lot, which is what shutdown wants.
void collect_garbage(MetalDeviceState &state, u64 completed_frame);

// Uploader internals (mtl_upload.mm), driven by the device's frame loop.

[[nodiscard]] bool uploader_init(MetalDeviceState &state);
void uploader_shutdown(MetalDeviceState &state) noexcept;
/// Turns finished decodes into GPU textures, then commits the open batch so it
/// runs ahead of this frame's rendering.
void uploader_pump(MetalDeviceState &state);
/// Returns staging borrowed by batches the GPU has finished with.
void uploader_reclaim(MetalDeviceState &state) noexcept;

} // namespace nxe::rhi
