#pragma once

/**
 * @file vk_upload.h
 * @brief Asynchronous GPU upload, readback, and file/decode streaming.
 *
 * All GPU work happens on the render thread, so the machinery needs no lock.
 * Uploads stage through a bounded ring and are *batched*: each call records
 * its copy and returns an UploadTicket, but nothing is submitted until the
 * batch flushes - once per frame, on staging backpressure, or explicitly.
 *
 * File loading runs on one worker that does only the slow, thread-safe part
 * and never touches the device. It talks to the render thread through
 * lock-free queues and an atomic status table, so there is no mutex anywhere.
 *
 * With a distinct transfer queue, images move between the two families once
 * per frame rather than once per copy: uploads leave the image in TRANSFER_DST
 * and copy-family-owned, and the frame hand-off emits the release/acquire
 * pair. Batching the hand-off is what makes repeated partial uploads to one
 * texture - a font atlas growing glyphs - legal.
 *
 * Once handed over, the copy queue cannot take an image back: an acquire needs
 * a release, and only a graphics submit can record one. A later upload to such
 * an image is replayed into the frame's own command buffer by record_acquires.
 */

#include "core/foundation/containers/concurrent_queue.h"
#include "core/foundation/containers/small_vector.h"
#include "core/foundation/core/foundation.h"
#include "core/foundation/strings/utf8_string.h"
#include "core/foundation/threading/sync.h"
#include "core/foundation/threading/thread.h"
#include "core/rendering/rhi/handles.h"
#include "core/rendering/rhi/image/image.h"
#include "core/rendering/rhi/types.h"
#include "core/rendering/rhi/upload.h"
#include "core/rendering/rhi/vulkan/vk_common.h"

#include <vk_mem_alloc.h>

#include <atomic>
#include <span>

namespace nxe::rhi {

class VulkanDevice;

/// Handle to an in-flight readback. `data` is valid once is_complete(ticket)
/// (or after wait(ticket)) and stays valid until a later read_* call reuses the
/// same staging - copy it out promptly.
struct ReadbackResult {
  UploadTicket ticket;
  const u8 *data = nullptr;
  u64 size = 0;

  [[nodiscard]] std::span<const u8> bytes() const noexcept {
    return {data, nx::cast<usize>(size)};
  }
};

/// A snapshot of the uploader's internal load, for profiling and stress tests.
struct UploaderStats {
  u64 total_submits = 0;     ///< Lifetime queue submissions (one per batch).
  u64 completed_submits = 0; ///< Of those, finished on the GPU.
  u64 ring_used = 0;         ///< Staging-ring bytes not yet reclaimed.
  u64 ring_capacity = 0;
  usize extra_buffers = 0; ///< Dedicated oversized-upload / readback buffers.
  usize inflight_submits = 0; ///< Batches whose copy is not done yet.
  u64 loads_ready = 0;        ///< Async loads that reached Ready.
  u64 loads_failed = 0;
};

class VulkanUploader {
public:
  VulkanUploader() = default;
  ~VulkanUploader();

  VulkanUploader(const VulkanUploader &) = delete;
  VulkanUploader &operator=(const VulkanUploader &) = delete;

  [[nodiscard]] bool init(VulkanDevice *device);
  void shutdown() noexcept;

  // Direct uploads (render thread)

  /// Uploads @p layout's subresources (offsets into @p src) into @p handle and
  /// leaves it in @p final_state, ready for the graphics queue. The copy joins
  /// the current batch; the returned ticket completes when that batch is
  /// flushed and its GPU work finishes.
  ///
  /// On a dedicated transfer queue @p final_state is reached at the next frame
  /// hand-off, and a texture graphics already owns is uploaded by that frame
  /// instead - so the ticket comes back complete but not valid. Wait on such
  /// an upload by presenting a frame; ok() tells it from one that failed.
  [[nodiscard]] UploadTicket
  upload_texture(TextureHandle handle, std::span<const u8> src,
                 std::span<const ImageSubresource> layout,
                 ResourceState final_state = ResourceState::ShaderReadOnly);

  /// Convenience: upload a whole decoded image.
  [[nodiscard]] UploadTicket
  upload_texture(TextureHandle handle, const ImageData &image,
                 ResourceState final_state = ResourceState::ShaderReadOnly);

  [[nodiscard]] UploadTicket
  upload_buffer(BufferHandle handle, u64 dst_offset, std::span<const u8> src,
                ResourceState final_state = ResourceState::ShaderReadOnly);

  // Async loading (enqueue from any thread; poll from any thread)

  static constexpr TextureUsageFlags DEFAULT_LOAD_USAGE =
      TextureUsage::Sampled | TextureUsage::CopyDst;

  /// Queues a texture load from a file path; returns immediately. The worker
  /// thread reads and decodes it; pump() creates and uploads it.
  LoadRequest load_texture(nx::string_view path,
                           TextureUsageFlags usage = DEFAULT_LOAD_USAGE);
  /// Queues a load from encoded bytes (DDS/KTX/PNG/...); the bytes are copied.
  LoadRequest
  load_texture_from_memory(std::span<const u8> bytes,
                           TextureUsageFlags usage = DEFAULT_LOAD_USAGE);

  [[nodiscard]] LoadStatus load_status(LoadRequest req) const noexcept;
  /// The texture for a Ready request; a null handle otherwise.
  [[nodiscard]] TextureHandle loaded_texture(LoadRequest req) const noexcept;
  /// Recycles a finished request's slot. The TextureHandle is the caller's to
  /// keep or destroy; only call once the request is Ready or Failed.
  void release(LoadRequest req) noexcept;

  // Readbacks (render thread). The source must not be in concurrent use on the
  // graphics queue - read after wait_idle(), or from a resource only the copy
  // queue touches. Readbacks flush any open batch and submit on their own.
  //
  // With a dedicated transfer queue the copy family must also still *own* the
  // image, which it does until the next frame hands it over: read back before
  // presenting. read_texture warns and reads anyway once graphics owns it.

  /// Reads a range from a buffer created with BufferUsage::CopySrc. Invalid
  /// handles, usage, or ranges are rejected before GPU work is recorded.
  ReadbackResult read_buffer(BufferHandle handle, u64 offset, u64 size);
  ReadbackResult read_texture(TextureHandle handle, u32 mip = 0, u32 layer = 0);

  // Completion

  [[nodiscard]] bool is_complete(UploadTicket ticket) const noexcept;
  void wait(UploadTicket ticket) noexcept;
  /// Submits the open batch, then blocks until everything submitted has run.
  void flush() noexcept;

  [[nodiscard]] UploaderStats stats() const noexcept;

  // Device frame integration (called by VulkanDevice, render thread)

  /// Turns finished async decodes into GPU textures and advances completed ones
  /// to Ready. Runs before flush_batch so its uploads ride the frame's batch.
  void pump();
  /// Submits the batch accumulated since the last frame.
  void flush_batch() noexcept;
  /// Records queue-ownership acquire barriers for the copies finished since the
  /// last call into @p graphics_cmd, and returns the copy-timeline value the
  /// graphics submit must wait on (0 if nothing is pending).
  u64 record_acquires(VkCommandBuffer graphics_cmd);
  [[nodiscard]] VkSemaphore timeline() const noexcept { return m_timeline; }
  /// Advances the staging ring and returns finished command buffers / dedicated
  /// buffers to their reuse pools.
  void reclaim() noexcept;

private:
  /// A dedicated staging buffer for the two cases the ring does not serve: an
  /// upload larger than the whole ring, and readback (host-readable).
  struct Chunk {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    u8 *mapped = nullptr;
    u64 capacity = 0;
    u64 value = 0;     ///< Copy-timeline value that frees it; 0 when free.
    u64 gfx_value = 0; ///< Same, on the *device* timeline (graphics replays).
    bool host_read = false;
  };

  /// Marks how far the ring head had advanced when a batch was submitted; the
  /// tail may pass this point once `value` completes on the GPU. Bytes a
  /// graphics replay reads are held by `gfx_value` on the device timeline too,
  /// since the copy timeline says nothing about when the frame has run.
  struct Marker {
    u64 head = 0;
    u64 value = 0;
    u64 gfx_value = 0;
  };

  struct ImageAcquire {
    VkImage image = VK_NULL_HANDLE;
    VkImageAspectFlags aspect = 0;
    VkImageLayout old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ResourceState new_state = ResourceState::ShaderReadOnly;
  };
  /// An image the copy queue still owns and must hand to graphics at the next
  /// frame hand-off. One record per image however many copies touched it, so
  /// the release/acquire pair is emitted once and always names the layout the
  /// image is actually in.
  struct ImageRelease {
    TextureHandle handle;
    VkImage image = VK_NULL_HANDLE;
    VkImageAspectFlags aspect = 0;
    ResourceState old_state = ResourceState::CopyDst;
    ResourceState new_state = ResourceState::ShaderReadOnly;
  };
  /// A copy into an image the graphics family owns, waiting to be replayed into
  /// the next frame's command buffer (see m_graphics_hold for how its staging
  /// bytes are kept alive until that frame has run).
  struct GraphicsCopy {
    TextureHandle handle;
    VkImage image = VK_NULL_HANDLE;
    VkImageAspectFlags aspect = 0;
    ResourceState old_state = ResourceState::ShaderReadOnly;
    ResourceState new_state = ResourceState::ShaderReadOnly;
    VkBuffer buffer = VK_NULL_HANDLE;
    nx::small_vector<VkBufferImageCopy, 4> regions;
  };
  struct BufferAcquire {
    VkBuffer buffer = VK_NULL_HANDLE;
    u64 offset = 0;
    u64 size = 0;
    ResourceState new_state = ResourceState::ShaderReadOnly;
  };
  struct BufferRelease {
    BufferHandle handle;
    VkBuffer buffer = VK_NULL_HANDLE;
    u64 offset = 0;
    u64 size = 0;
    ResourceState new_state = ResourceState::ShaderReadOnly;
  };

  // Loader plumbing. The queue elements carry owning heap pointers so they stay
  // trivially movable (a requirement of the lock-free queues).
  struct LoadPayload {
    nx::string path;
    nx::blob<u8> bytes;
    TextureUsageFlags usage = DEFAULT_LOAD_USAGE;
  };
  struct LoadCommand {
    u32 id = 0;
    u32 generation = 0;
    LoadPayload *payload = nullptr;
  };
  struct DecodeResult {
    u32 id = 0;
    u32 generation = 0;
    ImageData *image = nullptr;
    TextureUsageFlags usage = DEFAULT_LOAD_USAGE;
  };
  static constexpr u32 MAX_LOADS = 1024;

  [[nodiscard]] u64 completed_value() const noexcept;
  /// Completed value of the *device's* graphics timeline, which is what says a
  /// frame - and so any copy replayed into it - has finished.
  [[nodiscard]] u64 graphics_completed() const noexcept;
  [[nodiscard]] bool needs_ownership_transfer() const noexcept {
    return m_copy_family != m_graphics_family;
  }

  VkCommandBuffer ensure_batch();
  void flush_locked();
  void reclaim_locked();
  u8 *ring_alloc(u64 size, u64 alignment, u64 *out_offset);
  Chunk *acquire_chunk(u64 size, bool host_read);
  VkCommandBuffer begin_cmd();
  void submit_signal(VkCommandBuffer cmd, u64 value);
  void wait_value(u64 value) noexcept;
  void wait_graphics(u64 value) noexcept;

  /// Finds or starts the pending hand-off record for @p image, so several
  /// copies into one texture share a single release/acquire pair.
  ImageRelease &pending_release(TextureHandle handle, VkImage image,
                                 VkImageAspectFlags aspect, ResourceState state);
  BufferRelease &pending_release(BufferHandle handle, VkBuffer buffer,
                                  u64 offset, u64 size, ResourceState state);
  /// Emits the copy-side half of the hand-off into the open batch and turns
  /// each record into the acquire record_acquires() will emit.
  void record_releases();
  /// Replays the copies that must run on the graphics queue into @p cmd.
  void replay_graphics_copies(VkCommandBuffer cmd);

  void worker_main();
  LoadRequest enqueue_load(LoadPayload *payload);

  VulkanDevice *m_device = nullptr;
  VkDevice m_vk = VK_NULL_HANDLE;
  VmaAllocator m_allocator = VK_NULL_HANDLE;
  VkQueue m_queue = VK_NULL_HANDLE;
  u32 m_copy_family = 0;
  u32 m_graphics_family = 0;

  VkCommandPool m_pool = VK_NULL_HANDLE;
  VkSemaphore m_timeline = VK_NULL_HANDLE;
  u64 m_value = 0; ///< Last value submitted (signalled) on the timeline.

  // Staging ring (monotonic head/tail; offset = head % capacity). Render-thread
  // only, hence lockless.
  VkBuffer m_ring = VK_NULL_HANDLE;
  VmaAllocation m_ring_alloc = VK_NULL_HANDLE;
  u8 *m_ring_mapped = nullptr;
  u64 m_ring_capacity = 0;
  u64 m_ring_head = 0;
  u64 m_ring_tail = 0;
  nx::small_vector<Marker, 32> m_markers;

  VkCommandBuffer m_batch_cmd = VK_NULL_HANDLE;
  bool m_batch_open = false;
  u64 m_batch_value = 0;

  nx::small_vector<VkCommandBuffer, 8> m_free_cmds;
  struct Inflight {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    u64 value = 0;
  };
  nx::small_vector<Inflight, 8> m_inflight;
  nx::small_vector<Chunk, 8> m_chunks;

  nx::small_vector<ImageAcquire, 32> m_image_acquires;
  nx::small_vector<ImageRelease, 32> m_image_releases;
  nx::small_vector<BufferAcquire, 16> m_buffer_acquires;
  nx::small_vector<BufferRelease, 16> m_buffer_releases;
  nx::small_vector<GraphicsCopy, 8> m_graphics_copies;
  u64 m_pending_wait = 0;
  /// Device-timeline value the frame that will replay m_graphics_copies is
  /// going to signal. Every ring marker pushed while copies are pending carries
  /// it, so the tail cannot walk over bytes that frame has yet to read.
  u64 m_graphics_hold = 0;

  // Async loader. The worker touches only the queues, the status table and the
  // heap payloads - never the device or the machinery above.
  nx::thread m_worker;
  std::atomic<bool> m_worker_stop{false};
  nx::semaphore m_worker_wake{0};
  nx::mpsc::Queue<LoadCommand, MAX_LOADS> m_load_requests;
  nx::mpsc::Queue<DecodeResult, MAX_LOADS> m_load_results;
  nx::mpmc::Queue<u32, MAX_LOADS> m_free_slots;
  std::atomic<LoadStatus> m_load_state[MAX_LOADS];
  std::atomic<u32> m_load_gen[MAX_LOADS];
  TextureHandle m_load_handle[MAX_LOADS];
  UploadTicket m_load_ticket[MAX_LOADS];
  nx::small_vector<LoadRequest, 64> m_load_uploading; ///< Render-thread only.
  std::atomic<u64> m_loads_ready{0};
  std::atomic<u64> m_loads_failed{0};
};

} // namespace nxe::rhi
