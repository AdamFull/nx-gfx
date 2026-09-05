#pragma once

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

struct UploaderStats {
  u64 total_submits = 0; ///< Lifetime queue submissions (one per batch).
  u64 completed_submits = 0;
  u64 ring_used = 0; ///< Staging-ring bytes not yet reclaimed.
  u64 ring_capacity = 0;
  usize extra_buffers = 0;
  usize inflight_submits = 0;
  u64 loads_ready = 0;
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

  [[nodiscard]] UploadTicket
  upload_texture(TextureHandle handle, std::span<const u8> src,
                 std::span<const ImageSubresource> layout,
                 ResourceState final_state = ResourceState::ShaderReadOnly);

  [[nodiscard]] UploadTicket
  upload_texture(TextureHandle handle, const ImageData &image,
                 ResourceState final_state = ResourceState::ShaderReadOnly);

  [[nodiscard]] UploadTicket
  upload_buffer(BufferHandle handle, u64 dst_offset, std::span<const u8> src,
                ResourceState final_state = ResourceState::ShaderReadOnly);

  [[nodiscard]] UploadTicket
  clear_buffer(BufferHandle handle,
               ResourceState final_state = ResourceState::ShaderReadOnly);

  static constexpr TextureUsageFlags DEFAULT_LOAD_USAGE =
      TextureUsage::Sampled | TextureUsage::CopyDst;

  LoadRequest load_texture(nx::string_view path,
                           TextureUsageFlags usage = DEFAULT_LOAD_USAGE);
  /// Queues a load from encoded bytes (DDS/KTX/PNG/...); the bytes are copied.
  LoadRequest
  load_texture_from_memory(std::span<const u8> bytes,
                           TextureUsageFlags usage = DEFAULT_LOAD_USAGE);

  [[nodiscard]] LoadStatus load_status(LoadRequest req) const noexcept;
  /// The texture for a Ready request; a null handle otherwise.
  [[nodiscard]] TextureHandle loaded_texture(LoadRequest req) const noexcept;
  void release(LoadRequest req) noexcept;

  // The source must not be in concurrent use on the graphics queue - read after
  // wait_idle(), or from a resource only the copy queue touches.

  ReadbackResult read_buffer(BufferHandle handle, u64 offset, u64 size);
  ReadbackResult read_texture(TextureHandle handle, u32 mip = 0, u32 layer = 0);

  [[nodiscard]] bool is_complete(UploadTicket ticket) const noexcept;
  void wait(UploadTicket ticket) noexcept;
  void flush() noexcept;

  [[nodiscard]] UploaderStats stats() const noexcept;

  void pump();
  void flush_batch() noexcept;
  /// Records queue-ownership acquire barriers for the copies finished since the
  /// last call into @p graphics_cmd, and returns the copy-timeline value the
  /// graphics submit must wait on (0 if nothing is pending).
  u64 record_acquires(VkCommandBuffer graphics_cmd);
  [[nodiscard]] VkSemaphore timeline() const noexcept { return m_timeline; }
  void reclaim() noexcept;

private:
  struct Chunk {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    u8 *mapped = nullptr;
    u64 capacity = 0;
    u64 value = 0;
    u64 gfx_value = 0;
    bool host_read = false;
  };

  /// Uses the ring for ordinary transient copies and reusable dedicated
  /// staging for large copies or when the ring is temporarily unavailable.
  u8 *acquire_upload_staging(u64 size, u64 alignment, VkBuffer &buffer,
                             u64 &offset, Chunk *&dedicated);
  [[nodiscard]] bool flush_upload_staging(const Chunk *dedicated, u64 offset,
                                          u64 size) const noexcept;

  /// Bytes a graphics replay reads are held by `gfx_value` on the device
  /// timeline too, since the copy timeline says nothing about when the frame
  /// has run.
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
  /// Upload recording is fed by the update thread while frame hand-off and
  /// reclaim run on the render thread. Keep command-buffer and staging state
  /// as one critical section so a frame cannot submit a batch being recorded.
  mutable nx::mutex m_command_mutex;
  VkDevice m_vk = VK_NULL_HANDLE;
  VmaAllocator m_allocator = VK_NULL_HANDLE;
  VkQueue m_queue = VK_NULL_HANDLE;
  u32 m_copy_family = 0;
  u32 m_graphics_family = 0;

  VkCommandPool m_pool = VK_NULL_HANDLE;
  VkSemaphore m_timeline = VK_NULL_HANDLE;
  u64 m_value = 0;

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
  nx::small_vector<LoadRequest, 64> m_load_uploading;
  std::atomic<u64> m_loads_ready{0};
  std::atomic<u64> m_loads_failed{0};
};

} // namespace nxe::rhi
