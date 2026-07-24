#pragma once

/**
 * @file vk_upload.h
 * @brief Asynchronous GPU upload and readback on the transfer queue.
 *
 * The uploader turns "put these bytes into this GPU resource" into copy-queue
 * work that overlaps rendering. Uploads stage through a bounded ring buffer and
 * are *batched*: each call records its copy into a shared command buffer and
 * returns an UploadTicket, but nothing is submitted until the batch is flushed
 * (once per frame by the device, on staging backpressure, or on an explicit
 * flush()/wait()). One submit drains a whole frame's worth of loads instead of
 * one submit per texture.
 *
 * When the transfer queue is a distinct hardware queue, each copy ends with a
 * queue-family release barrier; the device injects the matching acquire barrier
 * into the next graphics frame (record_acquires) and makes that frame's submit
 * wait on the batch, so a freshly uploaded resource is safe to sample from the
 * frame its acquire is recorded onward.
 *
 * Every public method is thread-safe: decode on a job, then upload from that
 * job. The batch command buffer and ring are guarded by one mutex, so recording
 * is serialized while submission stays off the caller's critical path. The RHI
 * owns no worker threads of its own.
 */

#include "foundation/containers/small_vector.h"
#include "foundation/core/foundation.h"
#include "foundation/threading/sync.h"
#include "rendering/rhi/handles.h"
#include "rendering/rhi/image/image.h"
#include "rendering/rhi/types.h"
#include "rendering/rhi/upload.h"
#include "rendering/rhi/vulkan/vk_common.h"

#include <vk_mem_alloc.h>

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
};

class VulkanUploader {
public:
  VulkanUploader() = default;
  ~VulkanUploader();

  VulkanUploader(const VulkanUploader &) = delete;
  VulkanUploader &operator=(const VulkanUploader &) = delete;

  [[nodiscard]] bool init(VulkanDevice *device);
  void shutdown() noexcept;

  // Uploads

  /// Uploads @p layout's subresources (offsets into @p src) into @p handle and
  /// leaves it in @p final_state, ready for the graphics queue. The copy joins
  /// the current batch; the returned ticket completes when that batch is
  /// flushed and its GPU work finishes.
  UploadTicket
  upload_texture(TextureHandle handle, std::span<const u8> src,
                 std::span<const ImageSubresource> layout,
                 ResourceState final_state = ResourceState::ShaderReadOnly);

  /// Convenience: upload a whole decoded image.
  UploadTicket
  upload_texture(TextureHandle handle, const ImageData &image,
                 ResourceState final_state = ResourceState::ShaderReadOnly);

  UploadTicket
  upload_buffer(BufferHandle handle, u64 dst_offset, std::span<const u8> src,
                ResourceState final_state = ResourceState::ShaderReadOnly);

  // Readbacks
  // The source must not be in concurrent use on the graphics queue - read after
  // wait_idle(), or from a resource only the copy queue touches. Readbacks
  // flush any open upload batch and submit on their own.

  ReadbackResult read_buffer(BufferHandle handle, u64 offset, u64 size);
  ReadbackResult read_texture(TextureHandle handle, u32 mip = 0, u32 layer = 0);

  // Completion

  [[nodiscard]] bool is_complete(UploadTicket ticket) const noexcept;
  void wait(UploadTicket ticket) noexcept;
  /// Submits the open batch, then blocks until everything submitted has run.
  void flush() noexcept;

  [[nodiscard]] UploaderStats stats() const noexcept;

  // Device frame integration (called by VulkanDevice)

  /// Submits the batch accumulated since the last frame. Called at the start of
  /// each graphics frame, before record_acquires.
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
  /// upload larger than the whole ring, and readback (host-readable). Freed for
  /// reuse (value -> 0) once its submit completes; the VkBuffer lives until
  /// shutdown, so a readback's mapped data survives until the buffer is reused.
  struct Chunk {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    u8 *mapped = nullptr;
    u64 capacity = 0;
    u64 value = 0;
    bool host_read = false;
  };

  /// Marks how far the ring head had advanced when a batch was submitted; the
  /// tail may pass this point once `value` completes on the GPU.
  struct Marker {
    u64 head = 0;
    u64 value = 0;
  };

  struct ImageAcquire {
    VkImage image = VK_NULL_HANDLE;
    VkImageAspectFlags aspect = 0;
    VkImageLayout old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ResourceState new_state = ResourceState::ShaderReadOnly;
  };
  struct BufferAcquire {
    VkBuffer buffer = VK_NULL_HANDLE;
    u64 offset = 0;
    u64 size = 0;
    ResourceState new_state = ResourceState::ShaderReadOnly;
  };

  [[nodiscard]] u64 completed_value() const noexcept;
  [[nodiscard]] bool needs_ownership_transfer() const noexcept {
    return m_copy_family != m_graphics_family;
  }

  /// Opens the batch command buffer if it is not already recording, reserving
  /// the timeline value the batch will signal. Returns the batch command
  /// buffer.
  VkCommandBuffer ensure_batch();
  void flush_locked();
  void reclaim_locked();

  /// Reserves @p size ring bytes aligned to @p alignment, flushing and waiting
  /// on in-flight batches for room if necessary. Returns the mapped pointer and
  /// sets @p out_offset, or nullptr if the request cannot fit the ring at all.
  u8 *ring_alloc(u64 size, u64 alignment, u64 *out_offset);
  Chunk *acquire_chunk(u64 size, bool host_read);
  VkCommandBuffer begin_cmd();
  void submit_signal(VkCommandBuffer cmd, u64 value);

  VulkanDevice *m_device = nullptr;
  VkDevice m_vk = VK_NULL_HANDLE;
  VmaAllocator m_allocator = VK_NULL_HANDLE;
  VkQueue m_queue = VK_NULL_HANDLE;
  u32 m_copy_family = 0;
  u32 m_graphics_family = 0;

  VkCommandPool m_pool = VK_NULL_HANDLE;
  VkSemaphore m_timeline = VK_NULL_HANDLE;
  u64 m_value = 0; ///< Last value submitted (signalled) on the timeline.

  mutable nx::mutex m_mutex;

  // Staging ring (monotonic head/tail; offset = head % capacity).
  VkBuffer m_ring = VK_NULL_HANDLE;
  VmaAllocation m_ring_alloc = VK_NULL_HANDLE;
  u8 *m_ring_mapped = nullptr;
  u64 m_ring_capacity = 0;
  u64 m_ring_head = 0;
  u64 m_ring_tail = 0;
  nx::small_vector<Marker, 32> m_markers;

  // Open batch.
  VkCommandBuffer m_batch_cmd = VK_NULL_HANDLE;
  bool m_batch_open = false;
  u64 m_batch_value = 0; ///< Value the open batch will signal once flushed.

  nx::small_vector<VkCommandBuffer, 8> m_free_cmds;
  struct Inflight {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    u64 value = 0;
  };
  nx::small_vector<Inflight, 8> m_inflight;

  // Dedicated buffers for oversized uploads and readbacks.
  nx::small_vector<Chunk, 8> m_chunks;

  /// Ownership acquires the next graphics frame must perform, and the timeline
  /// value it must wait on before it can. Only non-empty when the transfer
  /// queue is a distinct family.
  nx::small_vector<ImageAcquire, 32> m_image_acquires;
  nx::small_vector<BufferAcquire, 16> m_buffer_acquires;
  u64 m_pending_wait = 0;
};

} // namespace nxe::rhi
