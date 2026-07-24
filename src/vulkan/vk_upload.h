#pragma once

/**
 * @file vk_upload.h
 * @brief Asynchronous GPU upload and readback on the transfer queue.
 *
 * The uploader turns "put these bytes into this GPU resource" into copy-queue
 * work that overlaps rendering. Each call stages the data through a pooled
 * host-visible buffer, records a copy on the transfer queue, and returns an
 * UploadTicket naming the point on the uploader's own timeline semaphore where
 * the copy is done - it never blocks the caller.
 *
 * When the transfer queue is a distinct hardware queue, the copy ends with a
 * queue-family release barrier; the device injects the matching acquire barrier
 * into the next graphics frame (record_acquires) and makes that frame's submit
 * wait on the copy, so a freshly uploaded resource is safe to sample from the
 * frame its acquire is recorded onward.
 *
 * Every public method is thread-safe: decode on a job, then upload from that
 * job. The RHI owns no worker threads of its own.
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
  /// leaves it in @p final_state, ready for the graphics queue.
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
  // wait_idle(), or from a resource only the copy queue touches.

  ReadbackResult read_buffer(BufferHandle handle, u64 offset, u64 size);
  ReadbackResult read_texture(TextureHandle handle, u32 mip = 0, u32 layer = 0);

  // Completion

  [[nodiscard]] bool is_complete(UploadTicket ticket) const noexcept;
  void wait(UploadTicket ticket) noexcept;
  /// Blocks until every submitted upload and readback has completed.
  void flush() noexcept;

  // Device frame integration (called by VulkanDevice)

  /// Records queue-ownership acquire barriers for the copies finished since the
  /// last call into @p graphics_cmd, and returns the copy-timeline value the
  /// graphics submit must wait on (0 if nothing is pending).
  u64 record_acquires(VkCommandBuffer graphics_cmd);
  [[nodiscard]] VkSemaphore timeline() const noexcept { return m_timeline; }
  /// Returns staging buffers and command buffers whose copy has completed to
  /// their reuse pools.
  void reclaim() noexcept;

private:
  /// A pooled staging buffer. Freed for reuse (value -> 0) once its copy
  /// completes; the backing VkBuffer lives until shutdown, so a readback's
  /// mapped data survives until the chunk is handed out again.
  struct Chunk {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    u8 *mapped = nullptr;
    u64 capacity = 0;
    u64 value = 0;
    bool host_read = false;
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
  Chunk *acquire_chunk(u64 size, bool host_read);
  VkCommandBuffer begin_cmd();
  UploadTicket submit_locked(VkCommandBuffer cmd);

  VulkanDevice *m_device = nullptr;
  VkDevice m_vk = VK_NULL_HANDLE;
  VmaAllocator m_allocator = VK_NULL_HANDLE;
  VkQueue m_queue = VK_NULL_HANDLE;
  u32 m_copy_family = 0;
  u32 m_graphics_family = 0;

  VkCommandPool m_pool = VK_NULL_HANDLE;
  VkSemaphore m_timeline = VK_NULL_HANDLE;
  u64 m_value = 0;

  mutable nx::mutex m_mutex;
  nx::vector<Chunk> m_chunks;
  nx::small_vector<VkCommandBuffer, 8> m_free_cmds;

  struct Inflight {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    u64 value = 0;
  };
  nx::small_vector<Inflight, 8> m_inflight;

  /// Ownership acquires the next graphics frame must perform, and the timeline
  /// value it must wait on before it can. Only non-empty when the transfer
  /// queue is a distinct family.
  nx::vector<ImageAcquire> m_image_acquires;
  nx::vector<BufferAcquire> m_buffer_acquires;
  u64 m_pending_wait = 0;
};

} // namespace nxe::rhi
