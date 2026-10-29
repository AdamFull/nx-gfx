#pragma once

#include "core/foundation/containers/small_vector.h"
#include "core/foundation/core/foundation.h"
#include "core/foundation/threading/sync.h"
#include "nx/gfx/types.h"
#include "nx/gfx/upload.h"
#include "src/vulkan/vk_common.h"
#include "src/vulkan/vk_mem_alloc_include.h"

#include <span>

namespace nx::gfx::detail {

class VulkanDevice;

class VulkanUploader {
public:
  VulkanUploader() = default;
  ~VulkanUploader();

  VulkanUploader(const VulkanUploader &) = delete;
  VulkanUploader &operator=(const VulkanUploader &) = delete;

  [[nodiscard]] bool init(VulkanDevice *device);
  void shutdown() noexcept;

  [[nodiscard]] gfx::upload_ticket upload_texture(
      gfx::texture_handle handle, std::span<const u8> src,
      std::span<const gfx::image_subresource> layout,
      gfx::resource_state final_state = gfx::resource_state::ShaderReadOnly);

  [[nodiscard]] gfx::upload_ticket upload_texture(
      gfx::texture_handle handle, const gfx::texture_data &data,
      gfx::resource_state final_state = gfx::resource_state::ShaderReadOnly);

  [[nodiscard]] gfx::upload_ticket upload_buffer(
      gfx::buffer_handle handle, u64 dst_offset, std::span<const u8> src,
      gfx::resource_state final_state = gfx::resource_state::ShaderReadOnly);

  [[nodiscard]] gfx::upload_ticket clear_buffer(
      gfx::buffer_handle handle,
      gfx::resource_state final_state = gfx::resource_state::ShaderReadOnly);

  // The source must not be in concurrent use on the graphics queue - read after
  // wait_idle(), or from a resource only the copy queue touches.

  gfx::readback_result read_buffer(gfx::buffer_handle handle, u64 offset,
                                   u64 size);
  gfx::readback_result read_texture(gfx::texture_handle handle, u32 mip = 0,
                                    u32 layer = 0);

  [[nodiscard]] bool is_complete(gfx::upload_ticket ticket) const noexcept;
  void wait(gfx::upload_ticket ticket) noexcept;
  void flush() noexcept;

  [[nodiscard]] gfx::uploader_stats stats() const noexcept;

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

  /// Bytes a graphics replay reads are held by `gfx_value` on the device
  /// timeline too, since the copy timeline says nothing about when the frame
  /// has run.
  struct Marker {
    u64 head = 0;
    u64 value = 0;
    u64 gfx_value = 0;
  };

  /// Uses the ring for ordinary transient copies and reusable dedicated
  /// staging for large copies or when the ring is temporarily unavailable.
  /// A full ring whose oldest bytes are already submitted returns null with
  /// @p held set: the caller waits for them with the lock released and
  /// retries, so no worker blocks on the GPU while holding m_command_mutex.
  u8 *acquire_upload_staging(u64 size, u64 alignment, VkBuffer &buffer,
                             u64 &offset, Chunk *&dedicated, Marker &held);
  [[nodiscard]] bool flush_upload_staging(const Chunk *dedicated, u64 offset,
                                          u64 size) const noexcept;

  struct ImageAcquire {
    VkImage image = VK_NULL_HANDLE;
    VkImageAspectFlags aspect = 0;
    VkImageLayout old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    gfx::resource_state new_state = gfx::resource_state::ShaderReadOnly;
  };
  /// An image the copy queue still owns and must hand to graphics at the next
  /// frame hand-off. One record per image however many copies touched it, so
  /// the release/acquire pair is emitted once and always names the layout the
  /// image is actually in.
  struct ImageRelease {
    gfx::texture_handle handle;
    VkImage image = VK_NULL_HANDLE;
    VkImageAspectFlags aspect = 0;
    gfx::resource_state old_state = gfx::resource_state::CopyDst;
    gfx::resource_state new_state = gfx::resource_state::ShaderReadOnly;
  };
  /// A copy into an image the graphics family owns, waiting to be replayed into
  /// the next frame's command buffer (see m_graphics_hold for how its staging
  /// bytes are kept alive until that frame has run).
  struct GraphicsCopy {
    gfx::texture_handle handle;
    VkImage image = VK_NULL_HANDLE;
    VkImageAspectFlags aspect = 0;
    gfx::resource_state old_state = gfx::resource_state::ShaderReadOnly;
    gfx::resource_state new_state = gfx::resource_state::ShaderReadOnly;
    VkBuffer buffer = VK_NULL_HANDLE;
    nx::small_vector<VkBufferImageCopy, 4> regions;
  };
  struct BufferAcquire {
    VkBuffer buffer = VK_NULL_HANDLE;
    u64 offset = 0;
    u64 size = 0;
    gfx::resource_state new_state = gfx::resource_state::ShaderReadOnly;
  };
  struct BufferRelease {
    gfx::buffer_handle handle;
    VkBuffer buffer = VK_NULL_HANDLE;
    u64 offset = 0;
    u64 size = 0;
    gfx::resource_state new_state = gfx::resource_state::ShaderReadOnly;
  };

  [[nodiscard]] u64 completed_value() const noexcept;
  [[nodiscard]] u64 graphics_completed() const noexcept;
  [[nodiscard]] bool needs_ownership_transfer() const noexcept {
    return m_copy_family != m_graphics_family;
  }

  VkCommandBuffer ensure_batch();
  void flush_locked();
  void reclaim_locked();
  u8 *ring_alloc(u64 size, u64 alignment, u64 *out_offset, Marker &held);
  /// The upload bodies, under m_command_mutex; @p held as for
  /// acquire_upload_staging.
  gfx::upload_ticket
  try_upload_texture(gfx::texture_handle handle, std::span<const u8> src,
                     std::span<const gfx::image_subresource> layout,
                     gfx::resource_state final_state, Marker &held);
  gfx::upload_ticket try_upload_buffer(gfx::buffer_handle handle,
                                       u64 dst_offset, std::span<const u8> src,
                                       gfx::resource_state final_state,
                                       Marker &held);
  /// Blocks the calling fiber, not its worker, until both timelines reach
  /// @p held.
  void wait_for(const Marker &held) noexcept;
  Chunk *acquire_chunk(u64 size, bool host_read);
  VkCommandBuffer begin_cmd();
  /// A readback first waits for the frames submitted before it.
  void submit_signal(VkCommandBuffer cmd, u64 value, bool after_frames = false);
  void wait_value(u64 value) noexcept;
  void wait_graphics(u64 value) noexcept;

  /// Finds or starts the pending hand-off record for @p image, so several
  /// copies into one texture share a single release/acquire pair.
  ImageRelease &pending_release(gfx::texture_handle handle, VkImage image,
                                VkImageAspectFlags aspect,
                                gfx::resource_state state);
  BufferRelease &pending_release(gfx::buffer_handle handle, VkBuffer buffer,
                                 u64 offset, u64 size,
                                 gfx::resource_state state);
  /// Emits the copy-side half of the hand-off into the open batch and turns
  /// each record into the acquire record_acquires() will emit.
  void record_releases();
  /// Replays the copies that must run on the graphics queue into @p cmd.
  void replay_graphics_copies(VkCommandBuffer cmd);

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
};

} // namespace nx::gfx::detail
