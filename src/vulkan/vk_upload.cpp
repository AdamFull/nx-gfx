
#include "core/rendering/rhi/vulkan/vk_upload.h"

#include "core/rendering/rhi/vulkan/vk_device.h"

#include "core/foundation/diagnostics/log.h"
#include "core/foundation/diagnostics/profiler.h"

#include <chrono>
#include <cstring>
#include <limits>

namespace nxe::rhi {
namespace {

const nx::log::Category log_up = nx::log::category("upload");

constexpr u64 RING_CAPACITY = 32ull * 1024 * 1024;
constexpr u64 DEDICATED_UPLOAD_THRESHOLD = RING_CAPACITY / 2;

/// Buffer-copy offset must be a multiple of both 4 and the format's block size;
/// every rhi::Format's block size is a power of two, so max(4, size) satisfies
/// both.
[[nodiscard]] u64 copy_alignment(const Format format) noexcept {
  const u32 texel = format_texel_size(format);
  return texel < 4u ? 4u : texel;
}

VkImageMemoryBarrier2
whole_image_barrier(const VkImage image,
                    const VkImageAspectFlags aspect) noexcept {
  VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0,
                              VK_REMAINING_ARRAY_LAYERS};
  return barrier;
}

} // namespace

nx::string_view to_string(const LoadStatus status) noexcept {
  switch (status) {
  case LoadStatus::Pending:
    return "pending";
  case LoadStatus::Decoding:
    return "decoding";
  case LoadStatus::Uploading:
    return "uploading";
  case LoadStatus::Ready:
    return "ready";
  case LoadStatus::Failed:
    return "failed";
  default:
    return "none";
  }
}

VulkanUploader::~VulkanUploader() { shutdown(); }

bool VulkanUploader::init(VulkanDevice *device) {
  m_device = device;
  m_vk = device->m_device;
  m_allocator = device->m_allocator;
  m_copy_family = device->m_queue_families[nx::cast<usize>(QueueType::Copy)];
  m_graphics_family =
      device->m_queue_families[nx::cast<usize>(QueueType::Graphics)];
  m_queue = device->m_queues[nx::cast<usize>(QueueType::Copy)];

  VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = m_copy_family;
  if (NX_VK_CHECK(vkCreateCommandPool(m_vk, &pool_info, nullptr, &m_pool)) !=
      VK_SUCCESS)
    return false;

  VkSemaphoreTypeCreateInfo type_info{
      VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo sem_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  sem_info.pNext = &type_info;
  if (NX_VK_CHECK(vkCreateSemaphore(m_vk, &sem_info, nullptr, &m_timeline)) !=
      VK_SUCCESS)
    return false;
  m_device->set_object_name(vk_handle_value(m_timeline),
                            VK_OBJECT_TYPE_SEMAPHORE, "upload timeline");

  VkBufferCreateInfo buf_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buf_info.size = RING_CAPACITY;
  buf_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VmaAllocationCreateInfo ring_ci{};
  ring_ci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  ring_ci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                  VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  VmaAllocationInfo ring_info{};
  if (NX_VK_CHECK(vmaCreateBuffer(m_allocator, &buf_info, &ring_ci, &m_ring,
                                  &m_ring_alloc, &ring_info)) != VK_SUCCESS) {
    nx::loge(log_up, "staging ring ({} MiB) allocation failed",
             RING_CAPACITY / (1024 * 1024));
    return false;
  }
  m_ring_mapped = nx::cast<u8 *>(ring_info.pMappedData);
  m_ring_capacity = RING_CAPACITY;
  m_device->set_object_name(vk_handle_value(m_ring), VK_OBJECT_TYPE_BUFFER,
                            "upload ring");

  for (u32 s = 0; s < MAX_LOADS; ++s) {
    m_load_state[s].store(LoadStatus::None, std::memory_order_relaxed);
    m_load_gen[s].store(0, std::memory_order_relaxed);
    m_free_slots.try_push(s);
  }
  m_worker_stop.store(false, std::memory_order_relaxed);
  m_worker = nx::thread([this] { worker_main(); },
                        nx::thread_options{.name = "nx-loader"});
  if (!m_worker.joinable()) {
    nx::loge(log_up, "failed to start the loader thread");
    return false;
  }

  nx::logi(log_up,
           "uploader: transfer queue family {} ({}), {} MiB ring, "
           "loader thread up",
           m_copy_family,
           needs_ownership_transfer() ? "dedicated" : "shared with graphics",
           RING_CAPACITY / (1024 * 1024));
  return true;
}

void VulkanUploader::shutdown() noexcept {
  if (m_vk == VK_NULL_HANDLE)
    return;

  m_worker_stop.store(true, std::memory_order_release);
  m_worker_wake.release();
  if (m_worker.joinable())
    m_worker.join();

  LoadCommand cmd;
  while (m_load_requests.try_pop(cmd))
    nx::release(cmd.payload);
  DecodeResult res;
  while (m_load_results.try_pop(res))
    nx::release(res.image);

  if (m_ring != VK_NULL_HANDLE)
    vmaDestroyBuffer(m_allocator, m_ring, m_ring_alloc);
  m_ring = VK_NULL_HANDLE;

  for (Chunk &chunk : m_chunks)
    if (chunk.buffer != VK_NULL_HANDLE)
      vmaDestroyBuffer(m_allocator, chunk.buffer, chunk.alloc);
  m_chunks.clear();

  if (m_pool != VK_NULL_HANDLE)
    vkDestroyCommandPool(m_vk, m_pool, nullptr);
  m_pool = VK_NULL_HANDLE;
  m_free_cmds.clear();
  m_inflight.clear();
  m_markers.clear();

  if (m_timeline != VK_NULL_HANDLE)
    vkDestroySemaphore(m_vk, m_timeline, nullptr);
  m_timeline = VK_NULL_HANDLE;

  m_image_acquires.clear();
  m_image_releases.clear();
  m_buffer_acquires.clear();
  m_buffer_releases.clear();
  m_graphics_copies.clear();
  m_graphics_hold = 0;
  m_vk = VK_NULL_HANDLE;
}

u64 VulkanUploader::completed_value() const noexcept {
  u64 value = 0;
  if (m_timeline != VK_NULL_HANDLE)
    NX_VK_CHECK(vkGetSemaphoreCounterValueKHR(m_vk, m_timeline, &value));
  return value;
}

u64 VulkanUploader::graphics_completed() const noexcept {
  u64 value = 0;
  if (m_device != nullptr && m_device->m_timeline != VK_NULL_HANDLE)
    NX_VK_CHECK(
        vkGetSemaphoreCounterValueKHR(m_vk, m_device->m_timeline, &value));
  return value;
}

void VulkanUploader::wait_value(const u64 value) noexcept {
  if (value == 0)
    return;
  u64 v = value;
  VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  wait.semaphoreCount = 1;
  wait.pSemaphores = &m_timeline;
  wait.pValues = &v;
  NX_VK_CHECK(vkWaitSemaphoresKHR(m_vk, &wait, UINT64_MAX));
}

void VulkanUploader::wait_graphics(const u64 value) noexcept {
  if (value == 0 || m_device == nullptr ||
      m_device->m_timeline == VK_NULL_HANDLE)
    return;
  u64 v = value;
  VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  wait.semaphoreCount = 1;
  wait.pSemaphores = &m_device->m_timeline;
  wait.pValues = &v;
  NX_VK_CHECK(vkWaitSemaphoresKHR(m_vk, &wait, UINT64_MAX));
}

VkCommandBuffer VulkanUploader::begin_cmd() {
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (!m_free_cmds.empty()) {
    cmd = m_free_cmds.back();
    m_free_cmds.pop_back();
  } else {
    VkCommandBufferAllocateInfo alloc{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = m_pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    if (NX_VK_CHECK(vkAllocateCommandBuffers(m_vk, &alloc, &cmd)) != VK_SUCCESS)
      return VK_NULL_HANDLE;
  }

  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &begin);
  return cmd;
}

VkCommandBuffer VulkanUploader::ensure_batch() {
  if (m_batch_open)
    return m_batch_cmd;
  m_batch_value = m_value + 1;
  m_batch_cmd = begin_cmd();
  m_batch_open = m_batch_cmd != VK_NULL_HANDLE;
  return m_batch_cmd;
}

void VulkanUploader::submit_signal(const VkCommandBuffer cmd, const u64 value) {
  VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
  signal.semaphore = m_timeline;
  signal.value = value;
  signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

  VkCommandBufferSubmitInfo cmd_info{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
  cmd_info.commandBuffer = cmd;

  VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  submit.commandBufferInfoCount = 1;
  submit.pCommandBufferInfos = &cmd_info;
  submit.signalSemaphoreInfoCount = 1;
  submit.pSignalSemaphoreInfos = &signal;
  NX_VK_CHECK(vkQueueSubmit2KHR(m_queue, 1, &submit, VK_NULL_HANDLE));

  m_inflight.push_back({cmd, value});
}

void VulkanUploader::flush_locked() {
  if (!m_batch_open)
    return;
  vkEndCommandBuffer(m_batch_cmd);
  m_value = m_batch_value;
  submit_signal(m_batch_cmd, m_batch_value);
  m_markers.push_back({m_ring_head, m_batch_value, m_graphics_hold});
  m_batch_open = false;
  m_batch_cmd = VK_NULL_HANDLE;
}

void VulkanUploader::reclaim_locked() {
  const u64 completed = completed_value();
  const u64 gfx_completed = graphics_completed();

  while (!m_markers.empty() && m_markers.front().value <= completed &&
         m_markers.front().gfx_value <= gfx_completed) {
    m_ring_tail = m_markers.front().head;
    m_markers.erase(m_markers.begin());
  }

  for (Chunk &chunk : m_chunks)
    if (chunk.value <= completed && chunk.gfx_value <= gfx_completed) {
      chunk.value = 0;
      chunk.gfx_value = 0;
    }

  for (usize i = m_inflight.size(); i > 0; --i) {
    if (m_inflight[i - 1].value <= completed) {
      m_free_cmds.push_back(m_inflight[i - 1].cmd);
      m_inflight.erase(m_inflight.begin() + nx::cast<isize>(i - 1));
    }
  }
}

u8 *VulkanUploader::ring_alloc(const u64 size, const u64 alignment,
                               u64 *out_offset) {
  if (size > m_ring_capacity)
    return nullptr;

  const auto try_alloc = [&](u64 &offset) -> bool {
    u64 base = nx::align_up(m_ring_head, alignment);
    u64 off = base % m_ring_capacity;
    if (off + size > m_ring_capacity) {
      base = (base - off) + m_ring_capacity;
      off = 0;
    }
    if ((base + size) - m_ring_tail > m_ring_capacity)
      return false;
    m_ring_head = base + size;
    offset = off;
    return true;
  };

  u64 offset = 0;
  for (;;) {
    if (try_alloc(offset)) {
      *out_offset = offset;
      return m_ring_mapped + offset;
    }

    reclaim_locked();
    if (try_alloc(offset)) {
      *out_offset = offset;
      return m_ring_mapped + offset;
    }

    if (m_batch_open)
      flush_locked();
    if (m_markers.empty())
      return nullptr;

    // The oldest bytes can be held by a frame that is still being recorded.
    // Waiting on a value nothing has submitted yet would hang the render
    // thread against itself, so let the caller fall back to dedicated staging.
    const Marker oldest = m_markers.front();
    if (m_device != nullptr && oldest.gfx_value > m_device->m_timeline_value)
      return nullptr;
    wait_value(oldest.value);
    wait_graphics(oldest.gfx_value);
    reclaim_locked();
  }
}

VulkanUploader::Chunk *VulkanUploader::acquire_chunk(const u64 size,
                                                     const bool host_read) {
  for (Chunk &chunk : m_chunks) {
    if (chunk.value == 0 && chunk.gfx_value == 0 &&
        chunk.host_read == host_read && chunk.capacity >= size)
      return &chunk;
  }

  Chunk chunk;
  chunk.capacity = size;
  chunk.host_read = host_read;

  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = chunk.capacity;
  info.usage = host_read ? VK_BUFFER_USAGE_TRANSFER_DST_BIT
                         : VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo alloc{};
  alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  alloc.flags =
      VMA_ALLOCATION_CREATE_MAPPED_BIT |
      (host_read ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                 : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  VmaAllocationInfo alloc_info{};
  if (NX_VK_CHECK(vmaCreateBuffer(m_allocator, &info, &alloc, &chunk.buffer,
                                  &chunk.alloc, &alloc_info)) != VK_SUCCESS) {
    nx::loge(log_up, "dedicated staging of {} bytes failed", chunk.capacity);
    return nullptr;
  }
  chunk.mapped = nx::cast<u8 *>(alloc_info.pMappedData);
  m_device->set_object_name(
      vk_handle_value(chunk.buffer), VK_OBJECT_TYPE_BUFFER,
      host_read ? "readback staging" : "oversized upload");

  m_chunks.push_back(chunk);
  return &m_chunks.back();
}

u8 *VulkanUploader::acquire_upload_staging(const u64 size, const u64 alignment,
                                           VkBuffer &buffer, u64 &offset,
                                           Chunk *&dedicated) {
  buffer = VK_NULL_HANDLE;
  offset = 0;
  dedicated = nullptr;

  // Large transfers should not monopolise the shared ring. This also avoids
  // the impossible case where a ring-sized upload is requested after its
  // monotonic head has advanced away from the physical buffer boundary.
  if (size < DEDICATED_UPLOAD_THRESHOLD) {
    if (u8 *const staging = ring_alloc(size, alignment, &offset)) {
      buffer = m_ring;
      return staging;
    }
  }

  dedicated = acquire_chunk(size, false);
  if (dedicated == nullptr)
    return nullptr;
  buffer = dedicated->buffer;
  return dedicated->mapped;
}

UploadTicket
VulkanUploader::upload_texture(const TextureHandle handle,
                               const std::span<const u8> src,
                               const std::span<const ImageSubresource> layout,
                               const ResourceState final_state) {
  NX_PROFILE_ZONE("vk::upload_texture");
  const nx::scoped_lock lock(m_command_mutex);
  VulkanTexture *tex = m_device->m_textures.try_get(handle);
  if (tex == nullptr || tex->image == VK_NULL_HANDLE || layout.empty()) {
    nx::loge(log_up, "upload_texture: nothing to upload into");
    return UploadTicket::failure();
  }

  const VkImageAspectFlags aspect = to_vk_aspect(tex->format);
  const u64 align = copy_alignment(tex->format);
  const bool qfot = needs_ownership_transfer();
  // The copy queue cannot write an image the graphics family owns: keeping the
  // untouched parts - the whole point of a partial upload - would need an
  // acquire, and its matching release can only come from a graphics submit.
  const bool on_graphics = qfot && tex->owner == ImageOwner::Graphics;

  u64 staging_size = 0;
  const u64 source_size = src.size();
  for (const ImageSubresource &sr : layout) {
    if (sr.size == 0 || sr.offset > source_size ||
        sr.size > source_size - sr.offset || sr.mip >= tex->mip_levels ||
        sr.layer >= tex->layers || sr.width == 0 || sr.height == 0 ||
        sr.depth == 0) {
      nx::loge(log_up, "upload_texture: malformed source subresource");
      return UploadTicket::failure();
    }
    const u32 mip_width = nx::max(1u, tex->width >> sr.mip);
    const u32 mip_height = nx::max(1u, tex->height >> sr.mip);
    const u32 mip_depth = nx::max(1u, tex->depth >> sr.mip);
    if (sr.x > mip_width || sr.width > mip_width - sr.x || sr.y > mip_height ||
        sr.height > mip_height - sr.y || sr.z > mip_depth ||
        sr.depth > mip_depth - sr.z ||
        staging_size > std::numeric_limits<u64>::max() - (align - 1u)) {
      nx::loge(log_up, "upload_texture: subresource is outside the texture");
      return UploadTicket::failure();
    }
    const u64 aligned = nx::align_up(staging_size, align);
    if (sr.size > std::numeric_limits<u64>::max() - aligned) {
      nx::loge(log_up, "upload_texture: staging size overflow");
      return UploadTicket::failure();
    }
    staging_size = aligned + sr.size;
  }

  VkBuffer staging_buf = VK_NULL_HANDLE;
  u64 base_offset = 0;
  Chunk *dedicated = nullptr;
  u8 *const staging = acquire_upload_staging(staging_size, align, staging_buf,
                                             base_offset, dedicated);
  if (staging == nullptr) {
    nx::loge(log_up, "upload_texture: no staging for {} bytes", staging_size);
    return UploadTicket::failure();
  }

  const VkCommandBuffer cmd = on_graphics ? VK_NULL_HANDLE : ensure_batch();
  if (!on_graphics && cmd == VK_NULL_HANDLE) {
    nx::loge(log_up, "upload_texture: no command batch to record into");
    return UploadTicket::failure();
  }

  nx::small_vector<VkBufferImageCopy, 16> copies;
  u64 offset = 0;
  for (const ImageSubresource &sr : layout) {
    offset = nx::align_up(offset, align);
    std::memcpy(staging + offset, src.data() + sr.offset,
                nx::cast<usize>(sr.size));

    VkBufferImageCopy copy{};
    copy.bufferOffset = base_offset + offset;
    copy.imageSubresource = {aspect, sr.mip, sr.layer, 1};
    copy.imageOffset = {nx::cast<i32>(sr.x), nx::cast<i32>(sr.y),
                        nx::cast<i32>(sr.z)};
    copy.imageExtent = {sr.width, sr.height, sr.depth == 0 ? 1u : sr.depth};
    copies.push_back(copy);
    offset += sr.size;
  }

  if (on_graphics) {
    GraphicsCopy *pending = nullptr;
    for (GraphicsCopy &gc : m_graphics_copies)
      if (gc.image == tex->image && gc.buffer == staging_buf)
        pending = &gc;
    if (pending == nullptr) {
      m_graphics_copies.push_back({handle,
                                   tex->image,
                                   aspect,
                                   tex->state,
                                   final_state,
                                   staging_buf,
                                   {}});
      pending = &m_graphics_copies.back();
    }
    pending->old_state = tex->state;
    pending->new_state = final_state;
    for (const VkBufferImageCopy &copy : copies)
      pending->regions.push_back(copy);

    // This is a lower bound - the frame that gets it is not known until
    // replay_graphics_copies() - but it is enough to stop the ring reclaiming
    // the bytes off the copy timeline alone.
    const u64 hold = m_device->m_timeline_value + 1;
    if (hold > m_graphics_hold)
      m_graphics_hold = hold;
    if (dedicated != nullptr)
      dedicated->gfx_value = hold;

    tex->state = final_state;
    return {};
  }

  const VkStateInfo from = to_vk_state(tex->state);
  // nothing is lost, since a stage mask cannot order work on another queue
  // anyway - the release/acquire pair carries that dependency.
  const bool copy_to_copy = tex->state == ResourceState::CopyDst;
  VkImageMemoryBarrier2 pre = whole_image_barrier(tex->image, aspect);
  pre.srcStageMask =
      (qfot && !copy_to_copy) ? VK_PIPELINE_STAGE_2_NONE : from.stage;
  pre.srcAccessMask =
      copy_to_copy ? VK_ACCESS_2_TRANSFER_WRITE_BIT : VK_ACCESS_2_NONE;
  pre.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  pre.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  pre.oldLayout = from.layout;
  pre.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  VkDependencyInfo pre_dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  pre_dep.imageMemoryBarrierCount = 1;
  pre_dep.pImageMemoryBarriers = &pre;
  vkCmdPipelineBarrier2KHR(cmd, &pre_dep);

  vkCmdCopyBufferToImage(cmd, staging_buf, tex->image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         nx::cast<u32>(copies.size()), copies.data());

  if (!qfot) {
    const VkStateInfo fin = to_vk_state(final_state);
    VkImageMemoryBarrier2 post = whole_image_barrier(tex->image, aspect);
    post.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    post.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    post.dstStageMask = fin.stage;
    post.dstAccessMask = fin.access;
    post.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    post.newLayout = fin.layout;
    VkDependencyInfo post_dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    post_dep.imageMemoryBarrierCount = 1;
    post_dep.pImageMemoryBarriers = &post;
    vkCmdPipelineBarrier2KHR(cmd, &post_dep);
  }

  if (dedicated != nullptr)
    dedicated->value = m_batch_value;
  if (m_batch_value > m_pending_wait)
    m_pending_wait = m_batch_value;

  if (qfot) {
    tex->state = ResourceState::CopyDst;
    tex->owner = ImageOwner::Copy;
    ImageRelease &release =
        pending_release(handle, tex->image, aspect, final_state);
    release.old_state = ResourceState::CopyDst;
    release.new_state = final_state;
  } else {
    tex->state = final_state;
  }

  return {m_batch_value};
}

UploadTicket VulkanUploader::upload_texture(const TextureHandle handle,
                                            const ImageData &image,
                                            const ResourceState final_state) {
  return upload_texture(
      handle, std::span<const u8>(image.pixels.data(), image.pixels.size()),
      std::span<const ImageSubresource>(image.subresources.data(),
                                        image.subresources.size()),
      final_state);
}

UploadTicket VulkanUploader::upload_buffer(const BufferHandle handle,
                                           const u64 dst_offset,
                                           const std::span<const u8> src,
                                           const ResourceState final_state) {
  NX_PROFILE_ZONE("vk::upload_buffer");
  const nx::scoped_lock lock(m_command_mutex);
  VulkanBuffer *buf = m_device->m_buffers.try_get(handle);
  if (src.empty())
    return {};
  if (buf == nullptr || buf->buffer == VK_NULL_HANDLE) {
    nx::loge(log_up, "upload_buffer: nothing to upload into");
    return UploadTicket::failure();
  }

  VkBuffer staging_buf = VK_NULL_HANDLE;
  u64 base_offset = 0;
  Chunk *dedicated = nullptr;
  u8 *const staging = acquire_upload_staging(src.size(), 16, staging_buf,
                                             base_offset, dedicated);
  if (staging == nullptr) {
    nx::loge(log_up, "upload_buffer: no staging for {} bytes", src.size());
    return UploadTicket::failure();
  }
  std::memcpy(staging, src.data(), src.size());

  const VkCommandBuffer cmd = ensure_batch();
  if (cmd == VK_NULL_HANDLE) {
    nx::loge(log_up, "upload_buffer: no command batch to record into");
    return UploadTicket::failure();
  }

  VkBufferCopy region{};
  region.srcOffset = base_offset;
  region.dstOffset = dst_offset;
  region.size = src.size();

  VkBufferMemoryBarrier2 before_copy{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
  before_copy.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
  before_copy.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  before_copy.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  before_copy.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  before_copy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  before_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  before_copy.buffer = buf->buffer;
  before_copy.offset = dst_offset;
  before_copy.size = src.size();
  VkDependencyInfo before_copy_dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  before_copy_dep.bufferMemoryBarrierCount = 1;
  before_copy_dep.pBufferMemoryBarriers = &before_copy;
  vkCmdPipelineBarrier2KHR(cmd, &before_copy_dep);

  vkCmdCopyBuffer(cmd, staging_buf, buf->buffer, 1, &region);

  const bool qfot = needs_ownership_transfer();
  if (qfot) {
    // Do not release ownership after every patch: a later upload in this batch
    // still belongs to the copy queue. record_releases() emits one hand-off
    // after all copies have been recorded.
    (void)pending_release(handle, buf->buffer, dst_offset, src.size(),
                          final_state);
  } else {
    const VkStateInfo fin = to_vk_state(final_state);
    VkBufferMemoryBarrier2 bar{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    bar.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    bar.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    bar.dstStageMask = fin.stage;
    bar.dstAccessMask = fin.access;
    bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.buffer = buf->buffer;
    bar.offset = dst_offset;
    bar.size = src.size();
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.bufferMemoryBarrierCount = 1;
    dep.pBufferMemoryBarriers = &bar;
    vkCmdPipelineBarrier2KHR(cmd, &dep);
  }

  if (dedicated != nullptr)
    dedicated->value = m_batch_value;
  if (m_batch_value > m_pending_wait)
    m_pending_wait = m_batch_value;
  return {m_batch_value};
}

UploadTicket VulkanUploader::clear_buffer(const BufferHandle handle,
                                          const ResourceState final_state) {
  NX_PROFILE_ZONE("vk::clear_buffer");
  const nx::scoped_lock lock(m_command_mutex);
  VulkanBuffer *buf = m_device->m_buffers.try_get(handle);
  if (buf == nullptr || buf->buffer == VK_NULL_HANDLE) {
    nx::loge(log_up, "clear_buffer: handle is not live");
    return UploadTicket::failure();
  }
  if (!(buf->usage & BufferUsage::CopyDst)) {
    nx::loge(log_up, "clear_buffer: buffer has no copy-destination usage");
    return UploadTicket::failure();
  }

  const VkCommandBuffer cmd = ensure_batch();
  if (cmd == VK_NULL_HANDLE) {
    nx::loge(log_up, "clear_buffer: no command batch to record into");
    return UploadTicket::failure();
  }

  VkBufferMemoryBarrier2 before_fill{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
  before_fill.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
  before_fill.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  before_fill.dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
  before_fill.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  before_fill.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  before_fill.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  before_fill.buffer = buf->buffer;
  before_fill.offset = 0;
  before_fill.size = buf->size;
  VkDependencyInfo before_fill_dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  before_fill_dep.bufferMemoryBarrierCount = 1;
  before_fill_dep.pBufferMemoryBarriers = &before_fill;
  vkCmdPipelineBarrier2KHR(cmd, &before_fill_dep);

  vkCmdFillBuffer(cmd, buf->buffer, 0, VK_WHOLE_SIZE, 0);

  const bool qfot = needs_ownership_transfer();
  if (qfot) {
    (void)pending_release(handle, buf->buffer, 0, buf->size, final_state);
  } else {
    const VkStateInfo fin = to_vk_state(final_state);
    VkBufferMemoryBarrier2 after_fill{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    after_fill.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    after_fill.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    after_fill.dstStageMask = fin.stage;
    after_fill.dstAccessMask = fin.access;
    after_fill.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    after_fill.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    after_fill.buffer = buf->buffer;
    after_fill.offset = 0;
    after_fill.size = buf->size;
    VkDependencyInfo after_fill_dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    after_fill_dep.bufferMemoryBarrierCount = 1;
    after_fill_dep.pBufferMemoryBarriers = &after_fill;
    vkCmdPipelineBarrier2KHR(cmd, &after_fill_dep);
  }

  if (m_batch_value > m_pending_wait)
    m_pending_wait = m_batch_value;
  return {m_batch_value};
}

LoadRequest VulkanUploader::enqueue_load(LoadPayload *payload) {
  u32 slot = 0;
  if (!m_free_slots.try_pop(slot)) {
    nx::loge(log_up, "loader: more than {} loads in flight", MAX_LOADS);
    nx::release(payload);
    return {};
  }
  const u32 gen = m_load_gen[slot].load(std::memory_order_relaxed);
  m_load_handle[slot] = {};
  m_load_ticket[slot] = {};
  m_load_state[slot].store(LoadStatus::Pending, std::memory_order_release);

  LoadCommand cmd{slot + 1, gen, payload};
  while (!m_load_requests.try_push(cmd))
    nx::this_thread::yield();
  m_worker_wake.release();
  return {slot + 1, gen};
}

LoadRequest VulkanUploader::load_texture(const nx::string_view path,
                                         const TextureUsageFlags usage) {
  auto *payload = nx::allocate<LoadPayload>();
  if (payload == nullptr)
    return {};
  payload->path = nx::string(path);
  payload->usage = usage;
  return enqueue_load(payload);
}

LoadRequest
VulkanUploader::load_texture_from_memory(const std::span<const u8> bytes,
                                         const TextureUsageFlags usage) {
  auto *payload = nx::allocate<LoadPayload>();
  if (payload == nullptr)
    return {};
  payload->bytes = nx::blob<u8>(bytes);
  payload->usage = usage;
  return enqueue_load(payload);
}

LoadStatus VulkanUploader::load_status(const LoadRequest req) const noexcept {
  if (!req.valid() || req.id > MAX_LOADS)
    return LoadStatus::None;
  const u32 s = req.id - 1;
  if (m_load_gen[s].load(std::memory_order_acquire) != req.generation)
    return LoadStatus::None;
  return m_load_state[s].load(std::memory_order_acquire);
}

TextureHandle
VulkanUploader::loaded_texture(const LoadRequest req) const noexcept {
  if (!req.valid() || req.id > MAX_LOADS)
    return {};
  const u32 s = req.id - 1;
  if (m_load_gen[s].load(std::memory_order_acquire) != req.generation)
    return {};
  if (m_load_state[s].load(std::memory_order_acquire) != LoadStatus::Ready)
    return {};
  return m_load_handle[s];
}

void VulkanUploader::release(const LoadRequest req) noexcept {
  if (!req.valid() || req.id > MAX_LOADS)
    return;
  const u32 s = req.id - 1;
  if (m_load_gen[s].load(std::memory_order_acquire) != req.generation)
    return;
  m_load_state[s].store(LoadStatus::None, std::memory_order_release);
  m_load_handle[s] = {};
  m_load_gen[s].fetch_add(1, std::memory_order_acq_rel);
  while (!m_free_slots.try_push(s))
    nx::this_thread::yield();
}

void VulkanUploader::worker_main() {
  while (!m_worker_stop.load(std::memory_order_acquire)) {
    LoadCommand cmd;
    if (!m_load_requests.try_pop(cmd)) {
      (void)m_worker_wake.try_acquire_for(std::chrono::milliseconds(100));
      continue;
    }

    const u32 s = cmd.id - 1;
    LoadPayload *pl = cmd.payload;
    if (m_load_gen[s].load(std::memory_order_acquire) == cmd.generation)
      m_load_state[s].store(LoadStatus::Decoding, std::memory_order_release);

    auto decoded = pl->path.empty()
                       ? load_image_from_memory(std::span<const u8>(
                             pl->bytes.data(), pl->bytes.size()))
                       : load_image(pl->path);
    const TextureUsageFlags usage = pl->usage;
    nx::release(pl);

    if (!decoded) {
      nx::loge(log_up, "loader: decode failed: {}", to_string(decoded.error()));
      if (m_load_gen[s].load(std::memory_order_acquire) == cmd.generation)
        m_load_state[s].store(LoadStatus::Failed, std::memory_order_release);
      m_loads_failed.fetch_add(1, std::memory_order_relaxed);
      continue;
    }

    ImageData *image = nx::allocate<ImageData>(std::move(*decoded));
    if (image == nullptr) {
      if (m_load_gen[s].load(std::memory_order_acquire) == cmd.generation)
        m_load_state[s].store(LoadStatus::Failed, std::memory_order_release);
      m_loads_failed.fetch_add(1, std::memory_order_relaxed);
      continue;
    }

    DecodeResult res{cmd.id, cmd.generation, image, usage};
    while (!m_load_results.try_push(res)) {
      if (m_worker_stop.load(std::memory_order_acquire)) {
        nx::release(res.image);
        break;
      }
      nx::this_thread::yield();
    }
  }
}

void VulkanUploader::pump() {
  m_load_results.consume_all([this](DecodeResult res) {
    const u32 s = res.id - 1;
    if (m_load_gen[s].load(std::memory_order_acquire) != res.generation) {
      nx::release(res.image);
      return;
    }

    const TextureHandle handle =
        m_device->create_texture(res.image->texture_desc("loaded", res.usage));
    if (!handle.valid()) {
      nx::release(res.image);
      m_load_state[s].store(LoadStatus::Failed, std::memory_order_release);
      m_loads_failed.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    const UploadTicket ticket = upload_texture(handle, *res.image);
    nx::release(res.image);

    // Without this the load reaches Ready and hands out a texture holding
    // whatever was in that memory, which is worse than reporting the failure.
    if (!ticket.ok()) {
      m_device->destroy_texture(handle);
      m_load_state[s].store(LoadStatus::Failed, std::memory_order_release);
      m_loads_failed.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    m_load_handle[s] = handle;
    m_load_ticket[s] = ticket;
    m_load_state[s].store(LoadStatus::Uploading, std::memory_order_release);
    m_load_uploading.push_back({res.id, res.generation});
  });

  {
    const nx::scoped_lock lock(m_command_mutex);
    for (const LoadRequest &pending : m_load_uploading) {
      if (m_load_ticket[pending.id - 1].value > m_value) {
        flush_locked();
        break;
      }
    }
  }

  for (usize i = m_load_uploading.size(); i > 0; --i) {
    const LoadRequest u = m_load_uploading[i - 1];
    const u32 s = u.id - 1;
    const bool stale =
        m_load_gen[s].load(std::memory_order_acquire) != u.generation;
    if (!stale && !is_complete(m_load_ticket[s]))
      continue;

    if (!stale) {
      m_load_state[s].store(LoadStatus::Ready, std::memory_order_release);
      m_loads_ready.fetch_add(1, std::memory_order_relaxed);
    }
    m_load_uploading.erase(m_load_uploading.begin() + nx::cast<isize>(i - 1));
  }
}

ReadbackResult VulkanUploader::read_buffer(const BufferHandle handle,
                                           const u64 offset, const u64 size) {
  const nx::scoped_lock lock(m_command_mutex);
  VulkanBuffer *buf = m_device->m_buffers.try_get(handle);
  if (buf == nullptr || buf->buffer == VK_NULL_HANDLE || size == 0)
    return {};

  if (!(buf->usage & BufferUsage::CopySrc)) {
    nx::loge(log_up, "read_buffer: buffer was not created with CopySrc");
    return {};
  }

  if (offset > buf->size || size > buf->size - offset) {
    nx::logw(log_up, "read_buffer: {}..{} is outside the {}-byte buffer",
             offset, offset + size, buf->size);
    return {};
  }

  flush_locked();

  Chunk *chunk = acquire_chunk(size, true);
  if (chunk == nullptr)
    return {};

  const VkCommandBuffer cmd = begin_cmd();
  if (cmd == VK_NULL_HANDLE)
    return {};

  VkBufferMemoryBarrier2 before_read{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
  before_read.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  before_read.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  before_read.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  before_read.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
  before_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  before_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  before_read.buffer = buf->buffer;
  before_read.offset = offset;
  before_read.size = size;
  VkDependencyInfo before_read_dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  before_read_dep.bufferMemoryBarrierCount = 1;
  before_read_dep.pBufferMemoryBarriers = &before_read;
  vkCmdPipelineBarrier2KHR(cmd, &before_read_dep);

  VkBufferCopy region{};
  region.srcOffset = offset;
  region.dstOffset = 0;
  region.size = size;
  vkCmdCopyBuffer(cmd, buf->buffer, chunk->buffer, 1, &region);

  VkMemoryBarrier2 host{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  host.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  host.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  host.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
  host.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
  VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dep.memoryBarrierCount = 1;
  dep.pMemoryBarriers = &host;
  vkCmdPipelineBarrier2KHR(cmd, &dep);

  vkEndCommandBuffer(cmd);
  const u64 value = ++m_value;
  submit_signal(cmd, value);
  chunk->value = value;
  return {{value}, chunk->mapped, size};
}

ReadbackResult VulkanUploader::read_texture(const TextureHandle handle,
                                            const u32 mip, const u32 layer) {
  const nx::scoped_lock lock(m_command_mutex);
  VulkanTexture *tex = m_device->m_textures.try_get(handle);
  if (tex == nullptr || tex->image == VK_NULL_HANDLE)
    return {};
  if (mip >= tex->mip_levels || layer >= tex->layers) {
    nx::logw(log_up,
             "read_texture: mip {} / layer {} is outside the {}-mip, "
             "{}-layer texture",
             mip, layer, tex->mip_levels, tex->layers);
    return {};
  }

  const u32 w = (tex->width >> mip) == 0 ? 1u : (tex->width >> mip);
  const u32 h = (tex->height >> mip) == 0 ? 1u : (tex->height >> mip);
  const u64 size = format_subresource_size(tex->format, w, h, 1);
  const VkImageAspectFlags aspect = to_vk_aspect(tex->format);

  flush_locked();

  Chunk *chunk = acquire_chunk(size, true);
  if (chunk == nullptr)
    return {};

  const VkCommandBuffer cmd = begin_cmd();
  if (cmd == VK_NULL_HANDLE)
    return {};

  const VkStateInfo from = to_vk_state(tex->state);
  const bool qfot = needs_ownership_transfer();
  if (qfot && tex->owner == ImageOwner::Graphics) {
    nx::logw(log_up,
             "read_texture: the graphics family owns this image, so what the "
             "copy queue reads back is undefined. Read before presenting a "
             "frame, while the copy queue still owns what it uploaded.");
  }

  const bool copy_state = tex->state == ResourceState::CopyDst ||
                          tex->state == ResourceState::CopySrc;
  const bool foreign = qfot && !copy_state;
  VkImageMemoryBarrier2 to_src = whole_image_barrier(tex->image, aspect);
  to_src.srcStageMask = foreign ? VK_PIPELINE_STAGE_2_NONE : from.stage;
  to_src.srcAccessMask = foreign ? VK_ACCESS_2_NONE : from.access;
  to_src.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  to_src.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
  to_src.oldLayout = from.layout;
  to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dep.imageMemoryBarrierCount = 1;
  dep.pImageMemoryBarriers = &to_src;
  vkCmdPipelineBarrier2KHR(cmd, &dep);

  VkBufferImageCopy copy{};
  copy.imageSubresource = {aspect, mip, layer, 1};
  copy.imageExtent = {w, h, 1};
  vkCmdCopyImageToBuffer(cmd, tex->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         chunk->buffer, 1, &copy);
  tex->state = ResourceState::CopySrc;
  // Anything graphics does with it afterwards needs an acquire, and the only
  // barrier that can pair with one is the release the next frame hand-off emits
  // - so book it now rather than leaving the image stranded on this queue.
  if (qfot && tex->owner == ImageOwner::Copy) {
    ImageRelease &release =
        pending_release(handle, tex->image, aspect, ResourceState::CopySrc);
    release.old_state = ResourceState::CopySrc;
  }

  VkMemoryBarrier2 host{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  host.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  host.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  host.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
  host.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
  VkDependencyInfo host_dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  host_dep.memoryBarrierCount = 1;
  host_dep.pMemoryBarriers = &host;
  vkCmdPipelineBarrier2KHR(cmd, &host_dep);

  vkEndCommandBuffer(cmd);
  const u64 value = ++m_value;
  submit_signal(cmd, value);
  chunk->value = value;
  return {{value}, chunk->mapped, size};
}

bool VulkanUploader::is_complete(const UploadTicket ticket) const noexcept {
  return !ticket.valid() || completed_value() >= ticket.value;
}

void VulkanUploader::wait(const UploadTicket ticket) noexcept {
  if (!ticket.valid())
    return;
  {
    const nx::scoped_lock lock(m_command_mutex);
    if (ticket.value > m_value)
      flush_locked();
  }
  wait_value(ticket.value);
}

void VulkanUploader::flush() noexcept {
  NX_PROFILE_ZONE("vk::upload_flush");
  u64 value = 0;
  {
    const nx::scoped_lock lock(m_command_mutex);
    flush_locked();
    value = m_value;
  }
  wait_value(value);
  const nx::scoped_lock lock(m_command_mutex);
  reclaim_locked();
}

void VulkanUploader::flush_batch() noexcept {
  const nx::scoped_lock lock(m_command_mutex);
  // The frame boundary is the one point where the graphics queue is about to
  // record the matching acquires, so it is the only place a release may go.
  record_releases();
  flush_locked();
}

UploaderStats VulkanUploader::stats() const noexcept {
  const nx::scoped_lock lock(m_command_mutex);
  UploaderStats s;
  s.total_submits = m_value;
  s.completed_submits = completed_value();
  s.ring_used = m_ring_head - m_ring_tail;
  s.ring_capacity = m_ring_capacity;
  s.extra_buffers = m_chunks.size();
  s.inflight_submits = m_inflight.size();
  s.loads_ready = m_loads_ready.load(std::memory_order_relaxed);
  s.loads_failed = m_loads_failed.load(std::memory_order_relaxed);
  return s;
}

VulkanUploader::ImageRelease &
VulkanUploader::pending_release(const TextureHandle handle, const VkImage image,
                                const VkImageAspectFlags aspect,
                                const ResourceState state) {
  for (ImageRelease &release : m_image_releases)
    if (release.image == image)
      return release;
  m_image_releases.push_back({handle, image, aspect, state, state});
  return m_image_releases.back();
}

VulkanUploader::BufferRelease &
VulkanUploader::pending_release(const BufferHandle handle,
                                const VkBuffer buffer, const u64 offset,
                                const u64 size, const ResourceState state) {
  for (BufferRelease &release : m_buffer_releases) {
    if (release.buffer != buffer)
      continue;

    const u64 begin = release.offset < offset ? release.offset : offset;
    const u64 old_end = release.offset + release.size;
    const u64 new_end = offset + size;
    release.offset = begin;
    release.size = (old_end > new_end ? old_end : new_end) - begin;
    release.new_state = state;
    return release;
  }

  m_buffer_releases.push_back({handle, buffer, offset, size, state});
  return m_buffer_releases.back();
}

void VulkanUploader::record_releases() {
  if (m_image_releases.empty() && m_buffer_releases.empty())
    return;

  const VkCommandBuffer cmd = ensure_batch();
  if (cmd == VK_NULL_HANDLE)
    return;

  nx::small_vector<VkImageMemoryBarrier2, 32> barriers;
  for (const ImageRelease &release : m_image_releases) {
    // Destroying a texture the same frame it was uploaded is legal, and
    // collect_garbage has already run by the time we get here: no texture
    // means no VkImage to name in a barrier.
    VulkanTexture *tex = m_device->m_textures.try_get(release.handle);
    if (tex == nullptr)
      continue;

    const VkStateInfo from = to_vk_state(release.old_state);
    const VkStateInfo to = to_vk_state(release.new_state);
    VkImageMemoryBarrier2 barrier =
        whole_image_barrier(release.image, release.aspect);
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.srcAccessMask = release.old_state == ResourceState::CopySrc
                                ? VK_ACCESS_2_TRANSFER_READ_BIT
                                : VK_ACCESS_2_TRANSFER_WRITE_BIT;
    // The destination half belongs to the acquire on the other queue; a
    // release names no stages of its own.
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
    barrier.dstAccessMask = VK_ACCESS_2_NONE;
    barrier.oldLayout = from.layout;
    barrier.newLayout = to.layout;
    barrier.srcQueueFamilyIndex = m_copy_family;
    barrier.dstQueueFamilyIndex = m_graphics_family;
    barriers.push_back(barrier);

    // The acquire has to repeat this transition verbatim, so it is built from
    // the same pair rather than from whatever the texture says later.
    m_image_acquires.push_back(
        {release.image, release.aspect, from.layout, release.new_state});
    tex->state = release.new_state;
    tex->owner = ImageOwner::Graphics;
  }

  nx::small_vector<VkBufferMemoryBarrier2, 16> buffer_barriers;
  for (const BufferRelease &release : m_buffer_releases) {
    const VulkanBuffer *buf = m_device->m_buffers.try_get(release.handle);
    if (buf == nullptr || buf->buffer != release.buffer)
      continue;

    VkBufferMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
    barrier.dstAccessMask = VK_ACCESS_2_NONE;
    barrier.srcQueueFamilyIndex = m_copy_family;
    barrier.dstQueueFamilyIndex = m_graphics_family;
    barrier.buffer = release.buffer;
    barrier.offset = release.offset;
    barrier.size = release.size;
    buffer_barriers.push_back(barrier);

    m_buffer_acquires.push_back(
        {release.buffer, release.offset, release.size, release.new_state});
  }

  if (!barriers.empty() || !buffer_barriers.empty()) {
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = nx::cast<u32>(barriers.size());
    dep.pImageMemoryBarriers = barriers.data();
    dep.bufferMemoryBarrierCount = nx::cast<u32>(buffer_barriers.size());
    dep.pBufferMemoryBarriers = buffer_barriers.data();
    vkCmdPipelineBarrier2KHR(cmd, &dep);
  }

  m_image_releases.clear();
  m_buffer_releases.clear();
  if (m_batch_value > m_pending_wait)
    m_pending_wait = m_batch_value;
}

void VulkanUploader::replay_graphics_copies(const VkCommandBuffer cmd) {
  if (m_graphics_copies.empty())
    return;

  // Raise the hold to the frame about to be recorded and carry it to everything
  // already stamped with the guess, or the ring could hand those bytes out
  // before this frame has read them.
  const u64 hold = m_device->m_timeline_value + 1;
  if (hold > m_graphics_hold)
    m_graphics_hold = hold;
  for (Marker &marker : m_markers)
    if (marker.gfx_value != 0 && marker.gfx_value < m_graphics_hold)
      marker.gfx_value = m_graphics_hold;
  for (Chunk &chunk : m_chunks)
    if (chunk.gfx_value != 0 && chunk.gfx_value < m_graphics_hold)
      chunk.gfx_value = m_graphics_hold;

  // One transition per image, not per record: the first barrier moves it out
  // of its shader-read layout, so a second claiming the same oldLayout would
  // be describing a layout the image is no longer in.
  nx::small_vector<VkImageMemoryBarrier2, 8> pre;
  nx::small_vector<VkImageMemoryBarrier2, 8> post;
  for (const GraphicsCopy &copy : m_graphics_copies) {
    if (m_device->m_textures.try_get(copy.handle) == nullptr)
      continue;
    bool seen = false;
    for (const VkImageMemoryBarrier2 &barrier : pre)
      seen = seen || barrier.image == copy.image;
    if (seen)
      continue;

    const VkStateInfo from = to_vk_state(copy.old_state);
    const VkStateInfo to = to_vk_state(copy.new_state);
    // Everything here is one family's, so no ownership changes hands. The
    // first scope reaches back over every graphics submit before this one,
    // which is what keeps an earlier frame's sampling off these texels.
    VkImageMemoryBarrier2 in = whole_image_barrier(copy.image, copy.aspect);
    in.srcStageMask = from.stage;
    in.srcAccessMask = from.access;
    in.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    in.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    in.oldLayout = from.layout;
    in.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    pre.push_back(in);

    VkImageMemoryBarrier2 out = whole_image_barrier(copy.image, copy.aspect);
    out.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    out.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    out.dstStageMask = to.stage;
    out.dstAccessMask = to.access;
    out.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    out.newLayout = to.layout;
    post.push_back(out);
  }

  if (!pre.empty()) {
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = nx::cast<u32>(pre.size());
    dep.pImageMemoryBarriers = pre.data();
    vkCmdPipelineBarrier2KHR(cmd, &dep);

    for (const GraphicsCopy &copy : m_graphics_copies)
      if (m_device->m_textures.try_get(copy.handle) != nullptr)
        vkCmdCopyBufferToImage(
            cmd, copy.buffer, copy.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            nx::cast<u32>(copy.regions.size()), copy.regions.data());

    dep.imageMemoryBarrierCount = nx::cast<u32>(post.size());
    dep.pImageMemoryBarriers = post.data();
    vkCmdPipelineBarrier2KHR(cmd, &dep);
  }

  if (m_markers.empty() || m_markers.back().head != m_ring_head)
    m_markers.push_back({m_ring_head, m_value, m_graphics_hold});
  else if (m_graphics_hold > m_markers.back().gfx_value)
    m_markers.back().gfx_value = m_graphics_hold;

  m_graphics_copies.clear();
  m_graphics_hold = 0;
}

u64 VulkanUploader::record_acquires(const VkCommandBuffer graphics_cmd) {
  const nx::scoped_lock lock(m_command_mutex);
  // flush_batch() and this call used to be separate critical sections. An
  // update-thread upload could then open a new batch and publish its future
  // timeline value between them, making the graphics queue wait on work that
  // had not been submitted. Fold any such upload into this hand-off.
  record_releases();
  flush_locked();
  if (m_pending_wait == 0 && m_graphics_copies.empty())
    return 0;

  nx::small_vector<VkImageMemoryBarrier2, 32> images;
  for (const ImageAcquire &acq : m_image_acquires) {
    const VkStateInfo dst = to_vk_state(acq.new_state);
    VkImageMemoryBarrier2 barrier = whole_image_barrier(acq.image, acq.aspect);
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    barrier.srcAccessMask = VK_ACCESS_2_NONE;
    barrier.dstStageMask = dst.stage;
    barrier.dstAccessMask = dst.access;
    barrier.oldLayout = acq.old_layout;
    barrier.newLayout = dst.layout;
    barrier.srcQueueFamilyIndex = m_copy_family;
    barrier.dstQueueFamilyIndex = m_graphics_family;
    images.push_back(barrier);
  }

  nx::small_vector<VkBufferMemoryBarrier2, 16> buffers;
  for (const BufferAcquire &acq : m_buffer_acquires) {
    const VkStateInfo dst = to_vk_state(acq.new_state);
    VkBufferMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    barrier.srcAccessMask = VK_ACCESS_2_NONE;
    barrier.dstStageMask = dst.stage;
    barrier.dstAccessMask = dst.access;
    barrier.srcQueueFamilyIndex = m_copy_family;
    barrier.dstQueueFamilyIndex = m_graphics_family;
    barrier.buffer = acq.buffer;
    barrier.offset = acq.offset;
    barrier.size = acq.size;
    buffers.push_back(barrier);
  }

  if (!images.empty() || !buffers.empty()) {
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = nx::cast<u32>(images.size());
    dep.pImageMemoryBarriers = images.data();
    dep.bufferMemoryBarrierCount = nx::cast<u32>(buffers.size());
    dep.pBufferMemoryBarriers = buffers.data();
    vkCmdPipelineBarrier2KHR(graphics_cmd, &dep);
  }

  // After the acquires, so an image this frame is taking ownership of is not
  // also being patched here - a texture is only ever in one of the two lists.
  replay_graphics_copies(graphics_cmd);

  const u64 wait = m_pending_wait;
  m_image_acquires.clear();
  m_buffer_acquires.clear();
  m_pending_wait = 0;
  return wait;
}

void VulkanUploader::reclaim() noexcept {
  const nx::scoped_lock lock(m_command_mutex);
  reclaim_locked();
}

} // namespace nxe::rhi
