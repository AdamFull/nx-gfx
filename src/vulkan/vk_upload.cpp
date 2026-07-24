/**
 * @file vk_upload.cpp
 * @brief Batched transfer-queue staging, copies and queue-ownership transfer.
 *
 * Uploads accumulate into one command buffer and stage through a single ring
 * buffer; the batch is submitted once per frame (flush_batch), on staging
 * backpressure, or on an explicit flush/wait. An upload larger than the whole
 * ring, and every readback, fall back to a dedicated buffer instead.
 */

#include "rendering/rhi/vulkan/vk_upload.h"

#include "rendering/rhi/vulkan/vk_device.h"

#include "foundation/diagnostics/log.h"

#include <cstring>

namespace nxe::rhi {
namespace {

const nx::log::Category log_up = nx::log::category("upload");

/// Staging ring size. Sized to hold a busy frame of sprite loads plus a couple
/// of frames in flight; larger uploads take a dedicated buffer, so this is a
/// throughput knob, not a hard limit on texture size.
constexpr u64 RING_CAPACITY = 32ull * 1024 * 1024;

[[nodiscard]] u64 align_up(const u64 value, const u64 alignment) noexcept {
  return (value + alignment - 1) & ~(alignment - 1);
}

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
  VmaAllocationCreateInfo ring_alloc_info{};
  ring_alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  ring_alloc_info.flags =
      VMA_ALLOCATION_CREATE_MAPPED_BIT |
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  VmaAllocationInfo ring_info{};
  if (NX_VK_CHECK(vmaCreateBuffer(m_allocator, &buf_info, &ring_alloc_info,
                                  &m_ring, &m_ring_alloc, &ring_info)) !=
      VK_SUCCESS) {
    nx::loge(log_up, "staging ring ({} MiB) allocation failed",
             RING_CAPACITY / (1024 * 1024));
    return false;
  }
  m_ring_mapped = nx::cast<u8 *>(ring_info.pMappedData);
  m_ring_capacity = RING_CAPACITY;
  m_device->set_object_name(vk_handle_value(m_ring), VK_OBJECT_TYPE_BUFFER,
                            "upload ring");

  nx::logi(log_up, "uploader: transfer queue family {} ({}), {} MiB ring",
           m_copy_family,
           needs_ownership_transfer() ? "dedicated" : "shared with graphics",
           RING_CAPACITY / (1024 * 1024));
  return true;
}

void VulkanUploader::shutdown() noexcept {
  if (m_vk == VK_NULL_HANDLE)
    return;

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
  m_buffer_acquires.clear();
  m_vk = VK_NULL_HANDLE;
}

u64 VulkanUploader::completed_value() const noexcept {
  u64 value = 0;
  if (m_timeline != VK_NULL_HANDLE)
    NX_VK_CHECK(vkGetSemaphoreCounterValue(m_vk, m_timeline, &value));
  return value;
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
  NX_VK_CHECK(vkQueueSubmit2(m_queue, 1, &submit, VK_NULL_HANDLE));

  m_inflight.push_back({cmd, value});
}

void VulkanUploader::flush_locked() {
  if (!m_batch_open)
    return;
  vkEndCommandBuffer(m_batch_cmd);
  m_value = m_batch_value;
  submit_signal(m_batch_cmd, m_batch_value);
  m_markers.push_back({m_ring_head, m_batch_value});
  m_batch_open = false;
  m_batch_cmd = VK_NULL_HANDLE;
}

void VulkanUploader::reclaim_locked() {
  const u64 completed = completed_value();

  while (!m_markers.empty() && m_markers.front().value <= completed) {
    m_ring_tail = m_markers.front().head;
    m_markers.erase(m_markers.begin());
  }

  for (Chunk &chunk : m_chunks)
    if (chunk.value != 0 && chunk.value <= completed)
      chunk.value = 0;

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
    u64 base = align_up(m_ring_head, alignment);
    u64 off = base % m_ring_capacity;
    // A request that would straddle the wrap point skips to the next start.
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

    // Out of room: reclaim finished batches, then make the open batch waitable
    // and block on the oldest in flight until the tail can advance.
    reclaim_locked();
    if (try_alloc(offset)) {
      *out_offset = offset;
      return m_ring_mapped + offset;
    }

    if (m_batch_open)
      flush_locked();
    if (m_markers.empty())
      return nullptr; // Fits the ring, yet nothing is in flight: unreachable.

    const u64 oldest = m_markers.front().value;
    VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wait.semaphoreCount = 1;
    wait.pSemaphores = &m_timeline;
    wait.pValues = &oldest;
    NX_VK_CHECK(vkWaitSemaphores(m_vk, &wait, UINT64_MAX));
    reclaim_locked();
  }
}

VulkanUploader::Chunk *VulkanUploader::acquire_chunk(const u64 size,
                                                     const bool host_read) {
  for (Chunk &chunk : m_chunks) {
    if (chunk.value == 0 && chunk.host_read == host_read &&
        chunk.capacity >= size)
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

UploadTicket
VulkanUploader::upload_texture(const TextureHandle handle,
                               const std::span<const u8> src,
                               const std::span<const ImageSubresource> layout,
                               const ResourceState final_state) {
  nx::scoped_lock<nx::mutex> guard(m_mutex);

  VulkanTexture *tex = m_device->m_textures.try_get(handle);
  if (tex == nullptr || tex->image == VK_NULL_HANDLE || layout.empty())
    return {};

  const VkImageAspectFlags aspect = to_vk_aspect(tex->format);
  const u64 align = copy_alignment(tex->format);

  u64 staging_size = 0;
  for (const ImageSubresource &sr : layout)
    staging_size = align_up(staging_size, align) + sr.size;

  VkBuffer staging_buf = VK_NULL_HANDLE;
  u8 *staging = nullptr;
  u64 base_offset = 0;
  Chunk *oversized = nullptr;
  if (staging_size <= m_ring_capacity) {
    staging = ring_alloc(staging_size, align, &base_offset);
    staging_buf = m_ring;
  } else {
    oversized = acquire_chunk(staging_size, false);
    if (oversized != nullptr) {
      staging = oversized->mapped;
      staging_buf = oversized->buffer;
    }
  }
  if (staging == nullptr)
    return {};

  const VkCommandBuffer cmd = ensure_batch();
  if (cmd == VK_NULL_HANDLE)
    return {};

  nx::small_vector<VkBufferImageCopy, 16> copies;
  u64 offset = 0;
  for (const ImageSubresource &sr : layout) {
    offset = align_up(offset, align);
    std::memcpy(staging + offset, src.data() + sr.offset,
                nx::cast<usize>(sr.size));

    VkBufferImageCopy copy{};
    copy.bufferOffset = base_offset + offset;
    copy.imageSubresource = {aspect, sr.mip, sr.layer, 1};
    copy.imageExtent = {sr.width, sr.height, sr.depth == 0 ? 1u : sr.depth};
    copies.push_back(copy);
    offset += sr.size;
  }

  const VkStateInfo from = to_vk_state(tex->state);
  VkImageMemoryBarrier2 pre = whole_image_barrier(tex->image, aspect);
  pre.srcStageMask = from.stage;
  pre.srcAccessMask = VK_ACCESS_2_NONE;
  pre.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  pre.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  pre.oldLayout = from.layout;
  pre.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  VkDependencyInfo pre_dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  pre_dep.imageMemoryBarrierCount = 1;
  pre_dep.pImageMemoryBarriers = &pre;
  vkCmdPipelineBarrier2(cmd, &pre_dep);

  vkCmdCopyBufferToImage(cmd, staging_buf, tex->image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         nx::cast<u32>(copies.size()), copies.data());

  const VkStateInfo fin = to_vk_state(final_state);
  const bool qfot = needs_ownership_transfer();
  VkImageMemoryBarrier2 post = whole_image_barrier(tex->image, aspect);
  post.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  post.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  post.dstStageMask = qfot ? VK_PIPELINE_STAGE_2_NONE : fin.stage;
  post.dstAccessMask = qfot ? VK_ACCESS_2_NONE : fin.access;
  post.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  post.newLayout = fin.layout;
  if (qfot) {
    post.srcQueueFamilyIndex = m_copy_family;
    post.dstQueueFamilyIndex = m_graphics_family;
  }
  VkDependencyInfo post_dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  post_dep.imageMemoryBarrierCount = 1;
  post_dep.pImageMemoryBarriers = &post;
  vkCmdPipelineBarrier2(cmd, &post_dep);

  if (oversized != nullptr)
    oversized->value = m_batch_value;
  if (m_batch_value > m_pending_wait)
    m_pending_wait = m_batch_value;
  tex->state = final_state;
  if (qfot)
    m_image_acquires.push_back({tex->image, aspect,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                final_state});

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
  nx::scoped_lock<nx::mutex> guard(m_mutex);

  VulkanBuffer *buf = m_device->m_buffers.try_get(handle);
  if (buf == nullptr || buf->buffer == VK_NULL_HANDLE || src.empty())
    return {};

  VkBuffer staging_buf = VK_NULL_HANDLE;
  u8 *staging = nullptr;
  u64 base_offset = 0;
  Chunk *oversized = nullptr;
  if (src.size() <= m_ring_capacity) {
    staging = ring_alloc(src.size(), 16, &base_offset);
    staging_buf = m_ring;
  } else {
    oversized = acquire_chunk(src.size(), false);
    if (oversized != nullptr) {
      staging = oversized->mapped;
      staging_buf = oversized->buffer;
    }
  }
  if (staging == nullptr)
    return {};
  std::memcpy(staging, src.data(), src.size());

  const VkCommandBuffer cmd = ensure_batch();
  if (cmd == VK_NULL_HANDLE)
    return {};

  VkBufferCopy region{};
  region.srcOffset = base_offset;
  region.dstOffset = dst_offset;
  region.size = src.size();
  vkCmdCopyBuffer(cmd, staging_buf, buf->buffer, 1, &region);

  const VkStateInfo fin = to_vk_state(final_state);
  const bool qfot = needs_ownership_transfer();
  VkBufferMemoryBarrier2 bar{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
  bar.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  bar.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  bar.dstStageMask = qfot ? VK_PIPELINE_STAGE_2_NONE : fin.stage;
  bar.dstAccessMask = qfot ? VK_ACCESS_2_NONE : fin.access;
  bar.srcQueueFamilyIndex = qfot ? m_copy_family : VK_QUEUE_FAMILY_IGNORED;
  bar.dstQueueFamilyIndex = qfot ? m_graphics_family : VK_QUEUE_FAMILY_IGNORED;
  bar.buffer = buf->buffer;
  bar.offset = dst_offset;
  bar.size = src.size();
  VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dep.bufferMemoryBarrierCount = 1;
  dep.pBufferMemoryBarriers = &bar;
  vkCmdPipelineBarrier2(cmd, &dep);

  if (oversized != nullptr)
    oversized->value = m_batch_value;
  if (m_batch_value > m_pending_wait)
    m_pending_wait = m_batch_value;
  if (qfot)
    m_buffer_acquires.push_back(
        {buf->buffer, dst_offset, src.size(), final_state});

  return {m_batch_value};
}

ReadbackResult VulkanUploader::read_buffer(const BufferHandle handle,
                                           const u64 offset, const u64 size) {
  nx::scoped_lock<nx::mutex> guard(m_mutex);

  VulkanBuffer *buf = m_device->m_buffers.try_get(handle);
  if (buf == nullptr || buf->buffer == VK_NULL_HANDLE || size == 0)
    return {};

  flush_locked(); // Readback is its own submit; get pending uploads out first.

  Chunk *chunk = acquire_chunk(size, true);
  if (chunk == nullptr)
    return {};

  const VkCommandBuffer cmd = begin_cmd();
  if (cmd == VK_NULL_HANDLE)
    return {};

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
  vkCmdPipelineBarrier2(cmd, &dep);

  vkEndCommandBuffer(cmd);
  const u64 value = ++m_value;
  submit_signal(cmd, value);
  chunk->value = value;
  return {{value}, chunk->mapped, size};
}

ReadbackResult VulkanUploader::read_texture(const TextureHandle handle,
                                            const u32 mip, const u32 layer) {
  nx::scoped_lock<nx::mutex> guard(m_mutex);

  VulkanTexture *tex = m_device->m_textures.try_get(handle);
  if (tex == nullptr || tex->image == VK_NULL_HANDLE || mip >= tex->mip_levels)
    return {};

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
  VkImageMemoryBarrier2 to_src = whole_image_barrier(tex->image, aspect);
  to_src.srcStageMask = from.stage;
  to_src.srcAccessMask = from.access;
  to_src.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  to_src.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
  to_src.oldLayout = from.layout;
  to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dep.imageMemoryBarrierCount = 1;
  dep.pImageMemoryBarriers = &to_src;
  vkCmdPipelineBarrier2(cmd, &dep);

  VkBufferImageCopy copy{};
  copy.imageSubresource = {aspect, mip, layer, 1};
  copy.imageExtent = {w, h, 1};
  vkCmdCopyImageToBuffer(cmd, tex->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         chunk->buffer, 1, &copy);
  tex->state = ResourceState::CopySrc;

  VkMemoryBarrier2 host{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  host.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  host.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  host.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
  host.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
  VkDependencyInfo host_dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  host_dep.memoryBarrierCount = 1;
  host_dep.pMemoryBarriers = &host;
  vkCmdPipelineBarrier2(cmd, &host_dep);

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
    nx::scoped_lock<nx::mutex> guard(m_mutex);
    if (ticket.value > m_value)
      flush_locked(); // The ticket names the still-open batch: submit it.
  }
  u64 value = ticket.value;
  VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  wait.semaphoreCount = 1;
  wait.pSemaphores = &m_timeline;
  wait.pValues = &value;
  NX_VK_CHECK(vkWaitSemaphores(m_vk, &wait, UINT64_MAX));
}

void VulkanUploader::flush() noexcept {
  u64 target = 0;
  {
    nx::scoped_lock<nx::mutex> guard(m_mutex);
    flush_locked();
    target = m_value;
  }
  wait({target});
  reclaim();
}

void VulkanUploader::flush_batch() noexcept {
  nx::scoped_lock<nx::mutex> guard(m_mutex);
  flush_locked();
}

UploaderStats VulkanUploader::stats() const noexcept {
  nx::scoped_lock<nx::mutex> guard(m_mutex);
  UploaderStats s;
  s.total_submits = m_value;
  s.completed_submits = completed_value();
  s.ring_used = m_ring_head - m_ring_tail;
  s.ring_capacity = m_ring_capacity;
  s.extra_buffers = m_chunks.size();
  s.inflight_submits = m_inflight.size();
  return s;
}

u64 VulkanUploader::record_acquires(const VkCommandBuffer graphics_cmd) {
  nx::scoped_lock<nx::mutex> guard(m_mutex);
  if (m_pending_wait == 0)
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
    vkCmdPipelineBarrier2(graphics_cmd, &dep);
  }

  const u64 wait = m_pending_wait;
  m_image_acquires.clear();
  m_buffer_acquires.clear();
  m_pending_wait = 0;
  return wait;
}

void VulkanUploader::reclaim() noexcept {
  nx::scoped_lock<nx::mutex> guard(m_mutex);
  reclaim_locked();
}

} // namespace nxe::rhi
