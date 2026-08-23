/**
 * @file vk_resources.cpp
 * @brief Texture and buffer creation, and command recording.
 */

#include "core/rendering/rhi/vulkan/vk_device.h"

#include "core/foundation/diagnostics/log.h"

namespace nxe::rhi {
namespace {

const nx::log::Category log_rhi = nx::log::category("rhi");

VmaMemoryUsage to_vma_usage(const MemoryUsage usage) noexcept {
  switch (usage) {
  case MemoryUsage::Upload:
  case MemoryUsage::Readback:
    return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  default:
    return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  }
}

VmaAllocationCreateFlags to_vma_flags(const MemoryUsage usage,
                                      const bool persistently_mapped) noexcept {
  VmaAllocationCreateFlags flags = 0;
  if (usage == MemoryUsage::Upload)
    flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  else if (usage == MemoryUsage::Readback)
    flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
  if (persistently_mapped)
    flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
  return flags;
}

} // namespace

// Textures

bool VulkanDevice::create_texture_views(VulkanTexture &texture,
                                        const nx::string_view name) {
  const VkFormat format = to_vk_format(texture.format);
  const VkImageAspectFlags aspect = to_vk_aspect(texture.format);

  // Attachment and storage views cannot use a CUBE type; they address the
  // faces as an array instead.
  const VkImageViewType flat_type =
      texture.dimension == TextureDimension::Tex3D ? VK_IMAGE_VIEW_TYPE_3D
      : texture.layers > 1                         ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                                                   : VK_IMAGE_VIEW_TYPE_2D;

  const auto make_view = [&](const VkImageViewType type, const u32 base_mip,
                             const u32 mip_count, VkImageView &out) -> bool {
    VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    info.image = texture.image;
    info.viewType = type;
    info.format = format;
    info.subresourceRange.aspectMask = aspect;
    info.subresourceRange.baseMipLevel = base_mip;
    info.subresourceRange.levelCount = mip_count;
    info.subresourceRange.layerCount = texture.layers;
    return NX_VK_CHECK(vkCreateImageView(m_device, &info, nullptr, &out)) ==
           VK_SUCCESS;
  };

  if (texture.usage & TextureUsage::Sampled) {
    if (!make_view(to_vk_view_type(texture.dimension), 0, texture.mip_levels,
                   texture.srv))
      return false;
    set_object_name(vk_handle_value(texture.srv), VK_OBJECT_TYPE_IMAGE_VIEW,
                    name);
  }

  if (texture.usage & (TextureUsage::RenderTarget | TextureUsage::DepthStencil |
                       TextureUsage::Transient)) {
    if (!make_view(flat_type, 0, 1, texture.rtv))
      return false;
  }

  if (texture.usage & TextureUsage::Storage) {
    for (u32 mip = 0; mip < texture.mip_levels; ++mip) {
      VkImageView uav = VK_NULL_HANDLE;
      if (!make_view(flat_type, mip, 1, uav))
        return false;
      texture.uavs.push_back(uav);
    }
  }

  return true;
}

namespace {

/// The VkImageCreateInfo a TextureDesc maps to. Shared by the owning and the
/// heap-placed paths, so the two cannot drift - which would make the memory
/// requirements queried for one wrong for the other.
[[nodiscard]] VkImageCreateInfo
to_image_create_info(const TextureDesc &desc) noexcept {
  const bool is_cube = desc.dimension == TextureDimension::TexCube;
  const bool is_3d = desc.dimension == TextureDimension::Tex3D;

  VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  info.imageType = to_vk_image_type(desc.dimension);
  info.format = to_vk_format(desc.format);
  info.extent = {desc.width, desc.height, is_3d ? desc.depth_or_layers : 1};
  info.mipLevels = desc.mip_levels;
  info.arrayLayers = is_3d ? 1 : desc.depth_or_layers;
  info.samples = nx::cast<VkSampleCountFlagBits>(
      desc.sample_count == 0 ? 1 : desc.sample_count);
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = to_vk_image_usage(desc.usage, desc.format);
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (is_cube)
    info.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  return info;
}

} // namespace

TextureHandle VulkanDevice::create_texture(const TextureDesc &desc) {
  if (desc.format == Format::Unknown || desc.width == 0 || desc.height == 0) {
    nx::loge(log_rhi, "create_texture('{}'): invalid format or extent",
             desc.name);
    return {};
  }

  const VkImageCreateInfo info = to_image_create_info(desc);

  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = to_vma_usage(desc.memory);
  alloc_info.flags = to_vma_flags(desc.memory, false);
  if (desc.usage & TextureUsage::Transient)
    alloc_info.preferredFlags = VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;

  VulkanTexture texture;
  if (NX_VK_CHECK(vmaCreateImage(m_allocator, &info, &alloc_info,
                                 &texture.image, &texture.allocation,
                                 nullptr)) != VK_SUCCESS) {
    nx::loge(log_rhi, "create_texture('{}'): allocation failed", desc.name);
    return {};
  }

  texture.format = desc.format;
  texture.dimension = desc.dimension;
  texture.width = desc.width;
  texture.height = desc.height;
  texture.mip_levels = desc.mip_levels;
  texture.layers = info.arrayLayers;
  texture.usage = desc.usage;
  texture.state = ResourceState::Undefined;
  texture.owns_image = true;

  if (!create_texture_views(texture, desc.name)) {
    for (const VkImageView view : texture.uavs)
      vkDestroyImageView(m_device, view, nullptr);
    if (texture.srv != VK_NULL_HANDLE)
      vkDestroyImageView(m_device, texture.srv, nullptr);
    if (texture.rtv != VK_NULL_HANDLE)
      vkDestroyImageView(m_device, texture.rtv, nullptr);
    vmaDestroyImage(m_allocator, texture.image, texture.allocation);
    return {};
  }

  const TextureHandle handle = m_textures.insert(std::move(texture));
  VulkanTexture &stored = *m_textures.try_get(handle);
  set_object_name(vk_handle_value(stored.image), VK_OBJECT_TYPE_IMAGE,
                  desc.name);

  // Register in the bindless heap. The SRV slot is the handle index; each UAV
  // (one per mip) takes a slot from the storage free list.
  if (stored.srv != VK_NULL_HANDLE)
    write_texture_descriptor(nx::cast<u32>(handle.index()), stored.srv);
  for (const VkImageView uav : stored.uavs) {
    const u32 slot = alloc_rw_slot();
    stored.uav_indices.push_back(slot);
    write_rw_texture_descriptor(slot, uav);
  }

  return handle;
}

// Memory aliasing

MemoryRequirements
VulkanDevice::texture_requirements(const TextureDesc &desc) const {
  const VkImageCreateInfo info = to_image_create_info(desc);

  // Vulkan 1.3 answers this from the descriptor alone, so there is no throwaway
  // VkImage to create and destroy just to ask.
  VkDeviceImageMemoryRequirements query{
      VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS};
  query.pCreateInfo = &info;

  VkMemoryRequirements2 result{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
  vkGetDeviceImageMemoryRequirements(m_device, &query, &result);

  return MemoryRequirements{
      .size = result.memoryRequirements.size,
      .alignment = result.memoryRequirements.alignment,
      .memory_type_bits = result.memoryRequirements.memoryTypeBits,
  };
}

HeapHandle VulkanDevice::create_heap(const HeapDesc &desc) {
  if (desc.size == 0) {
    nx::loge(log_rhi, "create_heap('{}'): zero size", desc.name);
    return {};
  }

  // The heap is raw memory, so its requirements are stated rather than derived
  // from a resource. memory_type_bits is the intersection the caller computed
  // across everything it intends to place here.
  VkMemoryRequirements requirements{};
  requirements.size = desc.size;
  requirements.alignment = desc.alignment < 1 ? 1 : desc.alignment;
  requirements.memoryTypeBits = desc.memory_type_bits;

  // VMA_MEMORY_USAGE_AUTO* is rejected here: it infers the memory type from the
  // resource being created, and a raw allocation has none. State the properties
  // wanted instead.
  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = VMA_MEMORY_USAGE_UNKNOWN;
  alloc_info.flags = to_vma_flags(desc.memory, false);
  switch (desc.memory) {
  case MemoryUsage::Upload:
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    break;
  case MemoryUsage::Readback:
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    break;
  default:
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    break;
  }

  VulkanHeap heap;
  heap.size = desc.size;
  if (NX_VK_CHECK(vmaAllocateMemory(m_allocator, &requirements, &alloc_info,
                                    &heap.allocation, nullptr)) != VK_SUCCESS) {
    nx::loge(log_rhi, "create_heap('{}'): {} KiB allocation failed", desc.name,
             desc.size / 1024);
    return {};
  }
  return m_heaps.insert(heap);
}

void VulkanDevice::destroy_heap(const HeapHandle handle) noexcept {
  VulkanHeap *heap = m_heaps.try_get(handle);
  if (heap == nullptr)
    return;
  // Every image placed here must already be gone; the graph tears them down
  // together.
  if (heap->allocation != VK_NULL_HANDLE)
    vmaFreeMemory(m_allocator, heap->allocation);
  m_heaps.erase(handle);
}

TextureHandle VulkanDevice::create_texture_in_heap(const TextureDesc &desc,
                                                   const HeapHandle heap_handle,
                                                   const u64 offset) {
  const VulkanHeap *heap = m_heaps.try_get(heap_handle);
  if (heap == nullptr || desc.format == Format::Unknown || desc.width == 0 ||
      desc.height == 0) {
    nx::loge(log_rhi, "create_texture_in_heap('{}'): invalid heap or extent",
             desc.name);
    return {};
  }

  const VkImageCreateInfo info = to_image_create_info(desc);

  VulkanTexture texture;
  if (NX_VK_CHECK(vmaCreateAliasingImage2(m_allocator, heap->allocation, offset,
                                          &info, &texture.image)) !=
      VK_SUCCESS) {
    nx::loge(log_rhi,
             "create_texture_in_heap('{}'): binding at offset {} failed",
             desc.name, offset);
    return {};
  }

  texture.format = desc.format;
  texture.dimension = desc.dimension;
  texture.width = desc.width;
  texture.height = desc.height;
  texture.mip_levels = desc.mip_levels;
  texture.layers = info.arrayLayers;
  texture.usage = desc.usage;
  texture.state = ResourceState::Undefined;
  // Owns the VkImage but not the memory: allocation stays null so teardown
  // destroys the image and leaves the heap alone.
  texture.owns_image = true;
  texture.allocation = VK_NULL_HANDLE;

  if (!create_texture_views(texture, desc.name)) {
    for (const VkImageView view : texture.uavs)
      vkDestroyImageView(m_device, view, nullptr);
    if (texture.srv != VK_NULL_HANDLE)
      vkDestroyImageView(m_device, texture.srv, nullptr);
    if (texture.rtv != VK_NULL_HANDLE)
      vkDestroyImageView(m_device, texture.rtv, nullptr);
    vkDestroyImage(m_device, texture.image, nullptr);
    return {};
  }

  const TextureHandle handle = m_textures.insert(std::move(texture));
  VulkanTexture &stored = *m_textures.try_get(handle);
  set_object_name(vk_handle_value(stored.image), VK_OBJECT_TYPE_IMAGE,
                  desc.name);

  if (stored.srv != VK_NULL_HANDLE)
    write_texture_descriptor(nx::cast<u32>(handle.index()), stored.srv);
  for (const VkImageView uav : stored.uavs) {
    const u32 slot = alloc_rw_slot();
    stored.uav_indices.push_back(slot);
    write_rw_texture_descriptor(slot, uav);
  }

  return handle;
}

void VulkanDevice::destroy_texture(const TextureHandle handle) noexcept {
  if (!m_textures.contains(handle))
    return;
  // The slot_map entry - and its descriptor slots - stay live until the GPU
  // has passed this frame; collect_garbage does the actual teardown.
  m_pending_deletes.push_back({m_timeline_value, handle, {}, {}});
}

const VulkanTexture *
VulkanDevice::texture(const TextureHandle handle) const noexcept {
  return m_textures.try_get(handle);
}

VulkanTexture *VulkanDevice::texture_mut(const TextureHandle handle) noexcept {
  return m_textures.try_get(handle);
}

ResourceState
VulkanDevice::texture_state(const TextureHandle handle) const noexcept {
  const VulkanTexture *texture = m_textures.try_get(handle);
  return texture != nullptr ? texture->state : ResourceState::Undefined;
}

void VulkanDevice::set_texture_state(const TextureHandle handle,
                                     const ResourceState state) noexcept {
  if (VulkanTexture *texture = m_textures.try_get(handle))
    texture->state = state;
}

// Buffers

BufferHandle VulkanDevice::create_buffer(const BufferDesc &desc) {
  if (desc.size == 0) {
    nx::loge(log_rhi, "create_buffer('{}'): zero size", desc.name);
    return {};
  }

  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = desc.size;
  info.usage = to_vk_buffer_usage(desc.usage);
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = to_vma_usage(desc.memory);
  alloc_info.flags = to_vma_flags(desc.memory, desc.persistently_mapped);

  VulkanBuffer buffer;
  VmaAllocationInfo allocation_info{};
  if (NX_VK_CHECK(vmaCreateBuffer(m_allocator, &info, &alloc_info,
                                  &buffer.buffer, &buffer.allocation,
                                  &allocation_info)) != VK_SUCCESS) {
    nx::loge(log_rhi, "create_buffer('{}'): allocation failed ({} bytes)",
             desc.name, desc.size);
    return {};
  }

  buffer.size = desc.size;
  buffer.usage = desc.usage;
  buffer.mapped =
      desc.persistently_mapped ? allocation_info.pMappedData : nullptr;

  if (desc.usage & BufferUsage::DeviceAddress) {
    VkBufferDeviceAddressInfo address_info{
        VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    address_info.buffer = buffer.buffer;
    buffer.device_address =
        vkGetBufferDeviceAddressKHR(m_device, &address_info);
  }

  const BufferHandle handle = m_buffers.insert(buffer);
  set_object_name(vk_handle_value(m_buffers.try_get(handle)->buffer),
                  VK_OBJECT_TYPE_BUFFER, desc.name);
  return handle;
}

void VulkanDevice::destroy_buffer(const BufferHandle handle) noexcept {
  if (!m_buffers.contains(handle))
    return;
  m_pending_deletes.push_back({m_timeline_value, {}, handle, {}});
}

const VulkanBuffer *
VulkanDevice::buffer(const BufferHandle handle) const noexcept {
  return m_buffers.try_get(handle);
}

// Samplers

SamplerHandle VulkanDevice::create_sampler(const SamplerDesc &desc) {
  VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  info.magFilter = to_vk_filter(desc.mag_filter);
  info.minFilter = to_vk_filter(desc.min_filter);
  info.mipmapMode = to_vk_mip_mode(desc.mip_filter);
  info.addressModeU = to_vk_address_mode(desc.address_u);
  info.addressModeV = to_vk_address_mode(desc.address_v);
  info.addressModeW = to_vk_address_mode(desc.address_w);
  info.mipLodBias = desc.mip_lod_bias;
  info.anisotropyEnable = desc.max_anisotropy > 1.f ? VK_TRUE : VK_FALSE;
  info.maxAnisotropy = desc.max_anisotropy;
  info.compareEnable = desc.compare != CompareOp::Never ? VK_TRUE : VK_FALSE;
  info.compareOp = to_vk_compare_op(desc.compare);
  info.minLod = desc.min_lod;
  info.maxLod = desc.max_lod;
  info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;

  VulkanSampler sampler;
  if (NX_VK_CHECK(vkCreateSampler(m_device, &info, nullptr,
                                  &sampler.sampler)) != VK_SUCCESS) {
    nx::loge(log_rhi, "create_sampler('{}') failed", desc.name);
    return {};
  }

  const SamplerHandle handle = m_samplers.insert(sampler);
  set_object_name(vk_handle_value(sampler.sampler), VK_OBJECT_TYPE_SAMPLER,
                  desc.name);
  write_sampler_descriptor(nx::cast<u32>(handle.index()), sampler.sampler);
  return handle;
}

void VulkanDevice::destroy_sampler(const SamplerHandle handle) noexcept {
  if (!m_samplers.contains(handle))
    return;
  m_pending_deletes.push_back({m_timeline_value, {}, {}, handle});
}

QueryPoolHandle VulkanDevice::create_timestamp_pool(const u32 count) {
  if (!m_caps.timestamps || count == 0)
    return {};

  VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  info.queryCount = count;

  VulkanQueryPool queries;
  queries.count = count;
  if (NX_VK_CHECK(vkCreateQueryPool(m_device, &info, nullptr, &queries.pool)) !=
      VK_SUCCESS)
    return {};
  return m_query_pools.insert(queries);
}

void VulkanDevice::destroy_query_pool(const QueryPoolHandle handle) noexcept {
  VulkanQueryPool *const queries = m_query_pools.try_get(handle);
  if (queries == nullptr)
    return;
  if (queries->pool != VK_NULL_HANDLE)
    vkDestroyQueryPool(m_device, queries->pool, nullptr);
  m_query_pools.erase(handle);
}

bool VulkanDevice::read_timestamps(const QueryPoolHandle handle,
                                   const std::span<u64> out) const {
  const VulkanQueryPool *const queries = m_query_pools.try_get(handle);
  if (queries == nullptr || out.empty())
    return false;
  const u32 count = nx::min(queries->count, nx::cast<u32>(out.size()));

  // No WAIT bit: a pool the GPU has not finished with answers NOT_READY, and
  // the caller reads an older one rather than stalling the frame to time it.
  return vkGetQueryPoolResults(m_device, queries->pool, 0, count,
                               nx::cast<usize>(count) * sizeof(u64), out.data(),
                               sizeof(u64),
                               VK_QUERY_RESULT_64_BIT) == VK_SUCCESS;
}

u32 VulkanDevice::texture_index(const TextureHandle handle) const noexcept {
  const VulkanTexture *texture = m_textures.try_get(handle);
  if (texture == nullptr || texture->srv == VK_NULL_HANDLE)
    return BINDLESS_INVALID_INDEX;
  return nx::cast<u32>(handle.index());
}

u32 VulkanDevice::rw_texture_index(const TextureHandle handle,
                                   const u32 mip) const noexcept {
  const VulkanTexture *texture = m_textures.try_get(handle);
  if (texture == nullptr || mip >= texture->uav_indices.size())
    return BINDLESS_INVALID_INDEX;
  return texture->uav_indices[mip];
}

u32 VulkanDevice::sampler_index(const SamplerHandle handle) const noexcept {
  return m_samplers.contains(handle) ? nx::cast<u32>(handle.index())
                                     : BINDLESS_INVALID_INDEX;
}

VkDeviceAddress
VulkanDevice::buffer_address(const BufferHandle handle) const noexcept {
  const VulkanBuffer *buffer = m_buffers.try_get(handle);
  return buffer != nullptr ? buffer->device_address : 0;
}

void *VulkanDevice::buffer_mapped(const BufferHandle handle) const noexcept {
  const VulkanBuffer *buffer = m_buffers.try_get(handle);
  return buffer != nullptr ? buffer->mapped : nullptr;
}

// Command recording

void VulkanCommandContext::barrier(const TextureBarrier *textures,
                                   const u32 texture_count,
                                   const BufferBarrier *buffers,
                                   const u32 buffer_count,
                                   const StateTracking tracking) {
  if (m_device == nullptr || m_cmd == VK_NULL_HANDLE)
    return;

  nx::small_vector<VkImageMemoryBarrier2, 8> image_barriers;
  nx::small_vector<VkBufferMemoryBarrier2, 8> buffer_barriers;

  for (u32 i = 0; i < texture_count; ++i) {
    const TextureBarrier &desc = textures[i];
    VulkanTexture *texture = m_device->texture_mut(desc.texture);
    if (texture == nullptr)
      continue;

    // The tracked state is authoritative: callers may state `from`, but the
    // texture knows where it actually is, which avoids a whole class of
    // mismatched-layout validation errors.
    const ResourceState from = desc.discard ? ResourceState::Undefined
                               : desc.from == ResourceState::Undefined
                                   ? texture->state
                                   : desc.from;
    if (!desc.discard && from == desc.to)
      continue;

    VkStateInfo src = to_vk_state(from);
    const VkStateInfo dst = to_vk_state(desc.to);
    if (desc.discard) {
      src.stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
      src.access = VK_ACCESS_2_MEMORY_WRITE_BIT;
    }

    VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.srcStageMask = src.stage;
    barrier.srcAccessMask = src.access;
    barrier.dstStageMask = dst.stage;
    barrier.dstAccessMask = dst.access;
    barrier.oldLayout = src.layout;
    barrier.newLayout = dst.layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = texture->image;
    barrier.subresourceRange.aspectMask = to_vk_aspect(texture->format);
    barrier.subresourceRange.baseMipLevel = desc.base_mip;
    barrier.subresourceRange.levelCount =
        desc.mip_count == ~0u ? VK_REMAINING_MIP_LEVELS : desc.mip_count;
    barrier.subresourceRange.baseArrayLayer = desc.base_layer;
    barrier.subresourceRange.layerCount =
        desc.layer_count == ~0u ? VK_REMAINING_ARRAY_LAYERS : desc.layer_count;

    image_barriers.push_back(barrier);
    if (tracking == StateTracking::Update)
      texture->state = desc.to;
  }

  for (u32 i = 0; i < buffer_count; ++i) {
    const BufferBarrier &desc = buffers[i];
    const VulkanBuffer *buffer = m_device->buffer(desc.buffer);
    if (buffer == nullptr)
      continue;

    const VkStateInfo src = to_vk_state(desc.from);
    const VkStateInfo dst = to_vk_state(desc.to);

    VkBufferMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    barrier.srcStageMask = src.stage;
    barrier.srcAccessMask = src.access;
    barrier.dstStageMask = dst.stage;
    barrier.dstAccessMask = dst.access;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer->buffer;
    barrier.offset = desc.offset;
    barrier.size = desc.size == ~0ull ? VK_WHOLE_SIZE : desc.size;
    buffer_barriers.push_back(barrier);
  }

  if (image_barriers.empty() && buffer_barriers.empty())
    return;

  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.imageMemoryBarrierCount = nx::cast<u32>(image_barriers.size());
  dependency.pImageMemoryBarriers = image_barriers.data();
  dependency.bufferMemoryBarrierCount = nx::cast<u32>(buffer_barriers.size());
  dependency.pBufferMemoryBarriers = buffer_barriers.data();
  vkCmdPipelineBarrier2KHR(m_cmd, &dependency);
}

void VulkanCommandContext::begin_render_pass(const RenderPassDesc &desc) {
  if (m_device == nullptr || m_cmd == VK_NULL_HANDLE || m_in_render_pass)
    return;

  push_debug_group(desc.name);

  nx::fixed_vector<VkRenderingAttachmentInfo, MAX_COLOR_ATTACHMENTS> colors;
  u32 width = desc.render_area.extent.width;
  u32 height = desc.render_area.extent.height;

  for (u32 i = 0; i < desc.color_count && i < MAX_COLOR_ATTACHMENTS; ++i) {
    const RenderAttachment &attachment = desc.color[i];
    const VulkanTexture *texture = m_device->texture(attachment.texture);
    if (texture == nullptr)
      continue;

    // Zero render area means "the whole first attachment".
    if (width == 0 || height == 0) {
      width = texture->width;
      height = texture->height;
    }

    VkRenderingAttachmentInfo info{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    info.imageView = texture->rtv;
    info.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    info.loadOp = to_vk_load_op(attachment.load);
    info.storeOp = to_vk_store_op(attachment.store);
    std::memcpy(info.clearValue.color.float32, attachment.clear.color,
                sizeof(f32) * 4);

    if (const VulkanTexture *resolve =
            m_device->texture(attachment.resolve_target);
        resolve != nullptr && resolve->rtv != VK_NULL_HANDLE) {
      info.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
      info.resolveImageView = resolve->rtv;
      info.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    colors.push_back(info);
  }

  VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  bool has_depth = false;
  if (desc.has_depth_stencil) {
    if (const VulkanTexture *texture =
            m_device->texture(desc.depth_stencil.texture)) {
      depth.imageView = texture->rtv;
      depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      depth.loadOp = to_vk_load_op(desc.depth_stencil.load);
      depth.storeOp = to_vk_store_op(desc.depth_stencil.store);
      depth.clearValue.depthStencil.depth =
          desc.depth_stencil.clear.depth_stencil.depth;
      depth.clearValue.depthStencil.stencil =
          desc.depth_stencil.clear.depth_stencil.stencil;
      has_depth = true;
      if (width == 0 || height == 0) {
        width = texture->width;
        height = texture->height;
      }
    }
  }

  VkRenderingInfo info{VK_STRUCTURE_TYPE_RENDERING_INFO};
  info.renderArea.offset = {desc.render_area.offset.x,
                            desc.render_area.offset.y};
  info.renderArea.extent = {width, height};
  info.layerCount = 1;
  info.colorAttachmentCount = nx::cast<u32>(colors.size());
  info.pColorAttachments = colors.data();
  info.pDepthAttachment = has_depth ? &depth : nullptr;

  vkCmdBeginRenderingKHR(m_cmd, &info);
  m_in_render_pass = true;

  // Dynamic rendering leaves viewport and scissor unset; defaulting them to
  // the render area is what every caller wants and forgetting it draws
  // nothing, silently.
  const Viewport viewport{0.f, 0.f, nx::cast<f32>(width), nx::cast<f32>(height),
                          0.f, 1.f};
  set_viewport(viewport);
  set_scissor({{0, 0}, {width, height}});
}

void VulkanCommandContext::end_render_pass() {
  if (!m_in_render_pass)
    return;
  vkCmdEndRenderingKHR(m_cmd);
  m_in_render_pass = false;
  pop_debug_group();
}

void VulkanCommandContext::set_viewport(const Viewport &viewport) {
  if (m_cmd == VK_NULL_HANDLE)
    return;
  // Negative height flips to a Y-down clip space, matching every 2D convention
  // (and Metal/D3D) instead of Vulkan's Y-up default.
  VkViewport vk{};
  vk.x = viewport.x;
  vk.y = viewport.y + viewport.height;
  vk.width = viewport.width;
  vk.height = -viewport.height;
  vk.minDepth = viewport.min_depth;
  vk.maxDepth = viewport.max_depth;
  vkCmdSetViewport(m_cmd, 0, 1, &vk);
}

void VulkanCommandContext::set_scissor(const Rect2D &rect) {
  if (m_cmd == VK_NULL_HANDLE)
    return;
  const VkRect2D vk{{rect.offset.x, rect.offset.y},
                    {rect.extent.width, rect.extent.height}};
  vkCmdSetScissor(m_cmd, 0, 1, &vk);
}

void VulkanCommandContext::reset_queries(const QueryPoolHandle pool,
                                         const u32 first, const u32 count) {
  if (m_cmd == VK_NULL_HANDLE || m_device == nullptr || count == 0)
    return;
  const VulkanQueryPool *const queries = m_device->query_pool(pool);
  if (queries == nullptr)
    return;
  if (m_in_render_pass) {
    nx::loge(log_rhi, "reset_queries inside a render pass");
    return;
  }
  vkCmdResetQueryPool(m_cmd, queries->pool, first,
                      nx::min(count, queries->count));
}

void VulkanCommandContext::write_timestamp(const QueryPoolHandle pool,
                                           const u32 index) {
  if (m_cmd == VK_NULL_HANDLE || m_device == nullptr)
    return;
  const VulkanQueryPool *const queries = m_device->query_pool(pool);
  if (queries == nullptr || index >= queries->count)
    return;
  // BOTTOM_OF_PIPE: the stamp lands once everything recorded before it has
  // finished, which is what makes the gap between two of them a pass's time
  // rather than the time it took to record.
  vkCmdWriteTimestamp2KHR(m_cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                          queries->pool, index);
}

void VulkanCommandContext::push_debug_group(const nx::string_view name,
                                            const f32 r, const f32 g,
                                            const f32 b) {
  if (m_device == nullptr || !m_device->debug_utils_enabled() ||
      m_cmd == VK_NULL_HANDLE)
    return;

  const nx::string terminated(name);
  VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
  label.pLabelName = terminated.c_str();
  label.color[0] = r;
  label.color[1] = g;
  label.color[2] = b;
  label.color[3] = 1.f;
  vkCmdBeginDebugUtilsLabelEXT(m_cmd, &label);
}

void VulkanCommandContext::pop_debug_group() {
  if (m_device == nullptr || !m_device->debug_utils_enabled() ||
      m_cmd == VK_NULL_HANDLE)
    return;
  vkCmdEndDebugUtilsLabelEXT(m_cmd);
}

} // namespace nxe::rhi
