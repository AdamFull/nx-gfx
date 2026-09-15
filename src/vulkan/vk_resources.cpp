
#include "rendering/rhi/vulkan/vk_device.h"

#include "core/foundation/diagnostics/log.h"

namespace nxe::rhi {
namespace {

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
    return NX_VK_DEVICE_CHECK(*this, vkCreateImageView(m_device, &info, nullptr,
                                                       &out)) == VK_SUCCESS;
  };

  if (texture.usage & TextureUsage::Sampled) {
    if (texture.mip_levels == 0)
      return false;
    const u32 base_mip =
        nx::min(texture.sampled_base_mip, texture.mip_levels - 1u);
    if (!make_view(to_vk_view_type(texture.dimension), base_mip,
                   texture.mip_levels - base_mip, texture.srv))
      return false;
    set_object_name(vk_handle_value(texture.srv), VK_OBJECT_TYPE_IMAGE_VIEW,
                    nx::format("{}.sampled", name));
  }

  if (texture.usage & (TextureUsage::RenderTarget | TextureUsage::DepthStencil |
                       TextureUsage::Transient)) {
    if (!make_view(flat_type, 0, 1, texture.rtv))
      return false;
    set_object_name(vk_handle_value(texture.rtv), VK_OBJECT_TYPE_IMAGE_VIEW,
                    nx::format("{}.attachment", name));
  }

  if (texture.usage & TextureUsage::Storage) {
    for (u32 mip = 0; mip < texture.mip_levels; ++mip) {
      VkImageView uav = VK_NULL_HANDLE;
      if (!make_view(flat_type, mip, 1, uav))
        return false;
      texture.uavs.push_back(uav);
      set_object_name(vk_handle_value(uav), VK_OBJECT_TYPE_IMAGE_VIEW,
                      nx::format("{}.storage.mip{}", name, mip));
    }
  }

  return true;
}

namespace {

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
    nx::loge(nx::log::rhi, "create_texture('{}'): invalid format or extent",
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
  if (NX_VK_DEVICE_CHECK(
          *this, vmaCreateImage(m_allocator, &info, &alloc_info, &texture.image,
                                &texture.allocation, nullptr)) != VK_SUCCESS) {
    nx::loge(nx::log::rhi, "create_texture('{}'): allocation failed", desc.name);
    return {};
  }
  {
    const nx::string allocation_name(desc.name);
    vmaSetAllocationName(m_allocator, texture.allocation,
                         allocation_name.c_str());
  }

  texture.format = desc.format;
  texture.dimension = desc.dimension;
  texture.width = desc.width;
  texture.height = desc.height;
  texture.depth = info.extent.depth;
  texture.mip_levels = desc.mip_levels;
  texture.sampled_base_mip =
      desc.mip_levels > 0 ? nx::min(desc.sampled_base_mip, desc.mip_levels - 1u)
                          : 0u;
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

  if (stored.srv != VK_NULL_HANDLE)
    write_texture_descriptor(nx::cast<u32>(handle.index()), stored.srv);
  for (const VkImageView uav : stored.uavs) {
    const u32 slot = alloc_rw_slot();
    if (slot == BINDLESS_INVALID_INDEX) {
      nx::loge(nx::log::rhi, "'{}': out of rw-texture slots, destroying",
               desc.name);
      destroy_texture_now(stored);
      m_textures.erase(handle);
      return {};
    }
    stored.uav_indices.push_back(slot);
    write_rw_texture_descriptor(slot, uav);
  }

  return handle;
}

MemoryRequirements
VulkanDevice::texture_requirements(const TextureDesc &desc) const {
  const VkImageCreateInfo info = to_image_create_info(desc);

  VkImage image = VK_NULL_HANDLE;
  if (NX_VK_DEVICE_CHECK(
          *this, vkCreateImage(m_device, &info, nullptr, &image)) != VK_SUCCESS)
    return {};

  VkMemoryRequirements result{};
  vkGetImageMemoryRequirements(m_device, image, &result);
  vkDestroyImage(m_device, image, nullptr);

  return MemoryRequirements{
      .size = result.size,
      .alignment = result.alignment,
      .memory_type_bits = result.memoryTypeBits,
  };
}

HeapHandle VulkanDevice::create_heap(const HeapDesc &desc) {
  if (desc.size == 0) {
    nx::loge(nx::log::rhi, "create_heap('{}'): zero size", desc.name);
    return {};
  }

  VkMemoryRequirements requirements{};
  requirements.size = desc.size;
  requirements.alignment = desc.alignment < 1 ? 1 : desc.alignment;
  requirements.memoryTypeBits = desc.memory_type_bits;

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
  if (NX_VK_DEVICE_CHECK(*this, vmaAllocateMemory(m_allocator, &requirements,
                                                  &alloc_info, &heap.allocation,
                                                  nullptr)) != VK_SUCCESS) {
    nx::loge(nx::log::rhi, "create_heap('{}'): {} KiB allocation failed", desc.name,
             desc.size / 1024);
    return {};
  }
  {
    const nx::string allocation_name(desc.name);
    vmaSetAllocationName(m_allocator, heap.allocation, allocation_name.c_str());
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
    nx::loge(nx::log::rhi, "create_texture_in_heap('{}'): invalid heap or extent",
             desc.name);
    return {};
  }

  const VkImageCreateInfo info = to_image_create_info(desc);

  VulkanTexture texture;
  if (NX_VK_DEVICE_CHECK(*this, vmaCreateAliasingImage2(
                                    m_allocator, heap->allocation, offset,
                                    &info, &texture.image)) != VK_SUCCESS) {
    nx::loge(nx::log::rhi,
             "create_texture_in_heap('{}'): binding at offset {} failed",
             desc.name, offset);
    return {};
  }

  texture.format = desc.format;
  texture.dimension = desc.dimension;
  texture.width = desc.width;
  texture.height = desc.height;
  texture.depth = info.extent.depth;
  texture.mip_levels = desc.mip_levels;
  texture.sampled_base_mip =
      desc.mip_levels > 0 ? nx::min(desc.sampled_base_mip, desc.mip_levels - 1u)
                          : 0u;
  texture.layers = info.arrayLayers;
  texture.usage = desc.usage;
  texture.state = ResourceState::Undefined;
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
    if (slot == BINDLESS_INVALID_INDEX) {
      nx::loge(nx::log::rhi, "'{}': out of rw-texture slots, destroying",
               desc.name);
      destroy_texture_now(stored);
      m_textures.erase(handle);
      return {};
    }
    stored.uav_indices.push_back(slot);
    write_rw_texture_descriptor(slot, uav);
  }

  return handle;
}

void VulkanDevice::destroy_texture(const TextureHandle handle) noexcept {
  if (!m_textures.contains(handle))
    return;
  m_pending_deletes.push_back({m_timeline_value, handle, {}, {}});
}

bool VulkanDevice::replace_texture(const TextureHandle target,
                                   const TextureHandle replacement) {
  if (!target.valid() || !replacement.valid() || target == replacement ||
      !m_textures.contains(target) || !m_textures.contains(replacement))
    return false;

  VulkanTexture *const held = m_textures.try_get(target);
  VulkanTexture *const fresh = m_textures.try_get(replacement);
  if (held == nullptr || fresh == nullptr || held->usage != fresh->usage ||
      held->usage.has(TextureUsage::Storage))
    return false;

  wait_idle();
  if (device_status() == GpuDeviceStatus::Lost)
    return false;
  std::swap(*held, *fresh);
  if (held->srv != VK_NULL_HANDLE)
    write_texture_descriptor(nx::cast<u32>(target.index()), held->srv);

  // replacement now owns the old target allocation. Its handle was never
  // published, so it is safe to consume through the normal deferred path.
  m_pending_deletes.push_back({m_timeline_value, replacement, {}, {}});
  return true;
}

bool VulkanDevice::set_texture_view_base_mip(const TextureHandle handle,
                                             u32 base_mip) {
  VulkanTexture *const tex = m_textures.try_get(handle);
  if (tex == nullptr || !(tex->usage & TextureUsage::Sampled) ||
      tex->image == VK_NULL_HANDLE)
    return false;
  if (tex->mip_levels == 0)
    return false;
  if (base_mip >= tex->mip_levels)
    base_mip = tex->mip_levels - 1u;
  bool has_pending = false;
  for (auto pending = m_pending_texture_views.rbegin();
       pending != m_pending_texture_views.rend(); ++pending) {
    if (pending->texture != handle)
      continue;
    has_pending = true;
    if (pending->base_mip == base_mip)
      return true;
    break;
  }
  if (!has_pending && tex->sampled_base_mip == base_mip)
    return true;

  VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  info.image = tex->image;
  info.viewType = to_vk_view_type(tex->dimension);
  info.format = to_vk_format(tex->format);
  info.subresourceRange.aspectMask = to_vk_aspect(tex->format);
  info.subresourceRange.baseMipLevel = base_mip;
  info.subresourceRange.levelCount = tex->mip_levels - base_mip;
  info.subresourceRange.layerCount = tex->layers;
  VkImageView view = VK_NULL_HANDLE;
  if (NX_VK_DEVICE_CHECK(*this, vkCreateImageView(m_device, &info, nullptr,
                                                  &view)) != VK_SUCCESS)
    return false;
  set_object_name(
      vk_handle_value(view), VK_OBJECT_TYPE_IMAGE_VIEW,
      nx::format("texture {} sampled mip {}", handle.raw(), base_mip));

  // The bindless set is update-after-bind: an already submitted frame may see
  // an in-place descriptor update. Publishing a view that exposes the mip just
  // queued for upload would therefore let that older frame sample undefined
  // texels. Retain the new view until the graphics frame containing the upload
  // has completed; collect_garbage publishes it at that safe boundary.
  m_pending_texture_views.push_back(
      {handle, view,
       m_timeline_value == ~u64{0} ? ~u64{0} : m_timeline_value + 1u,
       base_mip});
  return true;
}

const VulkanTexture *
VulkanDevice::texture(const TextureHandle handle) const noexcept {
  return m_textures.try_get(handle);
}

TextureUsageFlags
VulkanDevice::texture_usage(const TextureHandle handle) const noexcept {
  const VulkanTexture *texture = m_textures.try_get(handle);
  return texture != nullptr ? texture->usage : TextureUsage::None;
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

BufferHandle VulkanDevice::create_buffer(const BufferDesc &desc) {
  if (desc.size == 0) {
    nx::loge(nx::log::rhi, "create_buffer('{}'): zero size", desc.name);
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
  if (NX_VK_DEVICE_CHECK(*this,
                         vmaCreateBuffer(m_allocator, &info, &alloc_info,
                                         &buffer.buffer, &buffer.allocation,
                                         &allocation_info)) != VK_SUCCESS) {
    nx::loge(nx::log::rhi, "create_buffer('{}'): allocation failed ({} bytes)",
             desc.name, desc.size);
    return {};
  }
  {
    const nx::string allocation_name(desc.name);
    vmaSetAllocationName(m_allocator, buffer.allocation,
                         allocation_name.c_str());
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
  if (NX_VK_DEVICE_CHECK(
          *this, vkCreateSampler(m_device, &info, nullptr, &sampler.sampler)) !=
      VK_SUCCESS) {
    nx::loge(nx::log::rhi, "create_sampler('{}') failed", desc.name);
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
  if (NX_VK_DEVICE_CHECK(*this, vkCreateQueryPool(m_device, &info, nullptr,
                                                  &queries.pool)) != VK_SUCCESS)
    return {};
  set_object_name(vk_handle_value(queries.pool), VK_OBJECT_TYPE_QUERY_POOL,
                  "render graph timestamps");
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
  return NX_VK_DEVICE_CHECK(*this, vkGetQueryPoolResults(
                                       m_device, queries->pool, 0, count,
                                       nx::cast<usize>(count) * sizeof(u64),
                                       out.data(), sizeof(u64),
                                       VK_QUERY_RESULT_64_BIT)) == VK_SUCCESS;
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

void VulkanDevice::flush_buffer(const BufferHandle handle, const u64 offset,
                                const u64 size) noexcept {
  const VulkanBuffer *buffer = m_buffers.try_get(handle);
  if (buffer == nullptr || buffer->allocation == VK_NULL_HANDLE)
    return;
  const u64 available = offset < buffer->size ? buffer->size - offset : 0;
  const u64 bytes = size == ~0ull ? available : nx::min(size, available);
  if (bytes != 0)
    (void)vmaFlushAllocation(m_allocator, buffer->allocation, offset, bytes);
}

void VulkanDevice::invalidate_buffer(const BufferHandle handle,
                                     const u64 offset,
                                     const u64 size) noexcept {
  const VulkanBuffer *buffer = m_buffers.try_get(handle);
  if (buffer == nullptr || buffer->allocation == VK_NULL_HANDLE)
    return;
  const u64 available = offset < buffer->size ? buffer->size - offset : 0;
  const u64 bytes = size == ~0ull ? available : nx::min(size, available);
  if (bytes != 0)
    (void)vmaInvalidateAllocation(m_allocator, buffer->allocation, offset,
                                  bytes);
}

bool VulkanCommandContext::copy_texture_to_buffer(
    const TextureHandle texture_handle, const BufferHandle buffer_handle,
    const u64 row_pitch, const u32 mip, const u32 layer) {
  if (m_device == nullptr || m_cmd == VK_NULL_HANDLE || m_in_render_pass)
    return false;
  const VulkanTexture *texture = m_device->texture(texture_handle);
  const VulkanBuffer *buffer = m_device->buffer(buffer_handle);
  if (texture == nullptr || buffer == nullptr ||
      !(texture->usage & TextureUsage::CopySrc) ||
      !(buffer->usage & BufferUsage::CopyDst) ||
      texture->state != ResourceState::CopySrc ||
      format_is_depth(texture->format) ||
      format_is_compressed(texture->format) || mip >= texture->mip_levels ||
      layer >= texture->layers)
    return false;

  const u32 width = nx::max<u32>(1, texture->width >> mip);
  const u32 height = nx::max<u32>(1, texture->height >> mip);
  const u32 texel = format_texel_size(texture->format);
  const u64 tight = format_row_pitch(texture->format, width);
  if (texel == 0 || row_pitch < tight || row_pitch % texel != 0 ||
      row_pitch / texel > ~u32{0} || row_pitch > buffer->size / height)
    return false;

  VkBufferImageCopy copy{};
  copy.bufferRowLength = nx::cast<u32>(row_pitch / texel);
  copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, layer, 1};
  copy.imageExtent = {width, height, 1};
  vkCmdCopyImageToBuffer(m_cmd, texture->image,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer->buffer,
                         1, &copy);

  VkBufferMemoryBarrier2 host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
  host.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  host.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  host.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
  host.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
  host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  host.buffer = buffer->buffer;
  host.offset = 0;
  host.size = row_pitch * height;
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.bufferMemoryBarrierCount = 1;
  dependency.pBufferMemoryBarriers = &host;
  vkCmdPipelineBarrier2KHR(m_cmd, &dependency);
  return true;
}

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
    nx::loge(nx::log::rhi, "reset_queries inside a render pass");
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
