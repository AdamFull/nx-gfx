#include "src/vulkan/vk_device.h"

#include "core/foundation/diagnostics/log.h"

#include <cstring>

namespace nx::gfx::detail {
namespace {

u32 clamp_count(const u32 desired, const u32 device_limit) noexcept {
  if (device_limit == 0)
    return desired;
  return desired < device_limit ? desired : device_limit;
}

[[nodiscard]] VkDeviceSize align_up(const VkDeviceSize value,
                                    const VkDeviceSize alignment) noexcept {
  return alignment <= 1 ? value
                        : (value + alignment - 1) / alignment * alignment;
}

constexpr VkDeviceSize GLOBALS_BYTES = 16;
constexpr VkDeviceSize NULL_PAGE_OFFSET = 256;

} // namespace

bool VulkanDevice::create_default_sampler() {
  const gfx::sampler_handle sampler = create_sampler(DEFAULT_SAMPLER_DESC);
  if (sampler_index(sampler) == DEFAULT_SAMPLER)
    return true;
  nx::loge(nx::log::rhi, "the default sampler did not take slot {}",
           DEFAULT_SAMPLER);
  return false;
}

bool VulkanDevice::create_bindless_heap() {
  m_bindless.max_samplers =
      clamp_count(MAX_BINDLESS_SAMPLERS, m_caps.max_bindless_samplers);
  m_bindless.max_textures =
      clamp_count(MAX_BINDLESS_TEXTURES, m_caps.max_bindless_sampled_images);
  m_bindless.max_rw_textures =
      clamp_count(MAX_BINDLESS_RW_TEXTURES, m_caps.max_bindless_storage_images);

  if (m_routes.descriptor_heap)
    return create_descriptor_heaps() && create_bindless_globals();

  const VkDescriptorBindingFlags binding_flags =
      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
      VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
  const VkDescriptorBindingFlags flags[4] = {binding_flags, binding_flags,
                                             binding_flags, 0};

  const VkDescriptorSetLayoutBinding bindings[4] = {
      {BINDLESS_BINDING_SAMPLERS, VK_DESCRIPTOR_TYPE_SAMPLER,
       m_bindless.max_samplers, VK_SHADER_STAGE_ALL, nullptr},
      {BINDLESS_BINDING_TEXTURES, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
       m_bindless.max_textures, VK_SHADER_STAGE_ALL, nullptr},
      {BINDLESS_BINDING_RW_TEXTURES, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
       m_bindless.max_rw_textures, VK_SHADER_STAGE_ALL, nullptr},
      {BINDLESS_BINDING_GLOBALS, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
       VK_SHADER_STAGE_ALL, nullptr},
  };

  VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
  flags_info.bindingCount = 4;
  flags_info.pBindingFlags = flags;

  VkDescriptorSetLayoutCreateInfo layout_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layout_info.pNext = &flags_info;
  layout_info.flags =
      VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
  layout_info.bindingCount = 4;
  layout_info.pBindings = bindings;
  NX_VK_DEVICE_TRY(*this,
                   vkCreateDescriptorSetLayout(m_device, &layout_info, nullptr,
                                               &m_bindless.layout));
  set_object_name(vk_handle_value(m_bindless.layout),
                  VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                  "bindless descriptor layout");

  const VkDescriptorPoolSize pool_sizes[4] = {
      {VK_DESCRIPTOR_TYPE_SAMPLER, m_bindless.max_samplers},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, m_bindless.max_textures},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, m_bindless.max_rw_textures},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
  };

  VkDescriptorPoolCreateInfo pool_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
  pool_info.maxSets = 1;
  pool_info.poolSizeCount = 4;
  pool_info.pPoolSizes = pool_sizes;
  NX_VK_DEVICE_TRY(*this, vkCreateDescriptorPool(m_device, &pool_info, nullptr,
                                                 &m_bindless.pool));
  set_object_name(vk_handle_value(m_bindless.pool),
                  VK_OBJECT_TYPE_DESCRIPTOR_POOL, "bindless descriptor pool");

  VkDescriptorSetAllocateInfo alloc{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  alloc.descriptorPool = m_bindless.pool;
  alloc.descriptorSetCount = 1;
  alloc.pSetLayouts = &m_bindless.layout;
  NX_VK_DEVICE_TRY(*this,
                   vkAllocateDescriptorSets(m_device, &alloc, &m_bindless.set));
  set_object_name(vk_handle_value(m_bindless.set),
                  VK_OBJECT_TYPE_DESCRIPTOR_SET, "bindless set");
  if (!create_bindless_globals())
    return false;

  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_ALL;
  push_range.offset = 0;
  push_range.size = PUSH_CONSTANT_SIZE;

  VkPipelineLayoutCreateInfo pipeline_layout{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipeline_layout.setLayoutCount = 1;
  pipeline_layout.pSetLayouts = &m_bindless.layout;
  pipeline_layout.pushConstantRangeCount = 1;
  pipeline_layout.pPushConstantRanges = &push_range;
  NX_VK_DEVICE_TRY(*this, vkCreatePipelineLayout(m_device, &pipeline_layout,
                                                 nullptr, &m_pipeline_layout));
  set_object_name(vk_handle_value(m_pipeline_layout),
                  VK_OBJECT_TYPE_PIPELINE_LAYOUT, "bindless pipeline layout");

  nx::logd(nx::log::rhi,
           "bindless set: {} samplers, {} textures, {} rw-textures",
           m_bindless.max_samplers, m_bindless.max_textures,
           m_bindless.max_rw_textures);
  return true;
}

bool VulkanDevice::create_descriptor_heap_buffer(DescriptorHeapBuffer &heap,
                                                 const VkDeviceSize alignment,
                                                 const nx::string_view name) {
  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = heap.size;
  info.usage = VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT |
               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo alloc{};
  alloc.usage = VMA_MEMORY_USAGE_AUTO;
  alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
  VmaAllocationInfo allocation{};
  NX_VK_DEVICE_TRY(*this, vmaCreateBufferWithAlignment(
                              m_allocator, &info, &alloc, alignment,
                              &heap.buffer, &heap.allocation, &allocation));
  const nx::string label(name);
  vmaSetAllocationName(m_allocator, heap.allocation, label.c_str());
  set_object_name(vk_handle_value(heap.buffer), VK_OBJECT_TYPE_BUFFER, name);

  VkBufferDeviceAddressInfo address_info{
      VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  address_info.buffer = heap.buffer;
  heap.address = vkGetBufferDeviceAddressKHR(m_device, &address_info);
  heap.data = static_cast<u8 *>(allocation.pMappedData);
  return heap.data != nullptr;
}

bool VulkanDevice::create_descriptor_heaps() {
  VkPhysicalDeviceDescriptorHeapPropertiesEXT props{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT};
  VkPhysicalDeviceProperties2 props2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  props2.pNext = &props;
  vkGetPhysicalDeviceProperties2(m_physical_device, &props2);

  const VkDeviceSize image_size =
      nx::max(vkGetPhysicalDeviceDescriptorSizeEXT(
                  m_physical_device, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
              vkGetPhysicalDeviceDescriptorSizeEXT(
                  m_physical_device, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE));
  m_bindless.image_stride =
      align_up(nx::max(image_size, props.imageDescriptorSize),
               props.imageDescriptorAlignment);
  m_bindless.sampler_stride =
      align_up(props.samplerDescriptorSize, props.samplerDescriptorAlignment);
  const VkDeviceSize buffer_size =
      nx::max(vkGetPhysicalDeviceDescriptorSizeEXT(
                  m_physical_device, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),
              props.bufferDescriptorSize);

  // Textures, storage textures and the globals uniform's descriptor, then the
  // range the implementation keeps for itself.
  DescriptorHeapBuffer &resources = m_bindless.resources;
  m_bindless.rw_offset =
      align_up(m_bindless.max_textures * m_bindless.image_stride,
               props.imageDescriptorAlignment);
  m_bindless.globals_offset =
      align_up(m_bindless.rw_offset +
                   m_bindless.max_rw_textures * m_bindless.image_stride,
               props.bufferDescriptorAlignment);
  resources.reserved_offset = align_up(
      m_bindless.globals_offset + buffer_size,
      nx::max(props.imageDescriptorAlignment, props.bufferDescriptorAlignment));
  resources.reserved_size = props.minResourceHeapReservedRange;
  resources.size = resources.reserved_offset + resources.reserved_size;

  DescriptorHeapBuffer &samplers = m_bindless.samplers;
  samplers.reserved_offset =
      align_up(m_bindless.max_samplers * m_bindless.sampler_stride,
               props.samplerDescriptorAlignment);
  samplers.reserved_size = props.minSamplerHeapReservedRange;
  samplers.size = samplers.reserved_offset + samplers.reserved_size;

  if (resources.size > props.maxResourceHeapSize ||
      samplers.size > props.maxSamplerHeapSize) {
    nx::loge(
        nx::log::rhi,
        "descriptor heaps of {} and {} bytes exceed the device's {} and {}",
        resources.size, samplers.size, props.maxResourceHeapSize,
        props.maxSamplerHeapSize);
    return false;
  }
  if (!create_descriptor_heap_buffer(resources, props.resourceHeapAlignment,
                                     "bindless resource heap") ||
      !create_descriptor_heap_buffer(samplers, props.samplerHeapAlignment,
                                     "bindless sampler heap"))
    return false;

  const auto map = [](const u32 binding, const VkSpirvResourceTypeFlagsEXT mask,
                      const VkDeviceSize offset, const VkDeviceSize stride) {
    VkDescriptorSetAndBindingMappingEXT mapping{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT};
    mapping.descriptorSet = BINDLESS_SET;
    mapping.firstBinding = binding;
    mapping.bindingCount = 1;
    mapping.resourceMask = mask;
    mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    mapping.sourceData.constantOffset.heapOffset = nx::cast<u32>(offset);
    mapping.sourceData.constantOffset.heapArrayStride = nx::cast<u32>(stride);
    return mapping;
  };
  m_bindless.mappings[0] =
      map(BINDLESS_BINDING_SAMPLERS, VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT, 0,
          m_bindless.sampler_stride);
  m_bindless.mappings[1] =
      map(BINDLESS_BINDING_TEXTURES,
          VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT |
              VK_SPIRV_RESOURCE_TYPE_READ_ONLY_IMAGE_BIT_EXT,
          0, m_bindless.image_stride);
  m_bindless.mappings[2] = map(BINDLESS_BINDING_RW_TEXTURES,
                               VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT,
                               m_bindless.rw_offset, m_bindless.image_stride);
  m_bindless.mappings[3] = map(BINDLESS_BINDING_GLOBALS,
                               VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT,
                               m_bindless.globals_offset, 0);

  nx::logd(nx::log::rhi,
           "descriptor heaps: {} samplers, {} textures, {} rw-textures in {} + "
           "{} bytes",
           m_bindless.max_samplers, m_bindless.max_textures,
           m_bindless.max_rw_textures, resources.size, samplers.size);
  return true;
}

VkShaderDescriptorSetAndBindingMappingInfoEXT
VulkanDevice::heap_mappings() const noexcept {
  VkShaderDescriptorSetAndBindingMappingInfoEXT info{
      VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT};
  info.mappingCount = nx::cast<u32>(std::size(m_bindless.mappings));
  info.pMappings = m_bindless.mappings;
  return info;
}

void VulkanDevice::bind_bindless(const VkCommandBuffer cmd) const noexcept {
  if (!m_routes.descriptor_heap) {
    for (const VkPipelineBindPoint point :
         {VK_PIPELINE_BIND_POINT_GRAPHICS, VK_PIPELINE_BIND_POINT_COMPUTE})
      vkCmdBindDescriptorSets(cmd, point, m_pipeline_layout, BINDLESS_SET, 1,
                              &m_bindless.set, 0, nullptr);
    return;
  }
  const auto bind_info = [](const DescriptorHeapBuffer &heap) {
    VkBindHeapInfoEXT info{VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT};
    info.heapRange = {heap.address, heap.size};
    info.reservedRangeOffset = heap.reserved_offset;
    info.reservedRangeSize = heap.reserved_size;
    return info;
  };
  const VkBindHeapInfoEXT resources = bind_info(m_bindless.resources);
  const VkBindHeapInfoEXT samplers = bind_info(m_bindless.samplers);
  vkCmdBindResourceHeapEXT(cmd, &resources);
  vkCmdBindSamplerHeapEXT(cmd, &samplers);
}

bool VulkanDevice::create_bindless_globals() {
  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = NULL_PAGE_OFFSET + NULL_PAGE_BYTES;
  info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo alloc{};
  alloc.usage = VMA_MEMORY_USAGE_AUTO;
  alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
  VmaAllocationInfo allocation{};
  NX_VK_DEVICE_TRY(
      *this, vmaCreateBuffer(m_allocator, &info, &alloc, &m_bindless.globals,
                             &m_bindless.globals_allocation, &allocation));
  vmaSetAllocationName(m_allocator, m_bindless.globals_allocation,
                       "bindless globals");
  set_object_name(vk_handle_value(m_bindless.globals), VK_OBJECT_TYPE_BUFFER,
                  "bindless globals and null page");

  VkBufferDeviceAddressInfo address_info{
      VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  address_info.buffer = m_bindless.globals;
  const u64 globals_address =
      vkGetBufferDeviceAddressKHR(m_device, &address_info);
  const u64 null_address = globals_address + NULL_PAGE_OFFSET;
  auto *bytes = static_cast<u8 *>(allocation.pMappedData);
  std::memset(bytes, 0, nx::cast<usize>(info.size));
  std::memcpy(bytes, &null_address, sizeof(null_address));
  NX_VK_DEVICE_TRY(*this, vmaFlushAllocation(m_allocator,
                                             m_bindless.globals_allocation, 0,
                                             VK_WHOLE_SIZE));

  if (m_routes.descriptor_heap) {
    const VkDeviceAddressRangeEXT range{globals_address, GLOBALS_BYTES};
    VkResourceDescriptorInfoEXT resource{
        VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT};
    resource.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    resource.data.pAddressRange = &range;
    DescriptorHeapBuffer &heap = m_bindless.resources;
    const VkHostAddressRangeEXT target{
        heap.data + m_bindless.globals_offset,
        nx::cast<usize>(heap.reserved_offset - m_bindless.globals_offset)};
    NX_VK_DEVICE_TRY(
        *this, vkWriteResourceDescriptorsEXT(m_device, 1, &resource, &target));
    NX_VK_DEVICE_TRY(*this, vmaFlushAllocation(m_allocator, heap.allocation,
                                               m_bindless.globals_offset,
                                               target.size));
    return true;
  }

  const VkDescriptorBufferInfo buffer{m_bindless.globals, 0, GLOBALS_BYTES};
  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = m_bindless.set;
  write.dstBinding = BINDLESS_BINDING_GLOBALS;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  write.pBufferInfo = &buffer;
  vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
  return true;
}

void VulkanDevice::destroy_bindless() noexcept {
  if (m_pipeline_layout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
    m_pipeline_layout = VK_NULL_HANDLE;
  }
  if (m_bindless.layout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(m_device, m_bindless.layout, nullptr);
  if (m_bindless.pool != VK_NULL_HANDLE)
    vkDestroyDescriptorPool(m_device, m_bindless.pool, nullptr);
  for (DescriptorHeapBuffer *const heap :
       {&m_bindless.resources, &m_bindless.samplers})
    if (heap->buffer != VK_NULL_HANDLE)
      vmaDestroyBuffer(m_allocator, heap->buffer, heap->allocation);
  if (m_bindless.globals != VK_NULL_HANDLE)
    vmaDestroyBuffer(m_allocator, m_bindless.globals,
                     m_bindless.globals_allocation);
  m_bindless = {};
}

void VulkanDevice::write_texture_descriptor(
    const u32 index, const VulkanTexture &texture) noexcept {
  if (index >= m_bindless.max_textures) {
    nx::loge(nx::log::rhi, "texture slot {} exceeds bindless capacity {}", index,
             m_bindless.max_textures);
    return;
  }
  if (m_routes.descriptor_heap) {
    const VkImageViewCreateInfo view =
        sampled_view_info(texture, texture.sampled_base_mip);
    VkImageDescriptorInfoEXT image{VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT};
    image.pView = &view;
    image.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkResourceDescriptorInfoEXT resource{
        VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT};
    resource.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    resource.data.pImage = &image;
    write_heap_descriptor(resource, index * m_bindless.image_stride);
    return;
  }

  VkDescriptorImageInfo image{};
  image.imageView = texture.srv;
  image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = m_bindless.set;
  write.dstBinding = BINDLESS_BINDING_TEXTURES;
  write.dstArrayElement = index;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  write.pImageInfo = &image;
  const nx::scoped_lock<nx::mutex> held(m_bindless_lock);
  vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
}

void VulkanDevice::write_rw_texture_descriptor(const u32 index,
                                               const VulkanTexture &texture,
                                               const u32 mip) noexcept {
  if (index >= m_bindless.max_rw_textures || mip >= texture.uavs.size())
    return;
  if (m_routes.descriptor_heap) {
    const VkImageViewCreateInfo view = storage_view_info(texture, mip);
    VkImageDescriptorInfoEXT image{VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT};
    image.pView = &view;
    image.layout = VK_IMAGE_LAYOUT_GENERAL;
    VkResourceDescriptorInfoEXT resource{
        VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT};
    resource.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    resource.data.pImage = &image;
    write_heap_descriptor(resource, m_bindless.rw_offset +
                                        index * m_bindless.image_stride);
    return;
  }

  VkDescriptorImageInfo image{};
  image.imageView = texture.uavs[mip];
  image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = m_bindless.set;
  write.dstBinding = BINDLESS_BINDING_RW_TEXTURES;
  write.dstArrayElement = index;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  write.pImageInfo = &image;
  const nx::scoped_lock<nx::mutex> held(m_bindless_lock);
  vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
}

void VulkanDevice::write_heap_descriptor(
    const VkResourceDescriptorInfoEXT &resource,
    const VkDeviceSize offset) noexcept {
  DescriptorHeapBuffer &heap = m_bindless.resources;
  const VkHostAddressRangeEXT target{heap.data + offset,
                                     nx::cast<usize>(m_bindless.image_stride)};
  // Each slot is its own bytes, but the allocation is flushed as one.
  const nx::scoped_lock<nx::mutex> held(m_bindless_lock);
  if (NX_VK_DEVICE_CHECK(*this,
                         vkWriteResourceDescriptorsEXT(m_device, 1, &resource,
                                                       &target)) == VK_SUCCESS)
    (void)vmaFlushAllocation(m_allocator, heap.allocation, offset, target.size);
}

void VulkanDevice::write_sampler_descriptor(
    const u32 index, const VkSampler sampler,
    const VkSamplerCreateInfo &info) noexcept {
  if (index >= m_bindless.max_samplers) {
    nx::loge(nx::log::rhi, "sampler slot {} exceeds bindless capacity {}", index,
             m_bindless.max_samplers);
    return;
  }
  if (m_routes.descriptor_heap) {
    DescriptorHeapBuffer &heap = m_bindless.samplers;
    const VkDeviceSize offset = index * m_bindless.sampler_stride;
    const VkHostAddressRangeEXT target{
        heap.data + offset, nx::cast<usize>(m_bindless.sampler_stride)};
    const nx::scoped_lock<nx::mutex> held(m_bindless_lock);
    if (NX_VK_DEVICE_CHECK(
            *this, vkWriteSamplerDescriptorsEXT(m_device, 1, &info, &target)) ==
        VK_SUCCESS)
      (void)vmaFlushAllocation(m_allocator, heap.allocation, offset,
                               target.size);
    return;
  }

  VkDescriptorImageInfo image{};
  image.sampler = sampler;

  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = m_bindless.set;
  write.dstBinding = BINDLESS_BINDING_SAMPLERS;
  write.dstArrayElement = index;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
  write.pImageInfo = &image;
  const nx::scoped_lock<nx::mutex> held(m_bindless_lock);
  vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
}

u32 VulkanDevice::alloc_rw_slot() noexcept {
  const nx::scoped_lock<nx::mutex> held(m_bindless_lock);
  if (!m_bindless.free_rw_slots.empty()) {
    const u32 slot = m_bindless.free_rw_slots.back();
    m_bindless.free_rw_slots.pop_back();
    return slot;
  }
  if (m_bindless.rw_high_water >= m_bindless.max_rw_textures) {
    nx::loge(nx::log::rhi, "out of rw-texture slots ({} used)",
             m_bindless.max_rw_textures);
    return BINDLESS_INVALID_INDEX;
  }
  return m_bindless.rw_high_water++;
}

void VulkanDevice::free_rw_slot(const u32 slot) noexcept {
  const nx::scoped_lock<nx::mutex> held(m_bindless_lock);
  if (slot != BINDLESS_INVALID_INDEX)
    m_bindless.free_rw_slots.push_back(slot);
}

} // namespace nx::gfx::detail
