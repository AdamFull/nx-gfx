
#include "rendering/rhi/vulkan/vk_device.h"

#include "core/foundation/diagnostics/log.h"

namespace nxe::rhi {
namespace {

u32 clamp_count(const u32 desired, const u32 device_limit) noexcept {
  if (device_limit == 0)
    return desired;
  return desired < device_limit ? desired : device_limit;
}

} // namespace

bool VulkanDevice::create_bindless_heap() {
  m_bindless.max_samplers =
      clamp_count(MAX_BINDLESS_SAMPLERS, m_caps.max_bindless_samplers);
  m_bindless.max_textures =
      clamp_count(MAX_BINDLESS_TEXTURES, m_caps.max_bindless_sampled_images);
  m_bindless.max_rw_textures =
      clamp_count(MAX_BINDLESS_RW_TEXTURES, m_caps.max_bindless_storage_images);

  const VkDescriptorBindingFlags binding_flags =
      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
      VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
  const VkDescriptorBindingFlags flags[3] = {binding_flags, binding_flags,
                                             binding_flags};

  const VkDescriptorSetLayoutBinding bindings[3] = {
      {BINDLESS_BINDING_SAMPLERS, VK_DESCRIPTOR_TYPE_SAMPLER,
       m_bindless.max_samplers, VK_SHADER_STAGE_ALL, nullptr},
      {BINDLESS_BINDING_TEXTURES, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
       m_bindless.max_textures, VK_SHADER_STAGE_ALL, nullptr},
      {BINDLESS_BINDING_RW_TEXTURES, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
       m_bindless.max_rw_textures, VK_SHADER_STAGE_ALL, nullptr},
  };

  VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
  flags_info.bindingCount = 3;
  flags_info.pBindingFlags = flags;

  VkDescriptorSetLayoutCreateInfo layout_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layout_info.pNext = &flags_info;
  layout_info.flags =
      VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
  layout_info.bindingCount = 3;
  layout_info.pBindings = bindings;
  NX_VK_DEVICE_TRY(*this,
                   vkCreateDescriptorSetLayout(m_device, &layout_info, nullptr,
                                               &m_bindless.layout));
  set_object_name(vk_handle_value(m_bindless.layout),
                  VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                  "bindless descriptor layout");

  const VkDescriptorPoolSize pool_sizes[3] = {
      {VK_DESCRIPTOR_TYPE_SAMPLER, m_bindless.max_samplers},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, m_bindless.max_textures},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, m_bindless.max_rw_textures},
  };

  VkDescriptorPoolCreateInfo pool_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
  pool_info.maxSets = 1;
  pool_info.poolSizeCount = 3;
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

  nx::logd(nx::log::rhi, "bindless heap: {} samplers, {} textures, {} rw-textures",
           m_bindless.max_samplers, m_bindless.max_textures,
           m_bindless.max_rw_textures);
  return true;
}

void VulkanDevice::write_texture_descriptor(const u32 index,
                                            const VkImageView view) noexcept {
  if (index >= m_bindless.max_textures) {
    nx::loge(nx::log::rhi, "texture slot {} exceeds bindless capacity {}", index,
             m_bindless.max_textures);
    return;
  }
  VkDescriptorImageInfo image{};
  image.imageView = view;
  image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = m_bindless.set;
  write.dstBinding = BINDLESS_BINDING_TEXTURES;
  write.dstArrayElement = index;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  write.pImageInfo = &image;
  vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
}

void VulkanDevice::write_rw_texture_descriptor(
    const u32 index, const VkImageView view) noexcept {
  if (index >= m_bindless.max_rw_textures)
    return;
  VkDescriptorImageInfo image{};
  image.imageView = view;
  image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = m_bindless.set;
  write.dstBinding = BINDLESS_BINDING_RW_TEXTURES;
  write.dstArrayElement = index;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  write.pImageInfo = &image;
  vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
}

void VulkanDevice::write_sampler_descriptor(const u32 index,
                                            const VkSampler sampler) noexcept {
  if (index >= m_bindless.max_samplers) {
    nx::loge(nx::log::rhi, "sampler slot {} exceeds bindless capacity {}", index,
             m_bindless.max_samplers);
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
  vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
}

u32 VulkanDevice::alloc_rw_slot() noexcept {
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
  if (slot != BINDLESS_INVALID_INDEX)
    m_bindless.free_rw_slots.push_back(slot);
}

} // namespace nxe::rhi
