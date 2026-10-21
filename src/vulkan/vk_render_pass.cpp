#include "src/vulkan/vk_render_pass.h"

#include "core/foundation/diagnostics/log.h"

namespace nx::gfx::detail {

void VulkanRenderPasses::init(VkDevice device,
                              const bool store_op_none) noexcept {
  m_device = device;
  m_store_op_none = store_op_none;
}

void VulkanRenderPasses::shutdown() noexcept {
  for (const auto &[key, framebuffer] : m_framebuffers)
    vkDestroyFramebuffer(m_device, framebuffer, nullptr);
  for (const auto &[key, pass] : m_passes)
    vkDestroyRenderPass(m_device, pass, nullptr);
  m_framebuffers.clear();
  m_passes.clear();
  m_device = VK_NULL_HANDLE;
}

VkRenderPass VulkanRenderPasses::pass(const RenderPassKey &key) {
  {
    const nx::scoped_lock<nx::mutex> held(m_lock);
    for (const auto &[have, pass] : m_passes)
      if (have == key)
        return pass;
  }
  const VkRenderPass made = make_pass(key);
  if (made == VK_NULL_HANDLE)
    return VK_NULL_HANDLE;
  const nx::scoped_lock<nx::mutex> held(m_lock);
  for (const auto &[have, pass] : m_passes) {
    if (have == key) {
      vkDestroyRenderPass(m_device, made, nullptr);
      return pass;
    }
  }
  m_passes.push_back({key, made});
  return made;
}

VkFramebuffer VulkanRenderPasses::framebuffer(const FramebufferKey &key) {
  {
    const nx::scoped_lock<nx::mutex> held(m_lock);
    for (const auto &[have, framebuffer] : m_framebuffers)
      if (have == key)
        return framebuffer;
  }
  const VkFramebuffer made = make_framebuffer(key);
  if (made == VK_NULL_HANDLE)
    return VK_NULL_HANDLE;
  const nx::scoped_lock<nx::mutex> held(m_lock);
  for (const auto &[have, framebuffer] : m_framebuffers) {
    if (have == key) {
      vkDestroyFramebuffer(m_device, made, nullptr);
      return framebuffer;
    }
  }
  m_framebuffers.push_back({key, made});
  return made;
}

VkRenderPass VulkanRenderPasses::make_pass(const RenderPassKey &key) const {
  VkAttachmentDescription attachments[MAX_PASS_ATTACHMENTS] = {};
  VkAttachmentReference colors[MAX_COLOR_ATTACHMENTS] = {};
  VkAttachmentReference resolves[MAX_COLOR_ATTACHMENTS] = {};
  u32 count = 0;
  bool resolving = false;

  const auto describe = [&](const RenderPassAttachment &attachment,
                            const VkImageLayout layout) {
    VkAttachmentDescription &out = attachments[count];
    out.format = attachment.format;
    out.samples = attachment.samples;
    out.loadOp = attachment.load;
    out.storeOp = attachment.store;
    out.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    out.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    out.initialLayout = layout;
    out.finalLayout = layout;
    return count++;
  };

  // A pipeline may leave a location without a format, as dynamic rendering
  // allows: that location has no attachment.
  for (u32 i = 0; i < key.color_count; ++i)
    colors[i] = key.colors[i].format == VK_FORMAT_UNDEFINED
                    ? VkAttachmentReference{VK_ATTACHMENT_UNUSED,
                                            VK_IMAGE_LAYOUT_UNDEFINED}
                    : VkAttachmentReference{
                          describe(key.colors[i],
                                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL),
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  for (u32 i = 0; i < key.color_count; ++i) {
    if (key.colors[i].resolve_format == VK_FORMAT_UNDEFINED) {
      resolves[i] = {VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_UNDEFINED};
      continue;
    }
    resolving = true;
    resolves[i] = {describe({.format = key.colors[i].resolve_format,
                             .store = VK_ATTACHMENT_STORE_OP_STORE},
                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL),
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  }

  VkAttachmentReference depth{};
  if (key.has_depth) {
    const VkImageLayout layout =
        key.depth_read_only ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                            : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    RenderPassAttachment attachment = key.depth;
    // Storing what a read-only pass read would be a write; without
    // STORE_OP_NONE storing is the only op that keeps the contents.
    if (key.depth_read_only)
      attachment.store = m_store_op_none ? VK_ATTACHMENT_STORE_OP_NONE
                                         : VK_ATTACHMENT_STORE_OP_STORE;
    const u32 index = describe(attachment, layout);
    // Dynamic rendering leaves stencil alone where a pass names only depth;
    // here it has to be kept explicitly.
    attachments[index].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[index].stencilStoreOp = attachment.store;
    depth = {index, layout};
  }

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = key.color_count;
  subpass.pColorAttachments = colors;
  subpass.pResolveAttachments = resolving ? resolves : nullptr;
  subpass.pDepthStencilAttachment = key.has_depth ? &depth : nullptr;

  VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  info.attachmentCount = count;
  info.pAttachments = attachments;
  info.subpassCount = 1;
  info.pSubpasses = &subpass;

  VkRenderPass pass = VK_NULL_HANDLE;
  if (const VkResult result =
          vkCreateRenderPass(m_device, &info, nullptr, &pass);
      result != VK_SUCCESS) {
    nx::loge(nx::log::rhi, "vkCreateRenderPass failed ({})",
             vk_result_string(result));
    return VK_NULL_HANDLE;
  }
  return pass;
}

VkFramebuffer
VulkanRenderPasses::make_framebuffer(const FramebufferKey &key) const {
  VkFramebufferAttachmentImageInfo images[MAX_PASS_ATTACHMENTS] = {};
  for (u32 i = 0; i < key.attachment_count; ++i) {
    const FramebufferAttachment &attachment = key.attachments[i];
    images[i].sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO;
    images[i].flags = attachment.flags;
    images[i].usage = attachment.usage;
    images[i].width = attachment.width;
    images[i].height = attachment.height;
    images[i].layerCount = attachment.layers;
    images[i].viewFormatCount = 1;
    images[i].pViewFormats = &attachment.format;
  }
  VkFramebufferAttachmentsCreateInfo attachments{
      VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENTS_CREATE_INFO};
  attachments.attachmentImageInfoCount = key.attachment_count;
  attachments.pAttachmentImageInfos = images;

  VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  info.pNext = &attachments;
  info.flags = VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT;
  info.renderPass = key.pass;
  info.attachmentCount = key.attachment_count;
  info.width = key.width;
  info.height = key.height;
  info.layers = 1;

  VkFramebuffer framebuffer = VK_NULL_HANDLE;
  if (const VkResult result =
          vkCreateFramebuffer(m_device, &info, nullptr, &framebuffer);
      result != VK_SUCCESS) {
    nx::loge(nx::log::rhi, "vkCreateFramebuffer failed ({})",
             vk_result_string(result));
    return VK_NULL_HANDLE;
  }
  return framebuffer;
}

} // namespace nx::gfx::detail
