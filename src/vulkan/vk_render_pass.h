#pragma once

#include "core/foundation/core/foundation.h"
#include "core/foundation/threading/sync.h"
#include "nx/gfx/types.h"
#include "src/vulkan/vk_common.h"

#include <utility>

namespace nx::gfx::detail {

/// One attachment of a pass on the render-pass route.
struct RenderPassAttachment {
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
  VkAttachmentLoadOp load = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  VkAttachmentStoreOp store = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  /// The single-sampled target a color attachment resolves into, if any.
  VkFormat resolve_format = VK_FORMAT_UNDEFINED;

  friend bool operator==(const RenderPassAttachment &,
                         const RenderPassAttachment &) = default;
};

/// What a VkRenderPass is made from. Every attachment starts and ends in the
/// layout the graph's barriers left it in, so the pass transitions nothing
/// and needs no external dependencies.
struct RenderPassKey {
  RenderPassAttachment colors[MAX_COLOR_ATTACHMENTS] = {};
  RenderPassAttachment depth;
  u32 color_count = 0;
  bool has_depth = false;
  bool depth_read_only = false;

  friend bool operator==(const RenderPassKey &,
                         const RenderPassKey &) = default;
};

/// The image behind one attachment of an imageless framebuffer, as it was
/// created: a view given at begin must match it exactly.
struct FramebufferAttachment {
  VkImageUsageFlags usage = 0;
  VkImageCreateFlags flags = 0;
  u32 width = 0;
  u32 height = 0;
  u32 layers = 1;
  VkFormat format = VK_FORMAT_UNDEFINED;

  friend bool operator==(const FramebufferAttachment &,
                         const FramebufferAttachment &) = default;
};

/// A pass's colors, then the resolve targets of those that resolve, then
/// depth: the order RenderPassKey's pass declares them in.
constexpr u32 MAX_PASS_ATTACHMENTS = 2 * MAX_COLOR_ATTACHMENTS + 1;

struct FramebufferKey {
  VkRenderPass pass = VK_NULL_HANDLE;
  u32 width = 0;
  u32 height = 0;
  u32 attachment_count = 0;
  FramebufferAttachment attachments[MAX_PASS_ATTACHMENTS] = {};

  friend bool operator==(const FramebufferKey &,
                         const FramebufferKey &) = default;
};

/// The render passes and imageless framebuffers of the render-pass route,
/// made on first use and kept until the device goes. Any thread: the lock
/// guards lookups and inserts, never a driver call.
class VulkanRenderPasses {
public:
  VulkanRenderPasses() = default;
  ~VulkanRenderPasses() { shutdown(); }

  VulkanRenderPasses(const VulkanRenderPasses &) = delete;
  VulkanRenderPasses &operator=(const VulkanRenderPasses &) = delete;

  /// @p store_op_none: a read-only depth attachment stores nothing, rather
  /// than storing what it read.
  void init(VkDevice device, bool store_op_none) noexcept;
  void shutdown() noexcept;

  /// The pass for @p key, made now if it is new; null if the driver refused.
  [[nodiscard]] VkRenderPass pass(const RenderPassKey &key);
  /// The framebuffer for @p key, made now if it is new; null if the driver
  /// refused.
  [[nodiscard]] VkFramebuffer framebuffer(const FramebufferKey &key);

private:
  [[nodiscard]] VkRenderPass make_pass(const RenderPassKey &key) const;
  [[nodiscard]] VkFramebuffer make_framebuffer(const FramebufferKey &key) const;

  VkDevice m_device = VK_NULL_HANDLE;
  bool m_store_op_none = false;
  nx::mutex m_lock;
  nx::vector<std::pair<RenderPassKey, VkRenderPass>> m_passes;
  nx::vector<std::pair<FramebufferKey, VkFramebuffer>> m_framebuffers;
};

} // namespace nx::gfx::detail
