#pragma once

/**
 * @file vk_video.h
 * @brief A hardware video-decode session (namespace nxe::rhi).
 */

#include "core/foundation/containers/small_vector.h"
#include "core/rendering/rhi/types.h"
#include "core/rendering/rhi/vulkan/vk_common.h"

namespace nxe::rhi {

class VulkanDevice;

struct VideoImage {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
};

class VideoDecoder {
public:
  VideoDecoder() = default;
  ~VideoDecoder();

  VideoDecoder(const VideoDecoder &) = delete;
  VideoDecoder &operator=(const VideoDecoder &) = delete;

  [[nodiscard]] bool init(VulkanDevice &device, VideoCodec codec, u32 width,
                          u32 height);
  void shutdown() noexcept;

  [[nodiscard]] bool ready() const noexcept {
    return m_session != VK_NULL_HANDLE;
  }
  [[nodiscard]] VkExtent2D extent() const noexcept { return m_extent; }
  [[nodiscard]] u32 dpb_slots() const noexcept { return m_max_dpb_slots; }

  [[nodiscard]] u32 dpb_image_count() const noexcept {
    return nx::cast<u32>(m_dpb.size());
  }
  [[nodiscard]] VkImageView dpb_view(u32 slot) const noexcept {
    return slot < m_dpb.size() ? m_dpb[slot].view : VK_NULL_HANDLE;
  }
  // The decode target, distinct from the DPB slots only where the device keeps
  // them apart; null on a coincide device, where the slot is the target.
  [[nodiscard]] VkImageView output_view() const noexcept {
    return m_output.view;
  }

private:
  VulkanDevice *m_device = nullptr;
  VkVideoSessionKHR m_session = VK_NULL_HANDLE;
  nx::small_vector<VkDeviceMemory, 4> m_memories;
  nx::small_vector<VideoImage, 16> m_dpb;
  VideoImage m_output;
  bool m_coincide = false;
  VkFormat m_picture_format = VK_FORMAT_UNDEFINED;
  VkFormat m_dpb_format = VK_FORMAT_UNDEFINED;
  VkExtent2D m_extent = {};
  u32 m_max_dpb_slots = 0;
};

} // namespace nxe::rhi
