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

private:
  VulkanDevice *m_device = nullptr;
  VkVideoSessionKHR m_session = VK_NULL_HANDLE;
  nx::small_vector<VkDeviceMemory, 4> m_memories;
  VkFormat m_picture_format = VK_FORMAT_UNDEFINED;
  VkFormat m_dpb_format = VK_FORMAT_UNDEFINED;
  VkExtent2D m_extent = {};
  u32 m_max_dpb_slots = 0;
};

} // namespace nxe::rhi
