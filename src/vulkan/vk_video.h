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
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

struct DecodedPicture {
  bool ok = false;
  u32 slot = 0;
  bool show = false;
  u32 width = 0;
  u32 height = 0;
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

  [[nodiscard]] bool decode(const u8 *frame, u32 size, DecodedPicture &out);
  [[nodiscard]] bool read_luma(u32 slot, u32 width, u32 height,
                               nx::vector<u8> &out);

private:
  [[nodiscard]] bool create_command_resources();
  [[nodiscard]] bool
  create_bitstream_buffer(u64 capacity,
                          const VkVideoProfileListInfoKHR &profiles);
  [[nodiscard]] u32 allocate_slot() noexcept;

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

  // Decode-command plumbing.
  VkCommandPool m_pool = VK_NULL_HANDLE;
  VkCommandBuffer m_cmd = VK_NULL_HANDLE;
  VkFence m_fence = VK_NULL_HANDLE;
  bool m_video_supports_transfer = false;
  bool m_needs_reset = true;

  VkBuffer m_bitstream = VK_NULL_HANDLE;
  VkDeviceMemory m_bitstream_memory = VK_NULL_HANDLE;
  void *m_bitstream_mapped = nullptr;
  u64 m_bitstream_capacity = 0;
  u64 m_bitstream_offset_align = 1;
  u64 m_bitstream_size_align = 1;

  VkBuffer m_readback = VK_NULL_HANDLE;
  VkDeviceMemory m_readback_memory = VK_NULL_HANDLE;
  void *m_readback_mapped = nullptr;
  u64 m_readback_capacity = 0;

  // VP9 keeps eight reference buffers; this maps each to the DPB slot that holds
  // it, and remembers each slot's coded size for building picture resources.
  u32 m_vp9_to_dpb[8] = {};
  bool m_vp9_valid[8] = {};
  u32 m_slot_width[16] = {};
  u32 m_slot_height[16] = {};
  u32 m_next_slot = 0;
};

} // namespace nxe::rhi
