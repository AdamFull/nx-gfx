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

enum class VideoCodec : u8 { VP9, AV1, Count };

struct VideoDecodeCaps {
  bool supported = false;
  u32 max_width = 0;
  u32 max_height = 0;
  u32 max_dpb_slots = 0;
  bool dpb_and_output_coincide = false;
};

[[nodiscard]] VideoDecodeCaps query_video_decode_caps(const VulkanDevice &device,
                                                      VideoCodec codec);

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

  [[nodiscard]] bool decode_frame(const u8 *frame, u32 size, bool &shown,
                                  u32 &width, u32 &height);
  [[nodiscard]] bool show_last();
  [[nodiscard]] TextureHandle luma_texture() const noexcept { return m_y; }
  [[nodiscard]] TextureHandle chroma_texture() const noexcept { return m_cbcr; }
  [[nodiscard]] Extent2D coded_extent() const noexcept {
    return {m_extent.width, m_extent.height};
  }

  void reset_stream() noexcept;

private:
  [[nodiscard]] bool decode(const u8 *frame, u32 size, DecodedPicture &out);
  [[nodiscard]] bool present(u32 slot);
  [[nodiscard]] bool create_command_resources();
  [[nodiscard]] bool
  create_bitstream_buffer(u64 capacity,
                          const VkVideoProfileListInfoKHR &profiles);
  [[nodiscard]] bool
  create_sampleable_image(const u32 *families, u32 family_count);
  [[nodiscard]] bool create_plane_resources();
  [[nodiscard]] u32 allocate_slot() noexcept;

  VulkanDevice *m_device = nullptr;
  VkVideoSessionKHR m_session = VK_NULL_HANDLE;
  nx::small_vector<VkDeviceMemory, 4> m_memories;
  nx::small_vector<VideoImage, 16> m_dpb;
  VideoImage m_output;
  // A plain sampled NV12 image the decoded picture is copied into, so it can be
  // read without disturbing the DPB slots.
  VideoImage m_sampleable;
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

  // VP9 keeps eight reference buffers; this maps each to the DPB slot that holds
  // it, and remembers each slot's coded size for building picture resources.
  u32 m_vp9_to_dpb[8] = {};
  bool m_vp9_valid[8] = {};
  u32 m_slot_width[16] = {};
  u32 m_slot_height[16] = {};
  u32 m_next_slot = 0;
  u32 m_last_slot = 0;
  bool m_have_last = false;

  // The plane resolve runs on the graphics queue: it copies the presented NV12
  // picture out into two bindless-sampled device textures.
  VkCommandPool m_gfx_pool = VK_NULL_HANDLE;
  VkCommandBuffer m_gfx_cmd = VK_NULL_HANDLE;
  VkFence m_gfx_fence = VK_NULL_HANDLE;
  TextureHandle m_y;    ///< R8 luma
  TextureHandle m_cbcr; ///< R8G8 interleaved chroma
};

} // namespace nxe::rhi
