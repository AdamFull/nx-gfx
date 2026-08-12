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

  [[nodiscard]] bool present(u32 slot);
  [[nodiscard]] VkImageView output_sampleable_view() const noexcept {
    return m_sampleable.view;
  }
  [[nodiscard]] VkFormat picture_format() const noexcept {
    return m_picture_format;
  }
  // Read the presented image's luma back - the verification counterpart to
  // present(), reading what a sampler would see.
  [[nodiscard]] bool read_output_luma(u32 width, u32 height,
                                      nx::vector<u8> &out);

  [[nodiscard]] bool convert(u32 slot);
  // The converted frame as a device texture: a normal sampled handle a caller
  // draws through the bindless path, so nothing Vulkan leaks out.
  [[nodiscard]] TextureHandle rgba_texture() const noexcept {
    return m_rgba_handle;
  }
  // Read the converted RGBA back - the verification counterpart to convert().
  [[nodiscard]] bool read_rgba(u32 width, u32 height, nx::vector<u8> &out);

private:
  [[nodiscard]] bool create_command_resources();
  [[nodiscard]] bool
  create_bitstream_buffer(u64 capacity,
                          const VkVideoProfileListInfoKHR &profiles);
  [[nodiscard]] bool
  create_sampleable_image(const u32 *families, u32 family_count);
  [[nodiscard]] bool create_convert_resources();
  [[nodiscard]] bool read_image_luma(VideoImage &img, VkImageLayout restore_to,
                                     u32 width, u32 height, nx::vector<u8> &out);
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

  VkCommandPool m_gfx_pool = VK_NULL_HANDLE;
  VkCommandBuffer m_gfx_cmd = VK_NULL_HANDLE;
  VkFence m_gfx_fence = VK_NULL_HANDLE;
  VkSamplerYcbcrConversion m_ycbcr = VK_NULL_HANDLE;
  VkSampler m_ycbcr_sampler = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_set_layout = VK_NULL_HANDLE;
  VkDescriptorPool m_desc_pool = VK_NULL_HANDLE;
  VkDescriptorSet m_set = VK_NULL_HANDLE;
  VkPipelineLayout m_pipe_layout = VK_NULL_HANDLE;
  VkPipeline m_pipeline = VK_NULL_HANDLE;
  // The RGBA target is a device texture (bindless-sampled by the caller), so its
  // layout is tracked here across the ycbcr pass and readback.
  TextureHandle m_rgba_handle;
  VkImageLayout m_rgba_layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

} // namespace nxe::rhi
