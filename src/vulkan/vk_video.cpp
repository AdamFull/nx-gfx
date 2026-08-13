/**
 * @file vk_video.cpp
 * @brief Vulkan Video: what the device can hardware-decode, and (later) the
 * decode itself. Kept apart from vk_device.cpp because the video queue is its
 * own world of sessions, DPB slots and codec std structs.
 */

#include "core/rendering/rhi/vulkan/vk_common.h"
#include "core/rendering/rhi/vulkan/vk_device.h"
#include "core/rendering/rhi/vulkan/vk_video.h"
#include "core/rendering/rhi/vulkan/vk_vp9.h"

#include "core/foundation/diagnostics/log.h"

#include <cstring>

namespace nxe::rhi {
namespace {

[[nodiscard]] bool extension_offered(const VkPhysicalDevice phys,
                                     const char *const name) {
  u32 count = 0;
  if (vkEnumerateDeviceExtensionProperties(phys, nullptr, &count, nullptr) !=
      VK_SUCCESS)
    return false;
  nx::vector<VkExtensionProperties> extensions(count);
  if (vkEnumerateDeviceExtensionProperties(phys, nullptr, &count,
                                           extensions.data()) != VK_SUCCESS)
    return false;
  for (const VkExtensionProperties &e : extensions)
    if (std::strcmp(e.extensionName, name) == 0)
      return true;
  return false;
}

} // namespace

u32 VulkanDevice::find_video_decode_family() const {
  constexpr u32 NONE = ~0u;
  if (m_physical_device == VK_NULL_HANDLE)
    return NONE;

  // All three extensions, or there is no session to build later.
  if (!extension_offered(m_physical_device, VK_KHR_VIDEO_QUEUE_EXTENSION_NAME) ||
      !extension_offered(m_physical_device,
                         VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME) ||
      !extension_offered(m_physical_device,
                         VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME))
    return NONE;

  // VP9 decode is a feature that has to be enable-able, not just an extension.
  VkPhysicalDeviceVideoDecodeVP9FeaturesKHR vp9{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_DECODE_VP9_FEATURES_KHR};
  VkPhysicalDeviceFeatures2 features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features.pNext = &vp9;
  vkGetPhysicalDeviceFeatures2(m_physical_device, &features);
  if (vp9.videoDecodeVP9 != VK_TRUE)
    return NONE;

  u32 count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties2(m_physical_device, &count, nullptr);
  nx::small_vector<VkQueueFamilyProperties2, 8> props(count);
  nx::small_vector<VkQueueFamilyVideoPropertiesKHR, 8> video(count);
  for (u32 i = 0; i < count; ++i) {
    props[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
    video[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR;
    props[i].pNext = &video[i];
  }
  vkGetPhysicalDeviceQueueFamilyProperties2(m_physical_device, &count,
                                            props.data());
  for (u32 i = 0; i < count; ++i) {
    const VkQueueFlags flags = props[i].queueFamilyProperties.queueFlags;
    if ((flags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) &&
        (video[i].videoCodecOperations &
         VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR) &&
        !(flags & VK_QUEUE_GRAPHICS_BIT))
      return i;
  }
  return NONE;
}

bool VulkanDevice::video_decode_av1_capable(const u32 family) const {
  // AV1 decode has no feature struct to enable, unlike VP9 - the extension plus
  // a queue family carrying the AV1 decode op are the whole gate.
  if (m_physical_device == VK_NULL_HANDLE || family == ~0u)
    return false;
  if (!extension_offered(m_physical_device,
                         VK_KHR_VIDEO_DECODE_AV1_EXTENSION_NAME))
    return false;

  u32 count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties2(m_physical_device, &count, nullptr);
  if (family >= count)
    return false;
  nx::small_vector<VkQueueFamilyProperties2, 8> props(count);
  nx::small_vector<VkQueueFamilyVideoPropertiesKHR, 8> video(count);
  for (u32 i = 0; i < count; ++i) {
    props[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
    video[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR;
    props[i].pNext = &video[i];
  }
  vkGetPhysicalDeviceQueueFamilyProperties2(m_physical_device, &count,
                                            props.data());
  return (video[family].videoCodecOperations &
          VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR) != 0;
}

VideoDecodeCaps query_video_decode_caps(const VulkanDevice &device,
                                        const VideoCodec codec) {
  VideoDecodeCaps out;
  const VkPhysicalDevice phys = device.vk_physical_device();
  if (phys == VK_NULL_HANDLE ||
      vkGetPhysicalDeviceVideoCapabilitiesKHR == nullptr)
    return out;
  if (!extension_offered(phys, VK_KHR_VIDEO_QUEUE_EXTENSION_NAME))
    return out;

  const char *const codec_extension =
      codec == VideoCodec::AV1 ? VK_KHR_VIDEO_DECODE_AV1_EXTENSION_NAME
                               : VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME;
  if (!extension_offered(phys, codec_extension))
    return out;

  VkVideoProfileInfoKHR profile{VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR};
  profile.chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
  profile.lumaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
  profile.chromaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;

  VkVideoDecodeCapabilitiesKHR decode{
      VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR};
  VkVideoCapabilitiesKHR caps{VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR};
  caps.pNext = &decode;

  // The codec-specific caps struct has to be chained too, or some drivers (Intel
  // among them) fault reaching for it inside the query rather than skipping it.
  VkVideoDecodeVP9ProfileInfoKHR vp9{
      VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_PROFILE_INFO_KHR};
  VkVideoDecodeVP9CapabilitiesKHR vp9_caps{
      VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_CAPABILITIES_KHR};
  VkVideoDecodeAV1ProfileInfoKHR av1{
      VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR};
  VkVideoDecodeAV1CapabilitiesKHR av1_caps{
      VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_CAPABILITIES_KHR};

  if (codec == VideoCodec::AV1) {
    av1.stdProfile = STD_VIDEO_AV1_PROFILE_MAIN;
    av1.filmGrainSupport = VK_FALSE;
    profile.pNext = &av1;
    profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR;
    decode.pNext = &av1_caps;
  } else {
    vp9.stdProfile = STD_VIDEO_VP9_PROFILE_0;
    profile.pNext = &vp9;
    profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR;
    decode.pNext = &vp9_caps;
  }

  if (vkGetPhysicalDeviceVideoCapabilitiesKHR(phys, &profile, &caps) !=
      VK_SUCCESS)
    return out;

  out.supported = true;
  out.max_width = caps.maxCodedExtent.width;
  out.max_height = caps.maxCodedExtent.height;
  out.max_height = caps.maxCodedExtent.height;
  out.max_dpb_slots = caps.maxDpbSlots;
  out.dpb_and_output_coincide =
      (decode.flags &
       VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR) != 0;
  return out;
}

namespace {

const nx::log::Category log_video = nx::log::category("rhi");

[[nodiscard]] u32 align_up(const u32 value, const u32 to) {
  return to <= 1 ? value : (value + to - 1) / to * to;
}

[[nodiscard]] u32 clamp_u32(const u32 value, const u32 lo, const u32 hi) {
  return value < lo ? lo : (value > hi ? hi : value);
}

[[nodiscard]] VkFormat first_video_format(const VkPhysicalDevice phys,
                                          const VkVideoProfileListInfoKHR &list,
                                          const VkImageUsageFlags usage) {
  VkPhysicalDeviceVideoFormatInfoKHR info{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_FORMAT_INFO_KHR};
  info.pNext = &list;
  info.imageUsage = usage;

  u32 count = 0;
  if (vkGetPhysicalDeviceVideoFormatPropertiesKHR(phys, &info, &count,
                                                  nullptr) != VK_SUCCESS ||
      count == 0)
    return VK_FORMAT_UNDEFINED;
  nx::small_vector<VkVideoFormatPropertiesKHR, 8> formats(count);
  for (VkVideoFormatPropertiesKHR &f : formats)
    f.sType = VK_STRUCTURE_TYPE_VIDEO_FORMAT_PROPERTIES_KHR;
  if (vkGetPhysicalDeviceVideoFormatPropertiesKHR(phys, &info, &count,
                                                  formats.data()) != VK_SUCCESS)
    return VK_FORMAT_UNDEFINED;
  return formats[0].format;
}

[[nodiscard]] u32 session_memory_type(const VkPhysicalDevice phys,
                                      const u32 type_bits) {
  VkPhysicalDeviceMemoryProperties props;
  vkGetPhysicalDeviceMemoryProperties(phys, &props);
  for (u32 i = 0; i < props.memoryTypeCount; ++i)
    if ((type_bits & (1u << i)) &&
        (props.memoryTypes[i].propertyFlags &
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
      return i;
  for (u32 i = 0; i < props.memoryTypeCount; ++i)
    if (type_bits & (1u << i))
      return i;
  return ~0u;
}

[[nodiscard]] bool create_video_image(const VkDevice dev,
                                      const VkPhysicalDevice phys,
                                      const VkVideoProfileListInfoKHR &profiles,
                                      const VkFormat format,
                                      const VkExtent2D extent,
                                      const VkImageUsageFlags usage,
                                      const u32 *const families,
                                      const u32 family_count, VideoImage &out) {
  VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ci.pNext = &profiles;
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.format = format;
  ci.extent = {extent.width, extent.height, 1};
  ci.mipLevels = 1;
  ci.arrayLayers = 1;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = usage;
  // The decode queue writes these; a transfer or graphics queue reads them
  // back. Concurrent sharing spares an ownership hand-off between the two.
  if (family_count > 1) {
    ci.sharingMode = VK_SHARING_MODE_CONCURRENT;
    ci.queueFamilyIndexCount = family_count;
    ci.pQueueFamilyIndices = families;
  } else {
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (vkCreateImage(dev, &ci, nullptr, &out.image) != VK_SUCCESS)
    return false;

  VkMemoryRequirements mr;
  vkGetImageMemoryRequirements(dev, out.image, &mr);
  const u32 type = session_memory_type(phys, mr.memoryTypeBits);
  if (type == ~0u)
    return false;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = type;
  if (vkAllocateMemory(dev, &ai, nullptr, &out.memory) != VK_SUCCESS)
    return false;
  if (vkBindImageMemory(dev, out.image, out.memory, 0) != VK_SUCCESS)
    return false;

  // A plain colour view spanning both planes: video decode reads and writes the
  // whole picture, so no ycbcr conversion here - that arrives with sampling.
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = out.image;
  vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vi.format = format;
  vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  return vkCreateImageView(dev, &vi, nullptr, &out.view) == VK_SUCCESS;
}

void destroy_video_image(const VkDevice dev, VideoImage &image) {
  if (image.view != VK_NULL_HANDLE)
    vkDestroyImageView(dev, image.view, nullptr);
  if (image.image != VK_NULL_HANDLE)
    vkDestroyImage(dev, image.image, nullptr);
  if (image.memory != VK_NULL_HANDLE)
    vkFreeMemory(dev, image.memory, nullptr);
  image = {};
}

[[nodiscard]] u64 align_up64(const u64 value, const u64 to) {
  return to <= 1 ? value : (value + to - 1) / to * to;
}

[[nodiscard]] u32 host_memory_type(const VkPhysicalDevice phys,
                                   const u32 type_bits) {
  VkPhysicalDeviceMemoryProperties props;
  vkGetPhysicalDeviceMemoryProperties(phys, &props);
  const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  for (u32 i = 0; i < props.memoryTypeCount; ++i)
    if ((type_bits & (1u << i)) &&
        (props.memoryTypes[i].propertyFlags & want) == want)
      return i;
  return ~0u;
}

[[nodiscard]] bool create_host_buffer(const VkDevice dev,
                                      const VkPhysicalDevice phys,
                                      const u64 size,
                                      const VkBufferUsageFlags usage,
                                      const void *const pNext, VkBuffer &buffer,
                                      VkDeviceMemory &memory, void *&mapped) {
  VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  ci.pNext = pNext;
  ci.size = size;
  ci.usage = usage;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(dev, &ci, nullptr, &buffer) != VK_SUCCESS)
    return false;

  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, buffer, &mr);
  const u32 type = host_memory_type(phys, mr.memoryTypeBits);
  if (type == ~0u)
    return false;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = type;
  if (vkAllocateMemory(dev, &ai, nullptr, &memory) != VK_SUCCESS)
    return false;
  if (vkBindBufferMemory(dev, buffer, memory, 0) != VK_SUCCESS)
    return false;
  return vkMapMemory(dev, memory, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS;
}

} // namespace

VideoDecoder::~VideoDecoder() { shutdown(); }

void VideoDecoder::shutdown() noexcept {
  if (m_device != nullptr) {
    const VkDevice dev = m_device->vk_device();
    vkDeviceWaitIdle(dev);

    // The plane resolve's textures and command plumbing.
    if (m_y.valid())
      m_device->destroy_texture(m_y);
    if (m_cbcr.valid())
      m_device->destroy_texture(m_cbcr);
    destroy_video_image(dev, m_sampleable);
    if (m_gfx_fence != VK_NULL_HANDLE)
      vkDestroyFence(dev, m_gfx_fence, nullptr);
    if (m_gfx_pool != VK_NULL_HANDLE)
      vkDestroyCommandPool(dev, m_gfx_pool, nullptr);

    if (m_bitstream != VK_NULL_HANDLE)
      vkDestroyBuffer(dev, m_bitstream, nullptr);
    if (m_bitstream_memory != VK_NULL_HANDLE)
      vkFreeMemory(dev, m_bitstream_memory, nullptr);
    if (m_fence != VK_NULL_HANDLE)
      vkDestroyFence(dev, m_fence, nullptr);
    if (m_pool != VK_NULL_HANDLE)
      vkDestroyCommandPool(dev, m_pool, nullptr);
    for (VideoImage &image : m_dpb)
      destroy_video_image(dev, image);
    destroy_video_image(dev, m_output);
    if (m_session_params != VK_NULL_HANDLE)
      vkDestroyVideoSessionParametersKHR(dev, m_session_params, nullptr);
    if (m_session != VK_NULL_HANDLE)
      vkDestroyVideoSessionKHR(dev, m_session, nullptr);
    for (const VkDeviceMemory memory : m_memories)
      vkFreeMemory(dev, memory, nullptr);
  }
  m_session = VK_NULL_HANDLE;
  m_session_params = VK_NULL_HANDLE;
  m_memories.clear();
  m_dpb.clear();
  m_output = {};
  m_sampleable = {};
  m_coincide = false;
  m_pool = VK_NULL_HANDLE;
  m_cmd = VK_NULL_HANDLE;
  m_fence = VK_NULL_HANDLE;
  m_video_supports_transfer = false;
  m_needs_reset = true;
  m_bitstream = VK_NULL_HANDLE;
  m_bitstream_memory = VK_NULL_HANDLE;
  m_bitstream_mapped = nullptr;
  m_bitstream_capacity = 0;
  m_bitstream_offset_align = 1;
  m_bitstream_size_align = 1;
  for (u32 i = 0; i < 8; ++i)
    m_ref_valid[i] = false;
  m_next_slot = 0;
  m_last_slot = 0;
  m_have_last = false;
  m_gfx_pool = VK_NULL_HANDLE;
  m_gfx_cmd = VK_NULL_HANDLE;
  m_gfx_fence = VK_NULL_HANDLE;
  m_y = {};
  m_cbcr = {};
  m_device = nullptr;
  m_picture_format = VK_FORMAT_UNDEFINED;
  m_dpb_format = VK_FORMAT_UNDEFINED;
  m_extent = {};
  m_max_dpb_slots = 0;
}

bool VideoDecoder::init(VulkanDevice &device, const VideoCodec codec,
                        const u32 width, const u32 height,
                        const Av1SequenceHeader *const av1_seq) {
  shutdown();
  if (!device.video_decode_available())
    return false;
  if (codec == VideoCodec::AV1) {
    if (!device.video_decode_av1_available() || av1_seq == nullptr ||
        !av1_seq->valid)
      return false;
  } else if (codec != VideoCodec::VP9) {
    return false;
  }

  m_device = &device;
  m_codec = codec;
  if (codec == VideoCodec::AV1) {
    m_av1_seq = *av1_seq;
    m_av1_seq.header.pColorConfig = &m_av1_seq.color;
    m_av1_seq.header.pTimingInfo =
        av1_seq->header.pTimingInfo != nullptr ? &m_av1_seq.timing : nullptr;
  }
  const VkPhysicalDevice phys = device.vk_physical_device();
  const VkDevice dev = device.vk_device();

  VkVideoDecodeVP9ProfileInfoKHR vp9{
      VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_PROFILE_INFO_KHR};
  vp9.stdProfile = STD_VIDEO_VP9_PROFILE_0;
  VkVideoDecodeAV1ProfileInfoKHR av1{
      VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR};
  av1.stdProfile = STD_VIDEO_AV1_PROFILE_MAIN;
  av1.filmGrainSupport = VK_FALSE;
  VkVideoProfileInfoKHR profile{VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR};
  profile.chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
  profile.lumaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
  profile.chromaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;

  VkVideoDecodeVP9CapabilitiesKHR vp9_caps{
      VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_CAPABILITIES_KHR};
  VkVideoDecodeAV1CapabilitiesKHR av1_caps{
      VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_CAPABILITIES_KHR};
  VkVideoDecodeCapabilitiesKHR decode_caps{
      VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR};
  if (codec == VideoCodec::AV1) {
    profile.pNext = &av1;
    profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR;
    decode_caps.pNext = &av1_caps;
  } else {
    profile.pNext = &vp9;
    profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR;
    decode_caps.pNext = &vp9_caps;
  }
  VkVideoCapabilitiesKHR caps{VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR};
  caps.pNext = &decode_caps;
  if (const VkResult r =
          vkGetPhysicalDeviceVideoCapabilitiesKHR(phys, &profile, &caps);
      r != VK_SUCCESS) {
    nx::logw(log_video, "video: capabilities query failed ({})",
             nx::cast<i32>(r));
    m_device = nullptr;
    return false;
  }

  VkVideoProfileListInfoKHR profiles{
      VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR};
  profiles.profileCount = 1;
  profiles.pProfiles = &profile;
  m_coincide =
      (decode_caps.flags &
       VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR) != 0;
  const bool coincide = m_coincide;
  if (coincide) {
    m_dpb_format = first_video_format(
        phys, profiles,
        VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR |
            VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR);
    m_picture_format = m_dpb_format;
  } else {
    m_dpb_format = first_video_format(phys, profiles,
                                      VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR);
    m_picture_format = first_video_format(
        phys, profiles, VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR);
  }
  if (m_dpb_format == VK_FORMAT_UNDEFINED ||
      m_picture_format == VK_FORMAT_UNDEFINED) {
    nx::logw(log_video, "video: no decode image format for the profile");
    m_device = nullptr;
    return false;
  }

  m_extent.width =
      clamp_u32(align_up(width, caps.pictureAccessGranularity.width),
                caps.minCodedExtent.width, caps.maxCodedExtent.width);
  m_extent.height =
      clamp_u32(align_up(height, caps.pictureAccessGranularity.height),
                caps.minCodedExtent.height, caps.maxCodedExtent.height);
  m_max_dpb_slots = caps.maxDpbSlots;
  m_bitstream_offset_align =
      nx::max<u64>(1, caps.minBitstreamBufferOffsetAlignment);
  m_bitstream_size_align =
      nx::max<u64>(1, caps.minBitstreamBufferSizeAlignment);

  VkVideoSessionCreateInfoKHR create{
      VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR};
  create.queueFamilyIndex = device.video_decode_family();
  create.pVideoProfile = &profile;
  create.pictureFormat = m_picture_format;
  create.maxCodedExtent = m_extent;
  create.referencePictureFormat = m_dpb_format;
  create.maxDpbSlots = caps.maxDpbSlots;
  create.maxActiveReferencePictures = caps.maxActiveReferencePictures;
  create.pStdHeaderVersion = &caps.stdHeaderVersion;
  if (const VkResult r = vkCreateVideoSessionKHR(dev, &create, nullptr,
                                                 &m_session);
      r != VK_SUCCESS) {
    nx::logw(log_video, "video: vkCreateVideoSessionKHR failed ({})",
             nx::cast<i32>(r));
    m_session = VK_NULL_HANDLE;
    m_device = nullptr;
    return false;
  }

  // The session needs backing memory, handed over in one or more binds the
  // driver names by index.
  u32 requirement_count = 0;
  vkGetVideoSessionMemoryRequirementsKHR(dev, m_session, &requirement_count,
                                         nullptr);
  nx::small_vector<VkVideoSessionMemoryRequirementsKHR, 4> requirements(
      requirement_count);
  for (VkVideoSessionMemoryRequirementsKHR &r : requirements)
    r.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR;
  vkGetVideoSessionMemoryRequirementsKHR(dev, m_session, &requirement_count,
                                         requirements.data());

  nx::small_vector<VkBindVideoSessionMemoryInfoKHR, 4> binds(requirement_count);
  for (u32 i = 0; i < requirement_count; ++i) {
    const VkMemoryRequirements &mr = requirements[i].memoryRequirements;
    const u32 type = session_memory_type(phys, mr.memoryTypeBits);
    if (type == ~0u) {
      nx::logw(log_video, "video: no memory type for the session");
      shutdown();
      return false;
    }
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = mr.size;
    allocate.memoryTypeIndex = type;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (const VkResult r = vkAllocateMemory(dev, &allocate, nullptr, &memory);
        r != VK_SUCCESS) {
      nx::logw(log_video, "video: session memory alloc failed ({})",
               nx::cast<i32>(r));
      shutdown();
      return false;
    }
    m_memories.push_back(memory);

    binds[i].sType = VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR;
    binds[i].memoryBindIndex = requirements[i].memoryBindIndex;
    binds[i].memory = memory;
    binds[i].memoryOffset = 0;
    binds[i].memorySize = mr.size;
  }
  if (const VkResult r = vkBindVideoSessionMemoryKHR(
          dev, m_session, requirement_count, binds.data());
      r != VK_SUCCESS) {
    nx::logw(log_video, "video: bind session memory failed ({})",
             nx::cast<i32>(r));
    shutdown();
    return false;
  }

  // AV1 carries its sequence header in session parameters; VP9 needs none.
  if (codec == VideoCodec::AV1 && !create_session_parameters()) {
    shutdown();
    return false;
  }

  // TRANSFER_SRC so a decoded picture can be copied back out for verification;
  // the images are shared with the graphics family for that copy.
  const u32 families[2] = {device.video_decode_family(),
                           device.graphics_family()};
  const VkImageUsageFlags dpb_usage =
      (coincide ? (VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR |
                   VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR)
                : VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR) |
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  m_dpb.resize(m_max_dpb_slots);
  for (VideoImage &slot : m_dpb) {
    if (!create_video_image(dev, phys, profiles, m_dpb_format, m_extent,
                            dpb_usage, families, 2, slot)) {
      nx::logw(log_video, "video: dpb image allocation failed");
      shutdown();
      return false;
    }
  }
  if (!coincide &&
      !create_video_image(dev, phys, profiles, m_picture_format, m_extent,
                          VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR |
                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                          families, 2, m_output)) {
    nx::logw(log_video, "video: output image allocation failed");
    shutdown();
    return false;
  }

  if (!create_command_resources() ||
      !create_bitstream_buffer(size_t{1} << 20, profiles) ||
      !create_sampleable_image(families, 2) || !create_plane_resources()) {
    shutdown();
    return false;
  }

  nx::logi(log_video, "video: decode session {}x{}, {} dpb slots", m_extent.width,
           m_extent.height, m_max_dpb_slots);
  return true;
}

bool VideoDecoder::create_command_resources() {
  const VkDevice dev = m_device->vk_device();
  const u32 video_family = m_device->video_decode_family();

  u32 count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(m_device->vk_physical_device(),
                                           &count, nullptr);
  nx::small_vector<VkQueueFamilyProperties, 8> props(count);
  vkGetPhysicalDeviceQueueFamilyProperties(m_device->vk_physical_device(),
                                           &count, props.data());
  m_video_supports_transfer =
      video_family < count &&
      (props[video_family].queueFlags & VK_QUEUE_TRANSFER_BIT) != 0;

  VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool.queueFamilyIndex = video_family;
  if (vkCreateCommandPool(dev, &pool, nullptr, &m_pool) != VK_SUCCESS)
    return false;

  VkCommandBufferAllocateInfo alloc{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  alloc.commandPool = m_pool;
  alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(dev, &alloc, &m_cmd) != VK_SUCCESS)
    return false;

  VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  return vkCreateFence(dev, &fence, nullptr, &m_fence) == VK_SUCCESS;
}

bool VideoDecoder::create_session_parameters() {
  VkVideoDecodeAV1SessionParametersCreateInfoKHR av1{
      VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_SESSION_PARAMETERS_CREATE_INFO_KHR};
  av1.pStdSequenceHeader = &m_av1_seq.header;
  VkVideoSessionParametersCreateInfoKHR create{
      VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR};
  create.pNext = &av1;
  create.videoSession = m_session;
  if (const VkResult r = vkCreateVideoSessionParametersKHR(
          m_device->vk_device(), &create, nullptr, &m_session_params);
      r != VK_SUCCESS) {
    nx::logw(log_video, "video: session parameters failed ({})",
             nx::cast<i32>(r));
    return false;
  }
  return true;
}

bool VideoDecoder::create_bitstream_buffer(
    const u64 capacity, const VkVideoProfileListInfoKHR &profiles) {
  m_bitstream_capacity = align_up64(capacity, m_bitstream_size_align);
  return create_host_buffer(m_device->vk_device(),
                            m_device->vk_physical_device(), m_bitstream_capacity,
                            VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR, &profiles,
                            m_bitstream, m_bitstream_memory, m_bitstream_mapped);
}

u32 VideoDecoder::allocate_slot() noexcept {
  // A slot no live VP9 reference buffer points at, cycling so references stay
  // put. With more slots than the eight VP9 buffers, one is always free.
  for (u32 tries = 0; tries < m_max_dpb_slots; ++tries) {
    const u32 slot = m_next_slot;
    m_next_slot = (m_next_slot + 1) % m_max_dpb_slots;
    bool referenced = false;
    for (u32 i = 0; i < 8; ++i)
      if (m_ref_valid[i] && m_ref_to_dpb[i] == slot) {
        referenced = true;
        break;
      }
    if (!referenced)
      return slot;
  }
  return m_next_slot;
}

bool VideoDecoder::decode(const u8 *const frame, const u32 size,
                          DecodedPicture &out) {
  out = DecodedPicture{};
  if (!ready() || frame == nullptr || size == 0)
    return false;
  return m_codec == VideoCodec::AV1 ? decode_av1(frame, size, out)
                                    : decode_vp9(frame, size, out);
}

bool VideoDecoder::decode_vp9(const u8 *const frame, const u32 size,
                              DecodedPicture &out) {
  u32 ref_w[8];
  u32 ref_h[8];
  for (u32 i = 0; i < 8; ++i) {
    const bool live = m_ref_valid[i];
    ref_w[i] = live ? m_slot_width[m_ref_to_dpb[i]] : 0;
    ref_h[i] = live ? m_slot_height[m_ref_to_dpb[i]] : 0;
  }

  Vp9FrameHeader hdr;
  if (!parse_vp9_frame_header(frame, size, ref_w, ref_h, hdr))
    return false;

  // A show-existing frame decodes nothing; it re-presents a buffer already held.
  if (hdr.show_existing_frame) {
    const u32 buf = hdr.frame_to_show_map_idx;
    if (buf >= 8 || !m_ref_valid[buf])
      return false;
    out.ok = true;
    out.show = true;
    out.slot = m_ref_to_dpb[buf];
    out.width = m_slot_width[out.slot];
    out.height = m_slot_height[out.slot];
    return true;
  }

  const u32 coded_w = hdr.frame_width != 0 ? hdr.frame_width : m_extent.width;
  const u32 coded_h = hdr.frame_height != 0 ? hdr.frame_height : m_extent.height;

  const u64 range = align_up64(size, m_bitstream_size_align);
  if (range > m_bitstream_capacity)
    return false;
  std::memcpy(m_bitstream_mapped, frame, size);
  if (range > size)
    std::memset(static_cast<u8 *>(m_bitstream_mapped) + size, 0,
                nx::cast<usize>(range - size));

  const u32 setup = allocate_slot();

  hdr.picture.pColorConfig = &hdr.color;
  hdr.picture.pLoopFilter = &hdr.loop_filter;
  hdr.picture.pSegmentation = &hdr.segmentation;

  VkVideoDecodeVP9PictureInfoKHR vp9pic{
      VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_PICTURE_INFO_KHR};
  vp9pic.pStdPictureInfo = &hdr.picture;
  for (u32 k = 0; k < VK_MAX_VIDEO_VP9_REFERENCES_PER_FRAME_KHR; ++k)
    vp9pic.referenceNameSlotIndices[k] = -1;
  const bool inter =
      hdr.picture.frame_type == STD_VIDEO_VP9_FRAME_TYPE_NON_KEY &&
      hdr.picture.flags.intra_only == 0;
  if (inter)
    for (u32 k = 0; k < VK_MAX_VIDEO_VP9_REFERENCES_PER_FRAME_KHR; ++k) {
      const u32 buf = hdr.ref_frame_idx[k];
      if (buf < 8 && m_ref_valid[buf])
        vp9pic.referenceNameSlotIndices[k] =
            nx::cast<i32>(m_ref_to_dpb[buf]);
    }
  vp9pic.uncompressedHeaderOffset = 0;
  vp9pic.compressedHeaderOffset = hdr.uncompressed_header_size;
  vp9pic.tilesOffset = hdr.uncompressed_header_size + hdr.compressed_header_size;

  VkVideoPictureResourceInfoKHR setup_res{
      VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR};
  setup_res.codedOffset = {0, 0};
  setup_res.codedExtent = {coded_w, coded_h};
  setup_res.baseArrayLayer = 0;
  setup_res.imageViewBinding = m_dpb[setup].view;

  // The unique DPB slots this frame reads from, each as a reference resource.
  VkVideoPictureResourceInfoKHR ref_res[8];
  VkVideoReferenceSlotInfoKHR decode_refs[8];
  u32 unique[8];
  u32 ref_count = 0;
  for (u32 k = 0; k < VK_MAX_VIDEO_VP9_REFERENCES_PER_FRAME_KHR; ++k) {
    const i32 s = vp9pic.referenceNameSlotIndices[k];
    if (s < 0)
      continue;
    bool seen = false;
    for (u32 j = 0; j < ref_count; ++j)
      if (unique[j] == nx::cast<u32>(s))
        seen = true;
    if (seen)
      continue;
    unique[ref_count] = nx::cast<u32>(s);
    ref_res[ref_count] = VkVideoPictureResourceInfoKHR{
        VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR};
    ref_res[ref_count].codedOffset = {0, 0};
    ref_res[ref_count].codedExtent = {m_slot_width[nx::cast<u32>(s)],
                                      m_slot_height[nx::cast<u32>(s)]};
    ref_res[ref_count].baseArrayLayer = 0;
    ref_res[ref_count].imageViewBinding = m_dpb[nx::cast<u32>(s)].view;
    decode_refs[ref_count] = VkVideoReferenceSlotInfoKHR{
        VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR};
    decode_refs[ref_count].slotIndex = s;
    decode_refs[ref_count].pPictureResource = &ref_res[ref_count];
    ++ref_count;
  }

  // Begin lists every active slot: the references, plus the setup slot as -1 to
  // say it is being activated by this decode.
  VkVideoReferenceSlotInfoKHR begin_slots[9];
  for (u32 j = 0; j < ref_count; ++j)
    begin_slots[j] = decode_refs[j];
  begin_slots[ref_count] = VkVideoReferenceSlotInfoKHR{
      VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR};
  begin_slots[ref_count].slotIndex = -1;
  begin_slots[ref_count].pPictureResource = &setup_res;

  VkVideoReferenceSlotInfoKHR setup_slot{
      VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR};
  setup_slot.slotIndex = nx::cast<i32>(setup);
  setup_slot.pPictureResource = &setup_res;

  VkVideoPictureResourceInfoKHR dst_res = setup_res;
  if (!m_coincide)
    dst_res.imageViewBinding = m_output.view;

  VkVideoDecodeInfoKHR di{VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR};
  di.pNext = &vp9pic;
  di.srcBuffer = m_bitstream;
  di.srcBufferOffset = 0;
  di.srcBufferRange = range;
  di.dstPictureResource = dst_res;
  di.pSetupReferenceSlot = &setup_slot;
  di.referenceSlotCount = ref_count;
  di.pReferenceSlots = ref_count != 0 ? decode_refs : nullptr;

  DecodeSubmit sub;
  sub.info = &di;
  sub.begin_slots = begin_slots;
  sub.begin_slot_count = ref_count + 1;
  sub.setup = setup;
  sub.unique_refs = unique;
  sub.unique_ref_count = ref_count;
  sub.coded_w = coded_w;
  sub.coded_h = coded_h;
  sub.refresh_frame_flags = hdr.picture.refresh_frame_flags;
  if (!submit_decode(sub))
    return false;

  out.ok = true;
  out.slot = setup;
  out.show = hdr.picture.flags.show_frame != 0;
  out.width = coded_w;
  out.height = coded_h;
  return true;
}

bool VideoDecoder::submit_decode(const DecodeSubmit &s) {
  const VkDevice dev = m_device->vk_device();
  vkResetCommandBuffer(m_cmd, 0);
  VkCommandBufferBeginInfo begin_cmd{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_cmd.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(m_cmd, &begin_cmd);

  // Bring every image the decode touches into its decode layout. References are
  // already there from earlier decodes and skip the barrier.
  VkImageMemoryBarrier2 barriers[10];
  u32 barrier_count = 0;
  const auto to_layout = [&](VideoImage &img, const VkImageLayout layout) {
    if (img.layout == layout)
      return;
    VkImageMemoryBarrier2 &b = barriers[barrier_count++];
    b = VkImageMemoryBarrier2{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
    b.srcAccessMask = 0;
    b.dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR;
    b.dstAccessMask = VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR |
                      VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR;
    b.oldLayout = img.layout;
    b.newLayout = layout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img.image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    img.layout = layout;
  };
  to_layout(m_dpb[s.setup], VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
  if (!m_coincide)
    to_layout(m_output, VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR);
  for (u32 j = 0; j < s.unique_ref_count; ++j)
    to_layout(m_dpb[s.unique_refs[j]], VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
  if (barrier_count != 0) {
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = barrier_count;
    dep.pImageMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(m_cmd, &dep);
  }

  VkVideoBeginCodingInfoKHR begin{VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR};
  begin.videoSession = m_session;
  begin.videoSessionParameters = m_session_params; // null for VP9, set for AV1
  begin.referenceSlotCount = s.begin_slot_count;
  begin.pReferenceSlots = s.begin_slots;
  vkCmdBeginVideoCodingKHR(m_cmd, &begin);

  if (m_needs_reset) {
    VkVideoCodingControlInfoKHR control{
        VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR};
    control.flags = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR;
    vkCmdControlVideoCodingKHR(m_cmd, &control);
    m_needs_reset = false;
  }

  vkCmdDecodeVideoKHR(m_cmd, s.info);

  VkVideoEndCodingInfoKHR end{VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR};
  vkCmdEndVideoCodingKHR(m_cmd, &end);
  vkEndCommandBuffer(m_cmd);

  vkResetFences(dev, 1, &m_fence);
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &m_cmd;
  if (vkQueueSubmit(m_device->video_decode_queue(), 1, &submit, m_fence) !=
      VK_SUCCESS)
    return false;
  vkWaitForFences(dev, 1, &m_fence, VK_TRUE, ~0ull);

  m_slot_width[s.setup] = s.coded_w;
  m_slot_height[s.setup] = s.coded_h;
  for (u32 i = 0; i < 8; ++i)
    if (s.refresh_frame_flags & (1u << i)) {
      m_ref_to_dpb[i] = s.setup;
      m_ref_valid[i] = true;
    }
  return true;
}

bool VideoDecoder::decode_av1(const u8 *const frame, const u32 size,
                              DecodedPicture &out) {
  Av1FrameHeader hdr;
  if (!parse_av1_frame_header(frame, size, m_av1_seq, m_av1_refs, hdr))
    return false;

  // A show-existing frame re-presents a decoded buffer, decoding nothing.
  if (hdr.show_existing_frame) {
    const u32 buf = hdr.frame_to_show_map_idx;
    if (buf >= 8 || !m_ref_valid[buf])
      return false;
    out.ok = true;
    out.show = true;
    out.slot = m_ref_to_dpb[buf];
    out.width = m_slot_width[out.slot];
    out.height = m_slot_height[out.slot];
    return true;
  }

  if (hdr.tile_count == 0 || hdr.tile_count > Av1FrameHeader::kMaxTiles)
    return false;
  const u32 coded_w = hdr.frame_width != 0 ? hdr.frame_width : m_extent.width;
  const u32 coded_h = hdr.frame_height != 0 ? hdr.frame_height : m_extent.height;

  // The decode buffer holds only the tile data: the driver reads the frame
  // header from pStdPictureInfo, never from the bitstream, so frameHeaderOffset
  // is 0 and the tiles are concatenated with their offsets pointing into this
  // buffer. (Same layout FFmpeg's vulkan_av1 feeds.)
  u32 buf_tile_offsets[Av1FrameHeader::kMaxTiles];
  u32 dst = 0;
  for (u32 t = 0; t < hdr.tile_count; ++t) {
    dst = nx::cast<u32>(align_up64(dst, m_bitstream_offset_align));
    const u32 tsz = hdr.tile_sizes[t];
    if (nx::cast<u64>(dst) + tsz > m_bitstream_capacity)
      return false;
    std::memcpy(static_cast<u8 *>(m_bitstream_mapped) + dst,
                frame + hdr.tile_offsets[t], tsz);
    buf_tile_offsets[t] = dst;
    dst += tsz;
  }
  const u64 range = align_up64(dst, m_bitstream_size_align);
  if (range > m_bitstream_capacity)
    return false;
  if (range > dst)
    std::memset(static_cast<u8 *>(m_bitstream_mapped) + dst, 0,
                nx::cast<usize>(range - dst));

  const u32 setup = allocate_slot();

  VkVideoDecodeAV1PictureInfoKHR av1pic{
      VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PICTURE_INFO_KHR};
  av1pic.pStdPictureInfo = &hdr.picture;
  for (u32 k = 0; k < VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR; ++k)
    av1pic.referenceNameSlotIndices[k] = -1;
  if (!hdr.frame_is_intra)
    for (u32 k = 0; k < VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR; ++k) {
      const u32 buf = hdr.ref_frame_idx[k];
      if (buf < 8 && m_ref_valid[buf])
        av1pic.referenceNameSlotIndices[k] = nx::cast<i32>(m_ref_to_dpb[buf]);
    }
  av1pic.frameHeaderOffset = 0;
  av1pic.tileCount = hdr.tile_count;
  av1pic.pTileOffsets = buf_tile_offsets;
  av1pic.pTileSizes = hdr.tile_sizes;

  // The setup slot's reference info: what this frame contributes when a later
  // frame references it. Saved into m_slot_ref_info[setup] after the decode.
  StdVideoDecodeAV1ReferenceInfo setup_info{};
  setup_info.flags.disable_frame_end_update_cdf =
      hdr.picture.flags.disable_frame_end_update_cdf;
  setup_info.flags.segmentation_enabled =
      hdr.picture.flags.segmentation_enabled;
  setup_info.frame_type = nx::cast<u8>(hdr.picture.frame_type);
  setup_info.OrderHint = hdr.picture.OrderHint;
  for (u32 j = 0; j < STD_VIDEO_AV1_NUM_REF_FRAMES; ++j)
    setup_info.SavedOrderHints[j] = nx::cast<u8>(hdr.picture.OrderHints[j]);

  VkVideoDecodeAV1DpbSlotInfoKHR setup_dpb{
      VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_DPB_SLOT_INFO_KHR};
  setup_dpb.pStdReferenceInfo = &setup_info;

  VkVideoPictureResourceInfoKHR setup_res{
      VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR};
  setup_res.codedOffset = {0, 0};
  setup_res.codedExtent = {coded_w, coded_h};
  setup_res.baseArrayLayer = 0;
  setup_res.imageViewBinding = m_dpb[setup].view;

  // Each unique DPB slot this frame reads, with its saved reference info.
  VkVideoPictureResourceInfoKHR ref_res[VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR];
  VkVideoDecodeAV1DpbSlotInfoKHR ref_dpb[VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR];
  VkVideoReferenceSlotInfoKHR decode_refs[VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR];
  u32 unique[VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR];
  u32 ref_count = 0;
  for (u32 k = 0; k < VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR; ++k) {
    const i32 sidx = av1pic.referenceNameSlotIndices[k];
    if (sidx < 0)
      continue;
    bool seen = false;
    for (u32 j = 0; j < ref_count; ++j)
      if (unique[j] == nx::cast<u32>(sidx))
        seen = true;
    if (seen)
      continue;
    const u32 sl = nx::cast<u32>(sidx);
    unique[ref_count] = sl;
    ref_res[ref_count] = VkVideoPictureResourceInfoKHR{
        VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR};
    ref_res[ref_count].codedOffset = {0, 0};
    ref_res[ref_count].codedExtent = {m_slot_width[sl], m_slot_height[sl]};
    ref_res[ref_count].baseArrayLayer = 0;
    ref_res[ref_count].imageViewBinding = m_dpb[sl].view;
    ref_dpb[ref_count] = VkVideoDecodeAV1DpbSlotInfoKHR{
        VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_DPB_SLOT_INFO_KHR};
    ref_dpb[ref_count].pStdReferenceInfo = &m_slot_ref_info[sl];
    decode_refs[ref_count] = VkVideoReferenceSlotInfoKHR{
        VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR};
    decode_refs[ref_count].pNext = &ref_dpb[ref_count];
    decode_refs[ref_count].slotIndex = sidx;
    decode_refs[ref_count].pPictureResource = &ref_res[ref_count];
    ++ref_count;
  }

  VkVideoReferenceSlotInfoKHR begin_slots[VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR + 1];
  for (u32 j = 0; j < ref_count; ++j)
    begin_slots[j] = decode_refs[j];
  begin_slots[ref_count] = VkVideoReferenceSlotInfoKHR{
      VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR};
  begin_slots[ref_count].pNext = &setup_dpb;
  begin_slots[ref_count].slotIndex = -1;
  begin_slots[ref_count].pPictureResource = &setup_res;

  VkVideoReferenceSlotInfoKHR setup_slot{
      VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR};
  setup_slot.pNext = &setup_dpb;
  setup_slot.slotIndex = nx::cast<i32>(setup);
  setup_slot.pPictureResource = &setup_res;

  VkVideoPictureResourceInfoKHR dst_res = setup_res;
  if (!m_coincide)
    dst_res.imageViewBinding = m_output.view;

  VkVideoDecodeInfoKHR di{VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR};
  di.pNext = &av1pic;
  di.srcBuffer = m_bitstream;
  di.srcBufferOffset = 0;
  di.srcBufferRange = range;
  di.dstPictureResource = dst_res;
  di.pSetupReferenceSlot = &setup_slot;
  di.referenceSlotCount = ref_count;
  di.pReferenceSlots = ref_count != 0 ? decode_refs : nullptr;

  DecodeSubmit sub;
  sub.info = &di;
  sub.begin_slots = begin_slots;
  sub.begin_slot_count = ref_count + 1;
  sub.setup = setup;
  sub.unique_refs = unique;
  sub.unique_ref_count = ref_count;
  sub.coded_w = coded_w;
  sub.coded_h = coded_h;
  sub.refresh_frame_flags = hdr.picture.refresh_frame_flags;
  if (!submit_decode(sub))
    return false;

  m_slot_ref_info[setup] = setup_info;
  update_av1_refs(hdr, setup);

  out.ok = true;
  out.slot = setup;
  out.show = hdr.show_frame;
  out.width = coded_w;
  out.height = coded_h;
  return true;
}

void VideoDecoder::update_av1_refs(const Av1FrameHeader &hdr, const u32 setup) {
  (void)setup;
  for (u32 i = 0; i < STD_VIDEO_AV1_NUM_REF_FRAMES; ++i) {
    if ((hdr.picture.refresh_frame_flags & (1u << i)) == 0)
      continue;
    m_av1_refs.valid[i] = true;
    m_av1_refs.frame_type[i] = nx::cast<u8>(hdr.picture.frame_type);
    m_av1_refs.frame_id[i] = hdr.picture.current_frame_id;
    m_av1_refs.upscaled_width[i] = hdr.upscaled_width;
    m_av1_refs.frame_width[i] = hdr.frame_width;
    m_av1_refs.frame_height[i] = hdr.frame_height;
    m_av1_refs.render_width[i] = hdr.render_width;
    m_av1_refs.render_height[i] = hdr.render_height;
    m_av1_refs.order_hint[i] = nx::cast<u8>(hdr.picture.OrderHint);
    for (u32 j = 0; j < STD_VIDEO_AV1_NUM_REF_FRAMES; ++j)
      m_av1_refs.saved_order_hints[i][j] =
          nx::cast<u8>(hdr.picture.OrderHints[j]);
    m_av1_refs.loop_filter[i] = hdr.loop_filter;
    m_av1_refs.segmentation[i] = hdr.segmentation;
    for (u32 r = 0; r < STD_VIDEO_AV1_NUM_REF_FRAMES; ++r)
      for (u32 p = 0; p < STD_VIDEO_AV1_GLOBAL_MOTION_PARAMS; ++p)
        m_av1_refs.gm_params[i][r][p] = hdr.global_motion.gm_params[r][p];
  }
}

bool VideoDecoder::create_sampleable_image(const u32 *const families,
                                           const u32 family_count) {
  const VkDevice dev = m_device->vk_device();
  VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.format = m_picture_format;
  ci.extent = {m_extent.width, m_extent.height, 1};
  ci.mipLevels = 1;
  ci.arrayLayers = 1;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
             VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  if (family_count > 1) {
    ci.sharingMode = VK_SHARING_MODE_CONCURRENT;
    ci.queueFamilyIndexCount = family_count;
    ci.pQueueFamilyIndices = families;
  } else {
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (vkCreateImage(dev, &ci, nullptr, &m_sampleable.image) != VK_SUCCESS)
    return false;

  VkMemoryRequirements mr;
  vkGetImageMemoryRequirements(dev, m_sampleable.image, &mr);
  const u32 type = session_memory_type(m_device->vk_physical_device(),
                                       mr.memoryTypeBits);
  if (type == ~0u)
    return false;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = type;
  if (vkAllocateMemory(dev, &ai, nullptr, &m_sampleable.memory) != VK_SUCCESS)
    return false;
  return vkBindImageMemory(dev, m_sampleable.image, m_sampleable.memory, 0) ==
         VK_SUCCESS;
}

bool VideoDecoder::present(const u32 slot) {
  if (!ready() || !m_video_supports_transfer)
    return false;
  VideoImage &src = m_coincide ? m_dpb[slot] : m_output;
  if (m_coincide && slot >= m_dpb.size())
    return false;
  const VkImageLayout restore = m_coincide
                                    ? VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR
                                    : VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR;

  const VkDevice dev = m_device->vk_device();
  vkResetCommandBuffer(m_cmd, 0);
  VkCommandBufferBeginInfo begin_cmd{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_cmd.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(m_cmd, &begin_cmd);

  const auto barrier = [&](VideoImage &img, const VkImageLayout to) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.srcAccessMask =
        VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask =
        VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.oldLayout = img.layout;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img.image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    img.layout = to;
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(m_cmd, &dep);
  };

  barrier(src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  barrier(m_sampleable, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

  // Both planes: full-res luma, then half-res interleaved chroma.
  VkImageCopy2 regions[2];
  regions[0] = VkImageCopy2{VK_STRUCTURE_TYPE_IMAGE_COPY_2};
  regions[0].srcSubresource = {VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1};
  regions[0].dstSubresource = {VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1};
  regions[0].extent = {m_extent.width, m_extent.height, 1};
  regions[1] = VkImageCopy2{VK_STRUCTURE_TYPE_IMAGE_COPY_2};
  regions[1].srcSubresource = {VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1};
  regions[1].dstSubresource = {VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1};
  regions[1].extent = {m_extent.width / 2, m_extent.height / 2, 1};
  VkCopyImageInfo2 copy{VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2};
  copy.srcImage = src.image;
  copy.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  copy.dstImage = m_sampleable.image;
  copy.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  copy.regionCount = 2;
  copy.pRegions = regions;
  vkCmdCopyImage2(m_cmd, &copy);

  barrier(src, restore);
  barrier(m_sampleable, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  vkEndCommandBuffer(m_cmd);

  vkResetFences(dev, 1, &m_fence);
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &m_cmd;
  if (vkQueueSubmit(m_device->video_decode_queue(), 1, &submit, m_fence) !=
      VK_SUCCESS)
    return false;
  vkWaitForFences(dev, 1, &m_fence, VK_TRUE, ~0ull);
  return true;
}

bool VideoDecoder::create_plane_resources() {
  const VkDevice dev = m_device->vk_device();

  // Command plumbing for the graphics-queue plane resolve.
  VkCommandPoolCreateInfo cpc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpc.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  cpc.queueFamilyIndex = m_device->graphics_family();
  if (vkCreateCommandPool(dev, &cpc, nullptr, &m_gfx_pool) != VK_SUCCESS)
    return false;
  VkCommandBufferAllocateInfo cba{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cba.commandPool = m_gfx_pool;
  cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cba.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(dev, &cba, &m_gfx_cmd) != VK_SUCCESS)
    return false;
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (vkCreateFence(dev, &fci, nullptr, &m_gfx_fence) != VK_SUCCESS)
    return false;

  // The two sampled planes the caller draws: full-res luma, half-res interleaved
  // chroma. Ordinary device textures, so they carry bindless slots for free.
  TextureDesc luma;
  luma.name = "video luma";
  luma.format = Format::R8_UNORM;
  luma.width = m_extent.width;
  luma.height = m_extent.height;
  luma.usage =
      TextureUsage::Sampled | TextureUsage::CopyDst | TextureUsage::CopySrc;
  m_y = m_device->create_texture(luma);

  TextureDesc chroma;
  chroma.name = "video chroma";
  chroma.format = Format::RG8_UNORM;
  chroma.width = m_extent.width / 2;
  chroma.height = m_extent.height / 2;
  chroma.usage =
      TextureUsage::Sampled | TextureUsage::CopyDst | TextureUsage::CopySrc;
  m_cbcr = m_device->create_texture(chroma);

  return m_y.valid() && m_cbcr.valid();
}

bool VideoDecoder::decode_frame(const u8 *const frame, const u32 size,
                                bool &shown, u32 &width, u32 &height) {
  DecodedPicture pic;
  if (!decode(frame, size, pic)) {
    m_have_last = false;
    return false;
  }
  m_last_slot = pic.slot;
  m_have_last = true;
  shown = pic.show;
  width = pic.width;
  height = pic.height;
  return true;
}

void VideoDecoder::reset_stream() noexcept {
  for (u32 i = 0; i < 8; ++i)
    m_ref_valid[i] = false;
  m_av1_refs = Av1RefState{};
  m_have_last = false;
  m_needs_reset = true;
}

bool VideoDecoder::show_last() {
  // present() copied the picture into the sampleable NV12 image on the decode
  // queue. Now resolve its two planes into the bindless luma/chroma textures on
  // the graphics queue, where a caller can sample them.
  if (!m_have_last || !present(m_last_slot))
    return false;

  VulkanTexture *const y = m_device->texture_mut(m_y);
  VulkanTexture *const c = m_device->texture_mut(m_cbcr);
  if (y == nullptr || c == nullptr)
    return false;
  const VkDevice dev = m_device->vk_device();

  vkResetCommandBuffer(m_gfx_cmd, 0);
  VkCommandBufferBeginInfo begin_cmd{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_cmd.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(m_gfx_cmd, &begin_cmd);

  const auto barrier = [&](const VkImage image, const VkImageLayout from,
                           const VkImageLayout to) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.srcAccessMask =
        VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask =
        VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo d{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    d.imageMemoryBarrierCount = 1;
    d.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(m_gfx_cmd, &d);
  };

  // The sampleable image is concurrent, so this queue can read it; the plane
  // textures are fully overwritten, so their old contents need no preserving.
  barrier(m_sampleable.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  barrier(y->image, VK_IMAGE_LAYOUT_UNDEFINED,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  barrier(c->image, VK_IMAGE_LAYOUT_UNDEFINED,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

  const auto copy_plane = [&](const VkImageAspectFlagBits src_aspect,
                              const VkImage dst, const u32 w, const u32 h) {
    VkImageCopy2 region{VK_STRUCTURE_TYPE_IMAGE_COPY_2};
    region.srcSubresource = {nx::cast<VkImageAspectFlags>(src_aspect), 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.extent = {w, h, 1};
    VkCopyImageInfo2 info{VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2};
    info.srcImage = m_sampleable.image;
    info.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    info.dstImage = dst;
    info.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    info.regionCount = 1;
    info.pRegions = &region;
    vkCmdCopyImage2(m_gfx_cmd, &info);
  };
  copy_plane(VK_IMAGE_ASPECT_PLANE_0_BIT, y->image, m_extent.width,
             m_extent.height);
  copy_plane(VK_IMAGE_ASPECT_PLANE_1_BIT, c->image, m_extent.width / 2,
             m_extent.height / 2);

  barrier(y->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  barrier(c->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  barrier(m_sampleable.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  vkEndCommandBuffer(m_gfx_cmd);

  vkResetFences(dev, 1, &m_gfx_fence);
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &m_gfx_cmd;
  if (vkQueueSubmit(m_device->graphics_queue(), 1, &submit, m_gfx_fence) !=
      VK_SUCCESS)
    return false;
  vkWaitForFences(dev, 1, &m_gfx_fence, VK_TRUE, ~0ull);

  m_device->set_texture_state(m_y, ResourceState::ShaderReadOnly);
  m_device->set_texture_state(m_cbcr, ResourceState::ShaderReadOnly);
  return true;
}

} // namespace nxe::rhi
