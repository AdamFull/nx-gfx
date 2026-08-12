/**
 * @file vk_video.cpp
 * @brief Vulkan Video: what the device can hardware-decode, and (later) the
 * decode itself. Kept apart from vk_device.cpp because the video queue is its
 * own world of sessions, DPB slots and codec std structs.
 */

#include "core/rendering/rhi/vulkan/vk_common.h"
#include "core/rendering/rhi/vulkan/vk_device.h"
#include "core/rendering/rhi/vulkan/vk_video.h"

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

VideoDecodeCaps VulkanDevice::video_decode_caps(const VideoCodec codec) const {
  VideoDecodeCaps out;
  if (m_physical_device == VK_NULL_HANDLE ||
      vkGetPhysicalDeviceVideoCapabilitiesKHR == nullptr)
    return out;
  if (!extension_offered(m_physical_device, VK_KHR_VIDEO_QUEUE_EXTENSION_NAME))
    return out;

  const char *const codec_extension =
      codec == VideoCodec::AV1 ? VK_KHR_VIDEO_DECODE_AV1_EXTENSION_NAME
                               : VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME;
  if (!extension_offered(m_physical_device, codec_extension))
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

  if (vkGetPhysicalDeviceVideoCapabilitiesKHR(m_physical_device, &profile,
                                              &caps) != VK_SUCCESS)
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
                                      VideoImage &out) {
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
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
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

} // namespace

VideoDecoder::~VideoDecoder() { shutdown(); }

void VideoDecoder::shutdown() noexcept {
  if (m_device != nullptr) {
    const VkDevice dev = m_device->vk_device();
    for (VideoImage &image : m_dpb)
      destroy_video_image(dev, image);
    destroy_video_image(dev, m_output);
    if (m_session != VK_NULL_HANDLE)
      vkDestroyVideoSessionKHR(dev, m_session, nullptr);
    for (const VkDeviceMemory memory : m_memories)
      vkFreeMemory(dev, memory, nullptr);
  }
  m_session = VK_NULL_HANDLE;
  m_memories.clear();
  m_dpb.clear();
  m_output = {};
  m_coincide = false;
  m_device = nullptr;
  m_picture_format = VK_FORMAT_UNDEFINED;
  m_dpb_format = VK_FORMAT_UNDEFINED;
  m_extent = {};
  m_max_dpb_slots = 0;
}

bool VideoDecoder::init(VulkanDevice &device, const VideoCodec codec,
                        const u32 width, const u32 height) {
  shutdown();
  if (!device.video_decode_available() || codec != VideoCodec::VP9)
    return false; // only VP9 is wired up so far

  m_device = &device;
  const VkPhysicalDevice phys = device.vk_physical_device();
  const VkDevice dev = device.vk_device();

  VkVideoDecodeVP9ProfileInfoKHR vp9{
      VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_PROFILE_INFO_KHR};
  vp9.stdProfile = STD_VIDEO_VP9_PROFILE_0;
  VkVideoProfileInfoKHR profile{VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR};
  profile.pNext = &vp9;
  profile.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR;
  profile.chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
  profile.lumaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
  profile.chromaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;

  VkVideoDecodeVP9CapabilitiesKHR vp9_caps{
      VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_CAPABILITIES_KHR};
  VkVideoDecodeCapabilitiesKHR decode_caps{
      VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR};
  decode_caps.pNext = &vp9_caps;
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

  const VkImageUsageFlags dpb_usage =
      coincide ? (VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR |
                  VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR)
               : VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR;
  m_dpb.resize(m_max_dpb_slots);
  for (VideoImage &slot : m_dpb) {
    if (!create_video_image(dev, phys, profiles, m_dpb_format, m_extent,
                            dpb_usage, slot)) {
      nx::logw(log_video, "video: dpb image allocation failed");
      shutdown();
      return false;
    }
  }
  if (!coincide && !create_video_image(dev, phys, profiles, m_picture_format,
                                       m_extent,
                                       VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR,
                                       m_output)) {
    nx::logw(log_video, "video: output image allocation failed");
    shutdown();
    return false;
  }

  nx::logi(log_video, "video: decode session {}x{}, {} dpb slots", m_extent.width,
           m_extent.height, m_max_dpb_slots);
  return true;
}

} // namespace nxe::rhi
