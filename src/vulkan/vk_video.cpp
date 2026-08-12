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

#include "core/rendering/rhi/vulkan/vk_video_convert_spv.inc"

[[nodiscard]] VkShaderModule create_shader_module(const VkDevice dev,
                                                  const u32 *const code,
                                                  const usize bytes) {
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = bytes;
  ci.pCode = code;
  VkShaderModule module = VK_NULL_HANDLE;
  if (vkCreateShaderModule(dev, &ci, nullptr, &module) != VK_SUCCESS)
    return VK_NULL_HANDLE;
  return module;
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

    // Conversion pass, dependents before the conversion object they reference.
    if (m_pipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(dev, m_pipeline, nullptr);
    if (m_pipe_layout != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(dev, m_pipe_layout, nullptr);
    if (m_desc_pool != VK_NULL_HANDLE)
      vkDestroyDescriptorPool(dev, m_desc_pool, nullptr);
    if (m_set_layout != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(dev, m_set_layout, nullptr);
    if (m_ycbcr_sampler != VK_NULL_HANDLE)
      vkDestroySampler(dev, m_ycbcr_sampler, nullptr);
    if (m_rgba_handle.valid())
      m_device->destroy_texture(m_rgba_handle);
    destroy_video_image(dev, m_sampleable); // its view uses the conversion
    if (m_ycbcr != VK_NULL_HANDLE)
      vkDestroySamplerYcbcrConversion(dev, m_ycbcr, nullptr);
    if (m_gfx_fence != VK_NULL_HANDLE)
      vkDestroyFence(dev, m_gfx_fence, nullptr);
    if (m_gfx_pool != VK_NULL_HANDLE)
      vkDestroyCommandPool(dev, m_gfx_pool, nullptr);

    if (m_bitstream != VK_NULL_HANDLE)
      vkDestroyBuffer(dev, m_bitstream, nullptr);
    if (m_bitstream_memory != VK_NULL_HANDLE)
      vkFreeMemory(dev, m_bitstream_memory, nullptr);
    if (m_readback != VK_NULL_HANDLE)
      vkDestroyBuffer(dev, m_readback, nullptr);
    if (m_readback_memory != VK_NULL_HANDLE)
      vkFreeMemory(dev, m_readback_memory, nullptr);
    if (m_fence != VK_NULL_HANDLE)
      vkDestroyFence(dev, m_fence, nullptr);
    if (m_pool != VK_NULL_HANDLE)
      vkDestroyCommandPool(dev, m_pool, nullptr);
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
  m_readback = VK_NULL_HANDLE;
  m_readback_memory = VK_NULL_HANDLE;
  m_readback_mapped = nullptr;
  m_readback_capacity = 0;
  for (u32 i = 0; i < 8; ++i)
    m_vp9_valid[i] = false;
  m_next_slot = 0;
  m_gfx_pool = VK_NULL_HANDLE;
  m_gfx_cmd = VK_NULL_HANDLE;
  m_gfx_fence = VK_NULL_HANDLE;
  m_ycbcr = VK_NULL_HANDLE;
  m_ycbcr_sampler = VK_NULL_HANDLE;
  m_set_layout = VK_NULL_HANDLE;
  m_desc_pool = VK_NULL_HANDLE;
  m_set = VK_NULL_HANDLE;
  m_pipe_layout = VK_NULL_HANDLE;
  m_pipeline = VK_NULL_HANDLE;
  m_rgba_handle = {};
  m_rgba_layout = VK_IMAGE_LAYOUT_UNDEFINED;
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
      !create_sampleable_image(families, 2) || !create_convert_resources()) {
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
      if (m_vp9_valid[i] && m_vp9_to_dpb[i] == slot) {
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

  u32 ref_w[8];
  u32 ref_h[8];
  for (u32 i = 0; i < 8; ++i) {
    const bool live = m_vp9_valid[i];
    ref_w[i] = live ? m_slot_width[m_vp9_to_dpb[i]] : 0;
    ref_h[i] = live ? m_slot_height[m_vp9_to_dpb[i]] : 0;
  }

  Vp9FrameHeader hdr;
  if (!parse_vp9_frame_header(frame, size, ref_w, ref_h, hdr))
    return false;

  // A show-existing frame decodes nothing; it re-presents a buffer already held.
  if (hdr.show_existing_frame) {
    const u32 buf = hdr.frame_to_show_map_idx;
    if (buf >= 8 || !m_vp9_valid[buf])
      return false;
    out.ok = true;
    out.show = true;
    out.slot = m_vp9_to_dpb[buf];
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
      if (buf < 8 && m_vp9_valid[buf])
        vp9pic.referenceNameSlotIndices[k] =
            nx::cast<i32>(m_vp9_to_dpb[buf]);
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
  to_layout(m_dpb[setup], VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
  if (!m_coincide)
    to_layout(m_output, VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR);
  for (u32 j = 0; j < ref_count; ++j)
    to_layout(m_dpb[unique[j]], VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR);
  if (barrier_count != 0) {
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = barrier_count;
    dep.pImageMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(m_cmd, &dep);
  }

  VkVideoBeginCodingInfoKHR begin{VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR};
  begin.videoSession = m_session;
  begin.referenceSlotCount = ref_count + 1;
  begin.pReferenceSlots = begin_slots;
  vkCmdBeginVideoCodingKHR(m_cmd, &begin);

  if (m_needs_reset) {
    VkVideoCodingControlInfoKHR control{
        VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR};
    control.flags = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR;
    vkCmdControlVideoCodingKHR(m_cmd, &control);
    m_needs_reset = false;
  }

  vkCmdDecodeVideoKHR(m_cmd, &di);

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

  m_slot_width[setup] = coded_w;
  m_slot_height[setup] = coded_h;
  for (u32 i = 0; i < 8; ++i)
    if (hdr.picture.refresh_frame_flags & (1u << i)) {
      m_vp9_to_dpb[i] = setup;
      m_vp9_valid[i] = true;
    }

  out.ok = true;
  out.slot = setup;
  out.show = hdr.picture.flags.show_frame != 0;
  out.width = coded_w;
  out.height = coded_h;
  return true;
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

bool VideoDecoder::read_image_luma(VideoImage &img,
                                   const VkImageLayout restore_to,
                                   const u32 width, const u32 height,
                                   nx::vector<u8> &out) {
  if (!m_video_supports_transfer) {
    nx::logw(log_video, "video: decode queue has no transfer, cannot read back");
    return false;
  }

  const VkDevice dev = m_device->vk_device();
  const u64 bytes = nx::cast<u64>(width) * height;
  if (bytes > m_readback_capacity) {
    if (m_readback != VK_NULL_HANDLE)
      vkDestroyBuffer(dev, m_readback, nullptr);
    if (m_readback_memory != VK_NULL_HANDLE)
      vkFreeMemory(dev, m_readback_memory, nullptr);
    if (!create_host_buffer(dev, m_device->vk_physical_device(), bytes,
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT, nullptr,
                            m_readback, m_readback_memory, m_readback_mapped))
      return false;
    m_readback_capacity = bytes;
  }

  vkResetCommandBuffer(m_cmd, 0);
  VkCommandBufferBeginInfo begin_cmd{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_cmd.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(m_cmd, &begin_cmd);

  const auto barrier = [&](const VkImageLayout from, const VkImageLayout to) {
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
    b.image = img.image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(m_cmd, &dep);
  };

  barrier(img.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

  VkBufferImageCopy2 region{VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource = {VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1};
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {width, height, 1};
  VkCopyImageToBufferInfo2 copy{VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2};
  copy.srcImage = img.image;
  copy.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  copy.dstBuffer = m_readback;
  copy.regionCount = 1;
  copy.pRegions = &region;
  vkCmdCopyImageToBuffer2(m_cmd, &copy);

  if (restore_to != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
    barrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, restore_to);
  img.layout = restore_to;
  vkEndCommandBuffer(m_cmd);

  vkResetFences(dev, 1, &m_fence);
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &m_cmd;
  if (vkQueueSubmit(m_device->video_decode_queue(), 1, &submit, m_fence) !=
      VK_SUCCESS)
    return false;
  vkWaitForFences(dev, 1, &m_fence, VK_TRUE, ~0ull);

  out.resize(nx::cast<usize>(bytes));
  std::memcpy(out.data(), m_readback_mapped, nx::cast<usize>(bytes));
  return true;
}

bool VideoDecoder::read_luma(const u32 slot, const u32 width, const u32 height,
                             nx::vector<u8> &out) {
  if (!ready() || slot >= m_dpb.size())
    return false;
  return read_image_luma(m_dpb[slot], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         width, height, out);
}

bool VideoDecoder::read_output_luma(const u32 width, const u32 height,
                                    nx::vector<u8> &out) {
  if (!ready())
    return false;
  return read_image_luma(m_sampleable,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, width, height,
                         out);
}

bool VideoDecoder::create_convert_resources() {
  const VkDevice dev = m_device->vk_device();

  VkSamplerYcbcrConversionCreateInfo conv{
      VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO};
  conv.format = m_picture_format;
  conv.ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
  conv.ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;
  conv.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                     VK_COMPONENT_SWIZZLE_IDENTITY,
                     VK_COMPONENT_SWIZZLE_IDENTITY};
  conv.xChromaOffset = VK_CHROMA_LOCATION_COSITED_EVEN;
  conv.yChromaOffset = VK_CHROMA_LOCATION_COSITED_EVEN;
  conv.chromaFilter = VK_FILTER_NEAREST;
  conv.forceExplicitReconstruction = VK_FALSE;
  if (vkCreateSamplerYcbcrConversion(dev, &conv, nullptr, &m_ycbcr) !=
      VK_SUCCESS)
    return false;

  VkSamplerYcbcrConversionInfo conv_info{
      VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO};
  conv_info.conversion = m_ycbcr;

  VkSamplerCreateInfo samp{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samp.pNext = &conv_info;
  samp.magFilter = VK_FILTER_LINEAR;
  samp.minFilter = VK_FILTER_LINEAR;
  samp.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samp.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samp.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samp.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  if (vkCreateSampler(dev, &samp, nullptr, &m_ycbcr_sampler) != VK_SUCCESS)
    return false;

  VkSamplerYcbcrConversionInfo view_conv{
      VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO};
  view_conv.conversion = m_ycbcr;
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.pNext = &view_conv;
  vi.image = m_sampleable.image;
  vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vi.format = m_picture_format;
  vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  if (vkCreateImageView(dev, &vi, nullptr, &m_sampleable.view) != VK_SUCCESS)
    return false;

  // The sampler is immutable in the layout - ycbcr requires it.
  VkDescriptorSetLayoutBinding binding{};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  binding.pImmutableSamplers = &m_ycbcr_sampler;
  VkDescriptorSetLayoutCreateInfo sl{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  sl.bindingCount = 1;
  sl.pBindings = &binding;
  if (vkCreateDescriptorSetLayout(dev, &sl, nullptr, &m_set_layout) !=
      VK_SUCCESS)
    return false;

  VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
  VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pi.maxSets = 1;
  pi.poolSizeCount = 1;
  pi.pPoolSizes = &pool_size;
  if (vkCreateDescriptorPool(dev, &pi, nullptr, &m_desc_pool) != VK_SUCCESS)
    return false;

  VkDescriptorSetAllocateInfo dai{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dai.descriptorPool = m_desc_pool;
  dai.descriptorSetCount = 1;
  dai.pSetLayouts = &m_set_layout;
  if (vkAllocateDescriptorSets(dev, &dai, &m_set) != VK_SUCCESS)
    return false;

  VkDescriptorImageInfo img_info{};
  img_info.imageView = m_sampleable.view;
  img_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = m_set;
  write.dstBinding = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  write.pImageInfo = &img_info;
  vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);

  VkPushConstantRange pc{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(f32) * 2};
  VkPipelineLayoutCreateInfo plc{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plc.setLayoutCount = 1;
  plc.pSetLayouts = &m_set_layout;
  plc.pushConstantRangeCount = 1;
  plc.pPushConstantRanges = &pc;
  if (vkCreatePipelineLayout(dev, &plc, nullptr, &m_pipe_layout) != VK_SUCCESS)
    return false;

  const VkShaderModule vs = create_shader_module(dev, VIDEO_CONVERT_VERT_SPV,
                                                 sizeof VIDEO_CONVERT_VERT_SPV);
  const VkShaderModule fs = create_shader_module(dev, VIDEO_CONVERT_FRAG_SPV,
                                                 sizeof VIDEO_CONVERT_FRAG_SPV);
  bool pipeline_ok = vs != VK_NULL_HANDLE && fs != VK_NULL_HANDLE;
  if (pipeline_ok) {
    VkPipelineShaderStageCreateInfo stages[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vin{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &blend;
    const VkDynamicState dynamics[2] = {VK_DYNAMIC_STATE_VIEWPORT,
                                        VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dynamics;

    const VkFormat rgba_format = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo rendering{
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &rgba_format;

    VkGraphicsPipelineCreateInfo gp{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.pNext = &rendering;
    gp.stageCount = 2;
    gp.pStages = stages;
    gp.pVertexInputState = &vin;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pColorBlendState = &cb;
    gp.pDynamicState = &dyn;
    gp.layout = m_pipe_layout;
    pipeline_ok = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr,
                                            &m_pipeline) == VK_SUCCESS;
  }
  if (vs != VK_NULL_HANDLE)
    vkDestroyShaderModule(dev, vs, nullptr);
  if (fs != VK_NULL_HANDLE)
    vkDestroyShaderModule(dev, fs, nullptr);
  if (!pipeline_ok)
    return false;

  // The RGBA target is a device texture so a caller can sample it bindlessly.
  TextureDesc rgba_desc;
  rgba_desc.name = "video rgba";
  rgba_desc.format = Format::RGBA8_UNORM;
  rgba_desc.width = m_extent.width;
  rgba_desc.height = m_extent.height;
  rgba_desc.usage = TextureUsage::Sampled | TextureUsage::RenderTarget |
                    TextureUsage::CopySrc;
  rgba_desc.initial_state = ResourceState::Undefined;
  m_rgba_handle = m_device->create_texture(rgba_desc);
  if (!m_rgba_handle.valid())
    return false;
  m_rgba_layout = VK_IMAGE_LAYOUT_UNDEFINED;

  // Graphics-queue command plumbing for the pass.
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
  return vkCreateFence(dev, &fci, nullptr, &m_gfx_fence) == VK_SUCCESS;
}

bool VideoDecoder::convert(const u32 slot) {
  if (!present(slot))
    return false;

  VulkanTexture *const rgba = m_device->texture_mut(m_rgba_handle);
  if (rgba == nullptr)
    return false;

  const VkDevice dev = m_device->vk_device();
  vkResetCommandBuffer(m_gfx_cmd, 0);
  VkCommandBufferBeginInfo begin_cmd{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_cmd.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(m_gfx_cmd, &begin_cmd);

  // The sampleable image's copy happened on the decode queue; make it visible to
  // this queue's sampler, and move the target into a colour attachment.
  VkImageMemoryBarrier2 pre[2];
  pre[0] = VkImageMemoryBarrier2{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  pre[0].srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  pre[0].srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
  pre[0].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
  pre[0].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
  pre[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  pre[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  pre[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  pre[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  pre[0].image = m_sampleable.image;
  pre[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  pre[1] = VkImageMemoryBarrier2{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  pre[1].srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  pre[1].srcAccessMask = 0;
  pre[1].dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  pre[1].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
  pre[1].oldLayout = m_rgba_layout;
  pre[1].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  pre[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  pre[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  pre[1].image = rgba->image;
  pre[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  m_rgba_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dep.imageMemoryBarrierCount = 2;
  dep.pImageMemoryBarriers = pre;
  vkCmdPipelineBarrier2(m_gfx_cmd, &dep);

  VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  color.imageView = rgba->rtv;
  color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  color.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, m_extent};
  ri.layerCount = 1;
  ri.colorAttachmentCount = 1;
  ri.pColorAttachments = &color;
  vkCmdBeginRendering(m_gfx_cmd, &ri);

  VkViewport viewport{0.0f, 0.0f, nx::cast<f32>(m_extent.width),
                      nx::cast<f32>(m_extent.height), 0.0f, 1.0f};
  VkRect2D scissor{{0, 0}, m_extent};
  vkCmdSetViewport(m_gfx_cmd, 0, 1, &viewport);
  vkCmdSetScissor(m_gfx_cmd, 0, 1, &scissor);
  vkCmdBindPipeline(m_gfx_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
  vkCmdBindDescriptorSets(m_gfx_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          m_pipe_layout, 0, 1, &m_set, 0, nullptr);
  const f32 uv_scale[2] = {1.0f, 1.0f};
  vkCmdPushConstants(m_gfx_cmd, m_pipe_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                     sizeof uv_scale, uv_scale);
  vkCmdDraw(m_gfx_cmd, 3, 1, 0, 0);
  vkCmdEndRendering(m_gfx_cmd);

  VkImageMemoryBarrier2 post{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  post.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  post.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
  post.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  post.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
  post.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  post.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  post.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  post.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  post.image = rgba->image;
  post.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  m_rgba_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  VkDependencyInfo post_dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  post_dep.imageMemoryBarrierCount = 1;
  post_dep.pImageMemoryBarriers = &post;
  vkCmdPipelineBarrier2(m_gfx_cmd, &post_dep);
  vkEndCommandBuffer(m_gfx_cmd);

  vkResetFences(dev, 1, &m_gfx_fence);
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &m_gfx_cmd;
  if (vkQueueSubmit(m_device->graphics_queue(), 1, &submit, m_gfx_fence) !=
      VK_SUCCESS)
    return false;
  vkWaitForFences(dev, 1, &m_gfx_fence, VK_TRUE, ~0ull);

  // The device tracked this texture; tell it the pass left it sampleable.
  m_device->set_texture_state(m_rgba_handle, ResourceState::ShaderReadOnly);
  return true;
}

bool VideoDecoder::read_rgba(const u32 width, const u32 height,
                             nx::vector<u8> &out) {
  if (!ready())
    return false;
  VulkanTexture *const rgba = m_device->texture_mut(m_rgba_handle);
  if (rgba == nullptr)
    return false;
  const VkDevice dev = m_device->vk_device();
  const u64 bytes = nx::cast<u64>(width) * height * 4;
  if (bytes > m_readback_capacity) {
    if (m_readback != VK_NULL_HANDLE)
      vkDestroyBuffer(dev, m_readback, nullptr);
    if (m_readback_memory != VK_NULL_HANDLE)
      vkFreeMemory(dev, m_readback_memory, nullptr);
    if (!create_host_buffer(dev, m_device->vk_physical_device(), bytes,
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT, nullptr,
                            m_readback, m_readback_memory, m_readback_mapped))
      return false;
    m_readback_capacity = bytes;
  }

  vkResetCommandBuffer(m_gfx_cmd, 0);
  VkCommandBufferBeginInfo begin_cmd{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_cmd.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(m_gfx_cmd, &begin_cmd);

  const auto barrier = [&](const VkImageLayout from, const VkImageLayout to) {
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
    b.image = rgba->image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo d{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    d.imageMemoryBarrierCount = 1;
    d.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(m_gfx_cmd, &d);
  };

  barrier(m_rgba_layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  VkBufferImageCopy2 region{VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = {width, height, 1};
  VkCopyImageToBufferInfo2 copy{VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2};
  copy.srcImage = rgba->image;
  copy.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  copy.dstBuffer = m_readback;
  copy.regionCount = 1;
  copy.pRegions = &region;
  vkCmdCopyImageToBuffer2(m_gfx_cmd, &copy);
  barrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  m_rgba_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  vkEndCommandBuffer(m_gfx_cmd);

  vkResetFences(dev, 1, &m_gfx_fence);
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &m_gfx_cmd;
  if (vkQueueSubmit(m_device->graphics_queue(), 1, &submit, m_gfx_fence) !=
      VK_SUCCESS)
    return false;
  vkWaitForFences(dev, 1, &m_gfx_fence, VK_TRUE, ~0ull);

  out.resize(nx::cast<usize>(bytes));
  std::memcpy(out.data(), m_readback_mapped, nx::cast<usize>(bytes));
  return true;
}

} // namespace nxe::rhi
