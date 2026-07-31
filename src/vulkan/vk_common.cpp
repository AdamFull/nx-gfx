#include "core/rendering/rhi/vulkan/vk_common.h"

#include "core/foundation/diagnostics/log.h"

namespace nxe::rhi {
namespace {

const nx::log::Category log_rhi = nx::log::category("rhi");

// Indexed by Format, so it must stay in the enum's order. The static_assert in
// types.cpp guards the name table; this one guards the mapping.
constexpr VkFormat VK_FORMATS[] = {
    VK_FORMAT_UNDEFINED,

    VK_FORMAT_R8_UNORM,
    VK_FORMAT_R8_SNORM,
    VK_FORMAT_R8_UINT,
    VK_FORMAT_R8_SINT,
    VK_FORMAT_R8G8_UNORM,
    VK_FORMAT_R8G8_UINT,
    VK_FORMAT_R8G8B8A8_UNORM,
    VK_FORMAT_R8G8B8A8_SRGB,
    VK_FORMAT_R8G8B8A8_SNORM,
    VK_FORMAT_R8G8B8A8_UINT,
    VK_FORMAT_B8G8R8A8_UNORM,
    VK_FORMAT_B8G8R8A8_SRGB,

    VK_FORMAT_R16_UNORM,
    VK_FORMAT_R16_SFLOAT,
    VK_FORMAT_R16_UINT,
    VK_FORMAT_R16_SINT,
    VK_FORMAT_R16G16_SFLOAT,
    VK_FORMAT_R16G16_UNORM,
    VK_FORMAT_R16G16B16A16_SFLOAT,
    VK_FORMAT_R16G16B16A16_UNORM,

    VK_FORMAT_R32_SFLOAT,
    VK_FORMAT_R32_UINT,
    VK_FORMAT_R32_SINT,
    VK_FORMAT_R32G32_SFLOAT,
    VK_FORMAT_R32G32_UINT,
    VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_R32G32B32A32_UINT,

    VK_FORMAT_A2B10G10R10_UNORM_PACK32,
    VK_FORMAT_B10G11R11_UFLOAT_PACK32,

    VK_FORMAT_D16_UNORM,
    VK_FORMAT_D32_SFLOAT,
    VK_FORMAT_D24_UNORM_S8_UINT,
    VK_FORMAT_D32_SFLOAT_S8_UINT,

    VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
    VK_FORMAT_BC1_RGBA_SRGB_BLOCK,
    VK_FORMAT_BC3_UNORM_BLOCK,
    VK_FORMAT_BC3_SRGB_BLOCK,
    VK_FORMAT_BC4_UNORM_BLOCK,
    VK_FORMAT_BC5_UNORM_BLOCK,
    VK_FORMAT_BC7_UNORM_BLOCK,
    VK_FORMAT_BC7_SRGB_BLOCK,

    VK_FORMAT_ASTC_4x4_UNORM_BLOCK,
    VK_FORMAT_ASTC_4x4_SRGB_BLOCK,
    VK_FORMAT_ASTC_6x6_UNORM_BLOCK,
    VK_FORMAT_ASTC_6x6_SRGB_BLOCK,
    VK_FORMAT_ASTC_8x8_UNORM_BLOCK,
    VK_FORMAT_ASTC_8x8_SRGB_BLOCK,

    VK_FORMAT_BC1_RGB_UNORM_BLOCK,
    VK_FORMAT_BC1_RGB_SRGB_BLOCK,
    VK_FORMAT_BC2_UNORM_BLOCK,
    VK_FORMAT_BC2_SRGB_BLOCK,
    VK_FORMAT_BC6H_UFLOAT_BLOCK,
    VK_FORMAT_BC6H_SFLOAT_BLOCK,
};
static_assert(std::size(VK_FORMATS) == nx::cast<usize>(Format::Count),
              "VK_FORMATS is out of sync with Format");

} // namespace

nx::string_view vk_result_string(const VkResult result) noexcept {
  switch (result) {
  case VK_SUCCESS:
    return "VK_SUCCESS";
  case VK_NOT_READY:
    return "VK_NOT_READY";
  case VK_TIMEOUT:
    return "VK_TIMEOUT";
  case VK_EVENT_SET:
    return "VK_EVENT_SET";
  case VK_EVENT_RESET:
    return "VK_EVENT_RESET";
  case VK_INCOMPLETE:
    return "VK_INCOMPLETE";
  case VK_SUBOPTIMAL_KHR:
    return "VK_SUBOPTIMAL_KHR";
  case VK_ERROR_OUT_OF_HOST_MEMORY:
    return "VK_ERROR_OUT_OF_HOST_MEMORY";
  case VK_ERROR_OUT_OF_DEVICE_MEMORY:
    return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
  case VK_ERROR_INITIALIZATION_FAILED:
    return "VK_ERROR_INITIALIZATION_FAILED";
  case VK_ERROR_DEVICE_LOST:
    return "VK_ERROR_DEVICE_LOST";
  case VK_ERROR_MEMORY_MAP_FAILED:
    return "VK_ERROR_MEMORY_MAP_FAILED";
  case VK_ERROR_LAYER_NOT_PRESENT:
    return "VK_ERROR_LAYER_NOT_PRESENT";
  case VK_ERROR_EXTENSION_NOT_PRESENT:
    return "VK_ERROR_EXTENSION_NOT_PRESENT";
  case VK_ERROR_FEATURE_NOT_PRESENT:
    return "VK_ERROR_FEATURE_NOT_PRESENT";
  case VK_ERROR_INCOMPATIBLE_DRIVER:
    return "VK_ERROR_INCOMPATIBLE_DRIVER";
  case VK_ERROR_TOO_MANY_OBJECTS:
    return "VK_ERROR_TOO_MANY_OBJECTS";
  case VK_ERROR_FORMAT_NOT_SUPPORTED:
    return "VK_ERROR_FORMAT_NOT_SUPPORTED";
  case VK_ERROR_FRAGMENTED_POOL:
    return "VK_ERROR_FRAGMENTED_POOL";
  case VK_ERROR_OUT_OF_POOL_MEMORY:
    return "VK_ERROR_OUT_OF_POOL_MEMORY";
  case VK_ERROR_SURFACE_LOST_KHR:
    return "VK_ERROR_SURFACE_LOST_KHR";
  case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
    return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
  case VK_ERROR_OUT_OF_DATE_KHR:
    return "VK_ERROR_OUT_OF_DATE_KHR";
  case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR:
    return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
  case VK_ERROR_VALIDATION_FAILED_EXT:
    return "VK_ERROR_VALIDATION_FAILED_EXT";
  case VK_ERROR_UNKNOWN:
    return "VK_ERROR_UNKNOWN";
  default:
    return "VK_ERROR_<unmapped>";
  }
}

VkResult vk_report(const VkResult result, const char *call, const char *file,
                   const i32 line) noexcept {
  // Not every non-VK_SUCCESS is a failure: the swapchain codes are normal
  // control flow and the caller decides what to do with them.
  if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR ||
      result == VK_NOT_READY || result == VK_TIMEOUT)
    return result;

  nx::loge(log_rhi, "{} failed: {} ({}:{})", call, vk_result_string(result),
           file, line);
  return result;
}

VkFormat to_vk_format(const Format format) noexcept {
  const auto i = nx::cast<usize>(format);
  return VK_FORMATS[i < nx::cast<usize>(Format::Count) ? i : 0];
}

Format from_vk_format(const VkFormat format) noexcept {
  for (usize i = 0; i < nx::cast<usize>(Format::Count); ++i)
    if (VK_FORMATS[i] == format)
      return nx::cast<Format>(i);
  return Format::Unknown;
}

VkImageUsageFlags to_vk_image_usage(const TextureUsageFlags usage,
                                    const Format format) noexcept {
  VkImageUsageFlags flags = 0;
  if (usage & TextureUsage::Sampled)
    flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
  if (usage & TextureUsage::Storage)
    flags |= VK_IMAGE_USAGE_STORAGE_BIT;
  if (usage & TextureUsage::RenderTarget)
    flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  if (usage & TextureUsage::DepthStencil)
    flags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  if (usage & TextureUsage::CopySrc)
    flags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  if (usage & TextureUsage::CopyDst)
    flags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (usage & TextureUsage::Transient) {
    // On a tiler this keeps the attachment in tile memory and never backs it
    // with real allocation. Elsewhere the flag is accepted and ignored.
    flags |= VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
    flags |= format_is_depth(format)
                 ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                 : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  }
  return flags;
}

VkBufferUsageFlags to_vk_buffer_usage(const BufferUsageFlags usage) noexcept {
  VkBufferUsageFlags flags = 0;
  if (usage & BufferUsage::Vertex)
    flags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  if (usage & BufferUsage::Index)
    flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  if (usage & BufferUsage::Uniform)
    flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  if (usage & BufferUsage::Storage)
    flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (usage & BufferUsage::Indirect)
    flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  if (usage & BufferUsage::CopySrc)
    flags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (usage & BufferUsage::CopyDst)
    flags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (usage & BufferUsage::DeviceAddress)
    flags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  return flags;
}

VkImageAspectFlags to_vk_aspect(const Format format) noexcept {
  if (format_is_depth(format)) {
    return format_is_stencil(format)
               ? (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)
               : VK_IMAGE_ASPECT_DEPTH_BIT;
  }
  return VK_IMAGE_ASPECT_COLOR_BIT;
}

VkImageType to_vk_image_type(const TextureDimension dimension) noexcept {
  return dimension == TextureDimension::Tex3D ? VK_IMAGE_TYPE_3D
                                              : VK_IMAGE_TYPE_2D;
}

VkImageViewType to_vk_view_type(const TextureDimension dimension) noexcept {
  switch (dimension) {
  case TextureDimension::Tex2D:
    return VK_IMAGE_VIEW_TYPE_2D;
  case TextureDimension::Tex2DArray:
    return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  case TextureDimension::TexCube:
    return VK_IMAGE_VIEW_TYPE_CUBE;
  case TextureDimension::Tex3D:
    return VK_IMAGE_VIEW_TYPE_3D;
  default:
    return VK_IMAGE_VIEW_TYPE_2D;
  }
}

VkAttachmentLoadOp to_vk_load_op(const LoadOp op) noexcept {
  switch (op) {
  case LoadOp::Load:
    return VK_ATTACHMENT_LOAD_OP_LOAD;
  case LoadOp::Clear:
    return VK_ATTACHMENT_LOAD_OP_CLEAR;
  default:
    return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  }
}

VkAttachmentStoreOp to_vk_store_op(const StoreOp op) noexcept {
  return op == StoreOp::Store ? VK_ATTACHMENT_STORE_OP_STORE
                              : VK_ATTACHMENT_STORE_OP_DONT_CARE;
}

VkFilter to_vk_filter(const FilterMode mode) noexcept {
  return mode == FilterMode::Linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
}

VkSamplerMipmapMode to_vk_mip_mode(const FilterMode mode) noexcept {
  return mode == FilterMode::Linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                    : VK_SAMPLER_MIPMAP_MODE_NEAREST;
}

VkSamplerAddressMode to_vk_address_mode(const AddressMode mode) noexcept {
  switch (mode) {
  case AddressMode::Repeat:
    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  case AddressMode::MirroredRepeat:
    return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  case AddressMode::ClampToEdge:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  case AddressMode::ClampToBorder:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  default:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  }
}

VkCompareOp to_vk_compare_op(const CompareOp op) noexcept {
  switch (op) {
  case CompareOp::Never:
    return VK_COMPARE_OP_NEVER;
  case CompareOp::Less:
    return VK_COMPARE_OP_LESS;
  case CompareOp::Equal:
    return VK_COMPARE_OP_EQUAL;
  case CompareOp::LessOrEqual:
    return VK_COMPARE_OP_LESS_OR_EQUAL;
  case CompareOp::Greater:
    return VK_COMPARE_OP_GREATER;
  case CompareOp::NotEqual:
    return VK_COMPARE_OP_NOT_EQUAL;
  case CompareOp::GreaterOrEqual:
    return VK_COMPARE_OP_GREATER_OR_EQUAL;
  default:
    return VK_COMPARE_OP_ALWAYS;
  }
}

VkPresentModeKHR to_vk_present_mode(const PresentMode mode) noexcept {
  switch (mode) {
  case PresentMode::Mailbox:
    return VK_PRESENT_MODE_MAILBOX_KHR;
  case PresentMode::Immediate:
    return VK_PRESENT_MODE_IMMEDIATE_KHR;
  default:
    return VK_PRESENT_MODE_FIFO_KHR;
  }
}

VkStateInfo to_vk_state(const ResourceState state) noexcept {
  constexpr VkPipelineStageFlags2 ALL_SHADERS =
      VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  constexpr VkPipelineStageFlags2 DEPTH_STAGES =
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;

  switch (state) {
  case ResourceState::Undefined:
    return {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
            VK_ACCESS_2_NONE};

  case ResourceState::General:
    return {VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};

  case ResourceState::CopySrc:
    return {VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT};

  case ResourceState::CopyDst:
    return {VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT};

  case ResourceState::ShaderReadOnly:
    return {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, ALL_SHADERS,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT};

  case ResourceState::ShaderWriteOnly:
    return {VK_IMAGE_LAYOUT_GENERAL, ALL_SHADERS,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT};

  case ResourceState::ShaderReadWrite:
    return {VK_IMAGE_LAYOUT_GENERAL, ALL_SHADERS,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT};

  case ResourceState::ColorAttachment:
    return {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT};

  case ResourceState::ColorAttachmentRead:
    return {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT};

  case ResourceState::ColorAttachmentWrite:
    return {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT};

  case ResourceState::DepthStencilRead:
    return {VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, DEPTH_STAGES,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT};

  case ResourceState::DepthStencilWrite:
    return {VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, DEPTH_STAGES,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};

  case ResourceState::DepthStencilReadWrite:
    return {VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, DEPTH_STAGES,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};

  case ResourceState::VertexBuffer:
    return {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
            VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT};

  case ResourceState::IndexBuffer:
    return {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
            VK_ACCESS_2_INDEX_READ_BIT};

  case ResourceState::IndirectBuffer:
    return {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT};

  case ResourceState::UniformBuffer:
    return {VK_IMAGE_LAYOUT_UNDEFINED,
            ALL_SHADERS | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
            VK_ACCESS_2_UNIFORM_READ_BIT};

  case ResourceState::HostRead:
    return {VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_HOST_BIT,
            VK_ACCESS_2_HOST_READ_BIT};

  case ResourceState::HostWrite:
    return {VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_HOST_BIT,
            VK_ACCESS_2_HOST_WRITE_BIT};

  case ResourceState::Present:
    // Present is ordered by the release semaphore, not by memory access.
    return {VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_ACCESS_2_NONE};

  default:
    return {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_NONE};
  }
}

} // namespace nxe::rhi
