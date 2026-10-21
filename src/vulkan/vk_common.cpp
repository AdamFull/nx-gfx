#include "src/vulkan/vk_common.h"

#include "core/foundation/diagnostics/log.h"

namespace nx::gfx::detail {
namespace {

// Indexed by gfx::format, so it must stay in the enum's order. The
// static_assert in types.cpp guards the name table; this one guards the
// mapping.
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
static_assert(std::size(VK_FORMATS) == nx::cast<usize>(gfx::format::Count),
              "VK_FORMATS is out of sync with gfx::format");
}

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
  if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR ||
      result == VK_NOT_READY || result == VK_TIMEOUT)
    return result;

  nx::loge(nx::log::rhi, "{} failed: {} ({}:{})", call, vk_result_string(result),
           file, line);
  return result;
}

VkFormat to_vk_format(const gfx::format format) noexcept {
  const auto i = nx::cast<usize>(format);
  return VK_FORMATS[i < nx::cast<usize>(gfx::format::Count) ? i : 0];
}

gfx::format from_vk_format(const VkFormat format) noexcept {
  for (usize i = 0; i < nx::cast<usize>(gfx::format::Count); ++i)
    if (VK_FORMATS[i] == format)
      return nx::cast<gfx::format>(i);
  return gfx::format::Unknown;
}

VkImageUsageFlags to_vk_image_usage(const gfx::texture_usage_flags usage,
                                    const gfx::format format) noexcept {
  VkImageUsageFlags flags = 0;
  if (usage & gfx::texture_usage::Sampled)
    flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
  if (usage & gfx::texture_usage::Storage)
    flags |= VK_IMAGE_USAGE_STORAGE_BIT;
  if (usage & gfx::texture_usage::RenderTarget)
    flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  if (usage & gfx::texture_usage::DepthStencil)
    flags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  if (usage & gfx::texture_usage::CopySrc)
    flags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  if (usage & gfx::texture_usage::CopyDst)
    flags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (usage & gfx::texture_usage::Transient) {
    flags |= VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
    flags |= format_is_depth(format)
                 ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                 : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  }
  return flags;
}

VkBufferUsageFlags
to_vk_buffer_usage(const gfx::buffer_usage_flags usage) noexcept {
  VkBufferUsageFlags flags = 0;
  if (usage & gfx::buffer_usage::Vertex)
    flags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  if (usage & gfx::buffer_usage::Index)
    flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  if (usage & gfx::buffer_usage::Uniform)
    flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  if (usage & gfx::buffer_usage::Storage)
    flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (usage & gfx::buffer_usage::Indirect)
    flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  if (usage & gfx::buffer_usage::CopySrc)
    flags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (usage & gfx::buffer_usage::CopyDst)
    flags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (usage & gfx::buffer_usage::DeviceAddress)
    flags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  return flags;
}

VkImageAspectFlags to_vk_aspect(const gfx::format format) noexcept {
  if (format_is_depth(format)) {
    return format_is_stencil(format)
               ? (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)
               : VK_IMAGE_ASPECT_DEPTH_BIT;
  }
  return VK_IMAGE_ASPECT_COLOR_BIT;
}

VkImageType to_vk_image_type(const gfx::texture_dimension dimension) noexcept {
  return dimension == gfx::texture_dimension::Tex3D ? VK_IMAGE_TYPE_3D
                                                    : VK_IMAGE_TYPE_2D;
}

VkImageViewType
to_vk_view_type(const gfx::texture_dimension dimension) noexcept {
  switch (dimension) {
  case gfx::texture_dimension::Tex2D:
    return VK_IMAGE_VIEW_TYPE_2D;
  case gfx::texture_dimension::Tex2DArray:
    return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  case gfx::texture_dimension::TexCube:
    return VK_IMAGE_VIEW_TYPE_CUBE;
  case gfx::texture_dimension::Tex3D:
    return VK_IMAGE_VIEW_TYPE_3D;
  default:
    return VK_IMAGE_VIEW_TYPE_2D;
  }
}

VkAttachmentLoadOp to_vk_load_op(const gfx::load_op op) noexcept {
  switch (op) {
  case gfx::load_op::Load:
    return VK_ATTACHMENT_LOAD_OP_LOAD;
  case gfx::load_op::Clear:
    return VK_ATTACHMENT_LOAD_OP_CLEAR;
  default:
    return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  }
}

VkAttachmentStoreOp to_vk_store_op(const gfx::store_op op) noexcept {
  return op == gfx::store_op::Store ? VK_ATTACHMENT_STORE_OP_STORE
                                    : VK_ATTACHMENT_STORE_OP_DONT_CARE;
}

VkFilter to_vk_filter(const gfx::filter_mode mode) noexcept {
  return mode == gfx::filter_mode::Linear ? VK_FILTER_LINEAR
                                          : VK_FILTER_NEAREST;
}

VkSamplerMipmapMode to_vk_mip_mode(const gfx::filter_mode mode) noexcept {
  return mode == gfx::filter_mode::Linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                          : VK_SAMPLER_MIPMAP_MODE_NEAREST;
}

VkSamplerAddressMode to_vk_address_mode(const gfx::address_mode mode) noexcept {
  switch (mode) {
  case gfx::address_mode::Repeat:
    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  case gfx::address_mode::MirroredRepeat:
    return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  case gfx::address_mode::ClampToEdge:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  case gfx::address_mode::ClampToBorder:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  default:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  }
}

VkCompareOp to_vk_compare_op(const gfx::compare_op op) noexcept {
  switch (op) {
  case gfx::compare_op::Never:
    return VK_COMPARE_OP_NEVER;
  case gfx::compare_op::Less:
    return VK_COMPARE_OP_LESS;
  case gfx::compare_op::Equal:
    return VK_COMPARE_OP_EQUAL;
  case gfx::compare_op::LessOrEqual:
    return VK_COMPARE_OP_LESS_OR_EQUAL;
  case gfx::compare_op::Greater:
    return VK_COMPARE_OP_GREATER;
  case gfx::compare_op::NotEqual:
    return VK_COMPARE_OP_NOT_EQUAL;
  case gfx::compare_op::GreaterOrEqual:
    return VK_COMPARE_OP_GREATER_OR_EQUAL;
  default:
    return VK_COMPARE_OP_ALWAYS;
  }
}

VkPresentModeKHR to_vk_present_mode(const gfx::present_mode mode) noexcept {
  switch (mode) {
  case gfx::present_mode::Mailbox:
    return VK_PRESENT_MODE_MAILBOX_KHR;
  case gfx::present_mode::Immediate:
    return VK_PRESENT_MODE_IMMEDIATE_KHR;
  default:
    return VK_PRESENT_MODE_FIFO_KHR;
  }
}

VkAccessFlags2 to_buffer_access(const VkAccessFlags2 access) noexcept {
  constexpr VkAccessFlags2 WRITES =
      VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
      VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_HOST_WRITE_BIT |
      VK_ACCESS_2_MEMORY_WRITE_BIT;
  if (access == VK_ACCESS_2_NONE)
    return VK_ACCESS_2_NONE;
  VkAccessFlags2 out = 0;
  if ((access & WRITES) != 0)
    out |= VK_ACCESS_2_MEMORY_WRITE_BIT;
  if ((access & ~WRITES) != 0)
    out |= VK_ACCESS_2_MEMORY_READ_BIT;
  return out;
}

VkStateInfo to_vk_state(const gfx::resource_state state,
                        const VkPipelineStageFlags2 shader_stages) noexcept {
  constexpr VkPipelineStageFlags2 DEPTH_STAGES =
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;

  switch (state) {
  case gfx::resource_state::Undefined:
    return {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
            VK_ACCESS_2_NONE};

  case gfx::resource_state::General:
    return {VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};

  case gfx::resource_state::CopySrc:
    return {VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT};

  case gfx::resource_state::CopyDst:
    return {VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT};

  case gfx::resource_state::ShaderReadOnly:
    return {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, shader_stages,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT};

  case gfx::resource_state::ShaderWriteOnly:
    return {VK_IMAGE_LAYOUT_GENERAL, shader_stages,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT};

  case gfx::resource_state::ShaderReadWrite:
    return {VK_IMAGE_LAYOUT_GENERAL, shader_stages,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT};

  case gfx::resource_state::ColorAttachment:
    return {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT};

  case gfx::resource_state::ColorAttachmentRead:
    return {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT};

  case gfx::resource_state::ColorAttachmentWrite:
    return {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT};

  case gfx::resource_state::DepthStencilRead:
    return {VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, DEPTH_STAGES,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT};

  case gfx::resource_state::DepthStencilWrite:
    return {VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, DEPTH_STAGES,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};

  case gfx::resource_state::DepthStencilReadWrite:
    return {VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, DEPTH_STAGES,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};

  case gfx::resource_state::VertexBuffer:
    return {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
            VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT};

  case gfx::resource_state::IndexBuffer:
    return {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
            VK_ACCESS_2_INDEX_READ_BIT};

  case gfx::resource_state::IndirectBuffer:
    return {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT};

  case gfx::resource_state::UniformBuffer:
    return {VK_IMAGE_LAYOUT_UNDEFINED,
            shader_stages | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
            VK_ACCESS_2_UNIFORM_READ_BIT};

  case gfx::resource_state::HostRead:
    return {VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_HOST_BIT,
            VK_ACCESS_2_HOST_READ_BIT};

  case gfx::resource_state::HostWrite:
    return {VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_HOST_BIT,
            VK_ACCESS_2_HOST_WRITE_BIT};

  case gfx::resource_state::Present:
    // Present is ordered by the release semaphore, not by memory access.
    return {VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_ACCESS_2_NONE};

  default:
    return {VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_NONE};
  }
}

} // namespace nx::gfx::detail
