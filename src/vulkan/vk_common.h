#pragma once

#include "core/foundation/core/foundation.h"
#include "nx/gfx/gfx.h"
#include "nx/gfx/types.h"

#include <volk.h>

#include <type_traits>

namespace nx::gfx::detail {

/// Vulkan's non-dispatchable handles (VkImage, VkBuffer, ...) are pointers on
/// 64-bit targets but plain uint64_t on 32-bit ones, so no single cast spelling
/// compiles for both. Debug-naming needs them as u64.
template <typename T>
[[nodiscard]] u64 vk_handle_value(const T handle) noexcept {
  if constexpr (std::is_pointer_v<T>)
    return reinterpret_cast<u64>(handle);
  else
    return nx::cast<u64>(handle);
}

[[nodiscard]] nx::string_view vk_result_string(VkResult result) noexcept;

VkResult vk_report(VkResult result, const char *call, const char *file,
                   i32 line) noexcept;

/// Checks a Vulkan call, logging on failure. Evaluates the expression exactly
/// once. Use NX_VK_TRY when the caller can recover; this one only reports.
#define NX_VK_CHECK(expr)                                                      \
  ::nx::gfx::detail::vk_report((expr), #expr, __FILE__, __LINE__)

/// Device-owned variant used after logical-device creation. In addition to
/// logging, it captures VK_ERROR_DEVICE_LOST exactly once.
#define NX_VK_DEVICE_CHECK(device, expr)                                       \
  (device).check_result((expr), #expr, __FILE__, __LINE__)

#define NX_VK_DEVICE_TRY(device, expr)                                         \
  do {                                                                         \
    if ((device).check_result((expr), #expr, __FILE__, __LINE__) !=            \
        VK_SUCCESS)                                                            \
      return false;                                                            \
  } while (false)

/// Checks a Vulkan call and returns false from the enclosing function on
/// failure. Bring-up code is full of "do this or give up" sequences.
#define NX_VK_TRY(expr)                                                        \
  do {                                                                         \
    if (::nx::gfx::detail::vk_report((expr), #expr, __FILE__, __LINE__) !=     \
        VK_SUCCESS)                                                            \
      return false;                                                            \
  } while (false)

[[nodiscard]] VkFormat to_vk_format(gfx::format format) noexcept;
[[nodiscard]] gfx::format from_vk_format(VkFormat format) noexcept;
[[nodiscard]] VkImageUsageFlags
to_vk_image_usage(gfx::texture_usage_flags usage, gfx::format format) noexcept;
[[nodiscard]] VkBufferUsageFlags
to_vk_buffer_usage(gfx::buffer_usage_flags usage) noexcept;
[[nodiscard]] VkImageAspectFlags to_vk_aspect(gfx::format format) noexcept;
[[nodiscard]] VkImageType
to_vk_image_type(gfx::texture_dimension dimension) noexcept;
[[nodiscard]] VkImageViewType
to_vk_view_type(gfx::texture_dimension dimension) noexcept;
[[nodiscard]] VkAttachmentLoadOp to_vk_load_op(gfx::load_op op) noexcept;
[[nodiscard]] VkAttachmentStoreOp to_vk_store_op(gfx::store_op op) noexcept;
[[nodiscard]] VkFilter to_vk_filter(gfx::filter_mode mode) noexcept;
[[nodiscard]] VkSamplerMipmapMode
to_vk_mip_mode(gfx::filter_mode mode) noexcept;
[[nodiscard]] VkSamplerAddressMode
to_vk_address_mode(gfx::address_mode mode) noexcept;
[[nodiscard]] VkCompareOp to_vk_compare_op(gfx::compare_op op) noexcept;
[[nodiscard]] VkPresentModeKHR
to_vk_present_mode(gfx::present_mode mode) noexcept;

struct VkStateInfo {
  VkImageLayout layout;
  VkPipelineStageFlags2 stage;
  VkAccessFlags2 access;
};

/// @p shader_stages is every shader stage the device runs: a shader state
/// orders all of them.
[[nodiscard]] VkStateInfo
to_vk_state(gfx::resource_state state,
            VkPipelineStageFlags2 shader_stages) noexcept;

[[nodiscard]] VkAccessFlags2 to_buffer_access(VkAccessFlags2 access) noexcept;

} // namespace nx::gfx::detail
