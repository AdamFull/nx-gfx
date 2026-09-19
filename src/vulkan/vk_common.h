#pragma once

#include "core/foundation/core/foundation.h"
#include "rendering/rhi/descs.h"
#include "rendering/rhi/types.h"

#include <volk.h>

#include <type_traits>

namespace nxe::rhi {

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
  ::nxe::rhi::vk_report((expr), #expr, __FILE__, __LINE__)

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
    if (::nxe::rhi::vk_report((expr), #expr, __FILE__, __LINE__) !=            \
        VK_SUCCESS)                                                            \
      return false;                                                            \
  } while (false)

[[nodiscard]] VkFormat to_vk_format(Format format) noexcept;
[[nodiscard]] Format from_vk_format(VkFormat format) noexcept;
[[nodiscard]] VkImageUsageFlags to_vk_image_usage(TextureUsageFlags usage,
                                                  Format format) noexcept;
[[nodiscard]] VkBufferUsageFlags
to_vk_buffer_usage(BufferUsageFlags usage) noexcept;
[[nodiscard]] VkImageAspectFlags to_vk_aspect(Format format) noexcept;
[[nodiscard]] VkImageType to_vk_image_type(TextureDimension dimension) noexcept;
[[nodiscard]] VkImageViewType
to_vk_view_type(TextureDimension dimension) noexcept;
[[nodiscard]] VkAttachmentLoadOp to_vk_load_op(LoadOp op) noexcept;
[[nodiscard]] VkAttachmentStoreOp to_vk_store_op(StoreOp op) noexcept;
[[nodiscard]] VkFilter to_vk_filter(FilterMode mode) noexcept;
[[nodiscard]] VkSamplerMipmapMode to_vk_mip_mode(FilterMode mode) noexcept;
[[nodiscard]] VkSamplerAddressMode
to_vk_address_mode(AddressMode mode) noexcept;
[[nodiscard]] VkCompareOp to_vk_compare_op(CompareOp op) noexcept;
[[nodiscard]] VkPresentModeKHR to_vk_present_mode(PresentMode mode) noexcept;

struct VkStateInfo {
  VkImageLayout layout;
  VkPipelineStageFlags2 stage;
  VkAccessFlags2 access;
};

[[nodiscard]] VkStateInfo to_vk_state(ResourceState state) noexcept;

[[nodiscard]] VkAccessFlags2 to_buffer_access(VkAccessFlags2 access) noexcept;

} // namespace nxe::rhi
