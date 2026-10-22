#pragma once

#include "core/foundation/threading/sync.h"
#include "nx/gfx/crash.h"
#include "src/vulkan/vk_aftermath.h"
#include "src/vulkan/vk_common.h"

#include <atomic>

namespace nx::gfx::detail {

enum class VulkanFaultRoute : u8 { None, Khr, Ext };

class VulkanGpuCrashReporter {
public:
  /// Starts Nsight Aftermath, where it is built in, when @p mode is
  /// Detailed, naming dumps as @p application's.
  void configure(gfx::gpu_crash_mode mode, bool break_on_validation_error,
                 nx::string_view application = {}) noexcept;
  void attach(VkDevice device, nx::string_view device_name,
              VulkanFaultRoute route, bool vendor_binary,
              bool address_tracking);
  void reset() noexcept;

  [[nodiscard]] bool enabled() const noexcept { return m_enabled; }
  [[nodiscard]] VulkanAftermath &aftermath() noexcept { return m_aftermath; }
  [[nodiscard]] bool detailed() const noexcept { return m_detailed; }
  [[nodiscard]] gfx::gpu_device_status status() const noexcept {
    return m_lost.load(std::memory_order_acquire)
               ? gfx::gpu_device_status::Lost
               : gfx::gpu_device_status::Ready;
  }

  VkBool32
  debug_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                const VkDebugUtilsMessengerCallbackDataEXT *data) noexcept;
  void name_object(u64 object, VkObjectType type,
                   nx::string_view name) noexcept;
  VkResult check(VkResult result, const char *call, const char *file, i32 line,
                 u64 frame) noexcept;

private:
  static constexpr usize BINDING_CAPACITY = 256;
  static constexpr usize OBJECT_CAPACITY = 512;
  static constexpr usize TEXT_CAPACITY = 96;

  struct BindingEvent {
    u64 sequence = 0;
    u64 object = 0;
    u64 address = 0;
    u64 size = 0;
    VkObjectType object_type = VK_OBJECT_TYPE_UNKNOWN;
    bool bound = false;
    bool internal = false;
    char name[TEXT_CAPACITY] = {};
  };

  struct ObjectNameEvent {
    u64 sequence = 0;
    u64 object = 0;
    VkObjectType object_type = VK_OBJECT_TYPE_UNKNOWN;
    char name[TEXT_CAPACITY] = {};
  };

  void remember_binding(const VkDebugUtilsMessengerCallbackDataEXT &data,
                        const VkDeviceAddressBindingCallbackDataEXT &binding);
  void report_device_lost(const char *call, const char *file, i32 line,
                          u64 frame);
  void collect_khr(gfx::gpu_crash_report &out);
  void collect_ext(gfx::gpu_crash_report &out);
  void append_objects_and_bindings(gfx::gpu_crash_report &out);

  VkDevice m_device = VK_NULL_HANDLE;
  nx::string m_device_name;
  VulkanFaultRoute m_route = VulkanFaultRoute::None;
  std::atomic<bool> m_lost{false};
  std::atomic<bool> m_reported{false};
  nx::spin_lock m_binding_lock;
  BindingEvent m_bindings[BINDING_CAPACITY] = {};
  u64 m_binding_sequence = 0;
  ObjectNameEvent m_objects[OBJECT_CAPACITY] = {};
  u64 m_object_sequence = 0;
  bool m_enabled = true;
  bool m_detailed = true;
  bool m_break_on_validation_error = false;
  bool m_vendor_binary = false;
  bool m_address_tracking = false;
  VulkanAftermath m_aftermath;
};

} // namespace nx::gfx::detail
