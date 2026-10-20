#pragma once

#include "core/foundation/containers/small_vector.h"
#include "core/foundation/core/foundation.h"
#include "core/foundation/strings/utf8_string.h"
#include "core/foundation/strings/utf8_string_view.h"
#include "nx/gfx/types.h"

namespace nx::gfx {

/// The graphics API a device's own objects belong to.
enum class native_api : u8 { Vulkan, Metal };

/// A device's own objects, for a vendor library that works on them directly
/// (an upscaler SDK). Vulkan: VkInstance, VkPhysicalDevice, VkDevice, the
/// graphics VkQueue and its family, the API version the device was made
/// for, and PFN_vkGetInstanceProcAddr / PFN_vkGetDeviceProcAddr to load
/// through. Metal: id<MTLDevice> and the id<MTLCommandQueue>. A client asked
/// before the device is made sees no device or queue yet.
struct native_device {
  native_api api = native_api::Vulkan;
  void *instance = nullptr;
  void *adapter = nullptr;
  void *device = nullptr;
  void *queue = nullptr;
  u32 queue_family = 0;
  u32 api_version = 0;
  void *get_instance_proc_addr = nullptr;
  void *get_device_proc_addr = nullptr;
};

/// A texture's own objects. Vulkan: its VkImage, its sampled VkImageView (every
/// layer, and every mip but those still streaming in) and its VkFormat.
/// Metal: its id<MTLTexture> in `image`.
struct native_texture {
  void *image = nullptr;
  void *view = nullptr;
  u32 format = 0;
  u32 width = 0;
  u32 height = 0;
  u32 mip_levels = 1;
  u32 layers = 1;
};

using native_names = nx::small_vector<nx::string, 8>;

/// A queue the device made for a client, beside its own. Vulkan: the
/// VkQueue and its family.
struct native_queue {
  void *queue = nullptr;
  u32 family = ~0u;

  [[nodiscard]] bool valid() const noexcept { return queue != nullptr; }
};

/// What a vendor library asks of a device before it is made, on the thread
/// making it. The device enables the extensions it has and leaves out the
/// rest; where it cannot be made with every client's features, it is made
/// with none of their requests, which native_extension() then reports.
class native_device_client {
public:
  virtual ~native_device_client() = default;
  /// Vulkan instance extensions it would enable.
  virtual void instance_extensions(native_api, native_names &) {}
  /// Vulkan device extensions it would enable on @p adapter.
  virtual void device_extensions(const native_device &, native_names &) {}
  /// Patches what the device is made with. Vulkan: @p features is the
  /// VkPhysicalDeviceFeatures2 heading the device's pNext chain.
  virtual void device_features(const native_device &, void *) {}
  /// A queue family of @p adapter to make one queue of the client's own in,
  /// read back through native(device, client); none by default. It is made
  /// only in a family none of the device's own queues use.
  virtual u32 queue_family(const native_device &) { return ~0u; }
};

using native_driver_resolve = void *(*)(void *context,
                                        const char *symbol) noexcept;

/// Non-owning access to a process-local graphics driver library. The provider
/// that created this view must outlive the RHI device and release the library
/// only after every native object has been destroyed.
struct native_driver_view {
  void *context = nullptr;
  native_driver_resolve resolve = nullptr;
  nx::string_view identity = "system";

  [[nodiscard]] bool custom() const noexcept { return resolve != nullptr; }

  [[nodiscard]] void *find(const char *symbol) const noexcept {
    return resolve != nullptr ? resolve(context, symbol) : nullptr;
  }
};

/// The device's own objects. Metal reports only its API until it has them.
[[nodiscard]] native_device native(const device &dev) noexcept;
[[nodiscard]] native_texture native(const device &dev,
                                    texture_handle texture) noexcept;
/// The queue the device made for @p client; none when it made none.
[[nodiscard]] native_queue native(const device &dev,
                                  const native_device_client &client) noexcept;
/// The native layout (Vulkan: VkImageLayout) of @p state.
[[nodiscard]] u32 native_layout(const device &dev,
                                resource_state state) noexcept;
/// Whether the device enabled the instance or device extension @p name.
[[nodiscard]] bool native_extension(const device &dev,
                                    nx::string_view name) noexcept;

/// The command buffer itself (VkCommandBuffer), for a vendor library to
/// record into between here and end_native(); outside a render pass. Null
/// where the backend has none to give.
[[nodiscard]] void *begin_native(command_list &cmd) noexcept;
/// Takes the commands back: what the library bound in their place is bound
/// again.
void end_native(command_list &cmd) noexcept;

} // namespace nx::gfx
