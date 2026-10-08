#pragma once

#include "core/foundation/containers/small_vector.h"
#include "core/foundation/core/foundation.h"
#include "core/foundation/strings/utf8_string.h"

namespace nxe::rhi {

/// The graphics API a device's own objects belong to.
enum class NativeApi : u8 { Vulkan, Metal };

/// A device's own objects, for a vendor library that works on them directly
/// (an upscaler SDK). Vulkan: VkInstance, VkPhysicalDevice, VkDevice, the
/// graphics VkQueue and its family, the API version the device was made
/// for, and PFN_vkGetInstanceProcAddr / PFN_vkGetDeviceProcAddr to load
/// through. Metal: id<MTLDevice> and the id<MTLCommandQueue>. A client asked
/// before the device is made sees no device or queue yet.
struct NativeDevice {
  NativeApi api = NativeApi::Vulkan;
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
struct NativeTexture {
  void *image = nullptr;
  void *view = nullptr;
  u32 format = 0;
  u32 width = 0;
  u32 height = 0;
  u32 mip_levels = 1;
  u32 layers = 1;
};

using NativeNames = nx::small_vector<nx::string, 8>;

/// What a vendor library asks of a device before it is made, on the thread
/// making it. The device enables the extensions it has and leaves out the
/// rest; where it cannot be made with every client's features, it is made
/// with none of their requests, which native_extension() then reports.
class NativeDeviceClient {
public:
  virtual ~NativeDeviceClient() = default;
  /// Vulkan instance extensions it would enable.
  virtual void instance_extensions(NativeApi, NativeNames &) {}
  /// Vulkan device extensions it would enable on @p adapter.
  virtual void device_extensions(const NativeDevice &, NativeNames &) {}
  /// Patches what the device is made with. Vulkan: @p features is the
  /// VkPhysicalDeviceFeatures2 heading the device's pNext chain.
  virtual void device_features(const NativeDevice &, void *) {}
};

} // namespace nxe::rhi
