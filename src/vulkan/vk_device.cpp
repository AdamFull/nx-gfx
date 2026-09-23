#include "rendering/rhi/vulkan/vk_device.h"

#include <cstring>

#include "core/foundation/core/scope_guard.h"
#include "core/foundation/diagnostics/diagnostic.h"
#include "core/foundation/diagnostics/log.h"

namespace nxe::rhi {
namespace {

constexpr u32 REQUIRED_API_VERSION = VK_API_VERSION_1_1;

/// Links a pNext feature-struct chain one block at a time. Holds `tail` by
/// reference rather than by value: create_logical_device reads the caller's
/// own `tail` variable after the last chain() call to remember where the
/// video feature block was spliced in, for the retry-without-video path.
struct PNextChainer {
  void **&tail;
  template <typename T> void operator()(T &block) const noexcept {
    *tail = &block;
    tail = const_cast<void **>(&block.pNext);
  }
};

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*types*/,
    const VkDebugUtilsMessengerCallbackDataEXT *data, void *user_data) {
  return user_data != nullptr
             ? nx::cast<VulkanGpuCrashReporter *>(user_data)->debug_message(
                   severity, data)
             : VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT
debug_messenger_info(VulkanGpuCrashReporter &reporter, const bool detailed,
                     const bool address_bindings) noexcept {
  VkDebugUtilsMessengerCreateInfoEXT info{
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
  info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
  if (detailed)
    info.messageSeverity |= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
  info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  if (address_bindings)
    info.messageType |=
        VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT;
  info.pfnUserCallback = debug_callback;
  info.pUserData = &reporter;
  return info;
}

bool has_extension(const nx::small_vector<VkExtensionProperties, 64> &available,
                   const char *name) noexcept {
  for (const auto &ext : available)
    if (std::strcmp(ext.extensionName, name) == 0)
      return true;
  return false;
}

constexpr const char *platform_surface_extension() noexcept {
#if defined(VK_USE_PLATFORM_WIN32_KHR)
  return VK_KHR_WIN32_SURFACE_EXTENSION_NAME;
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
  return VK_KHR_ANDROID_SURFACE_EXTENSION_NAME;
#elif defined(VK_USE_PLATFORM_METAL_EXT)
  return VK_EXT_METAL_SURFACE_EXTENSION_NAME;
#elif defined(VK_USE_PLATFORM_XLIB_KHR)
  return VK_KHR_XLIB_SURFACE_EXTENSION_NAME;
#else
  return nullptr;
#endif
}

[[nodiscard]] bool
listed(const nx::small_vector<VkExtensionProperties, 128> &all,
       const char *name) {
  for (const VkExtensionProperties &one : all)
    if (std::strcmp(one.extensionName, name) == 0)
      return true;
  return false;
}

} // namespace

DeviceRoutes VulkanDevice::probe_routes(VkPhysicalDevice candidate,
                                        const VkPhysicalDeviceProperties &props,
                                        const u32 instance_api_version) {
  DeviceRoutes out;
  const u32 core_version = props.apiVersion < instance_api_version
                               ? props.apiVersion
                               : instance_api_version;
  out.core_12 = core_version >= VK_API_VERSION_1_2;
  out.core_13 = core_version >= VK_API_VERSION_1_3;

  u32 count = 0;
  vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, nullptr);
  nx::small_vector<VkExtensionProperties, 128> available(count);
  vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count,
                                       available.data());

  const bool has_swapchain = listed(available, VK_KHR_SWAPCHAIN_EXTENSION_NAME);

  const bool ext_buffer_device_address =
      listed(available, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  const bool ext_timeline_semaphore =
      listed(available, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
  const bool ext_synchronization2 =
      listed(available, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
  const bool ext_dynamic_rendering =
      listed(available, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
  const bool ext_descriptor_indexing =
      listed(available, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
  const bool ext_scalar_block_layout =
      listed(available, VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME);
  const bool ext_draw_indirect_count =
      listed(available, VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME);
  const bool ext_render_pass2 =
      listed(available, VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
  const bool ext_depth_stencil_resolve =
      listed(available, VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME);
  const bool ext_device_fault_khr =
      listed(available, VK_KHR_DEVICE_FAULT_EXTENSION_NAME);
  const bool ext_device_fault_ext =
      listed(available, VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
  const bool ext_address_binding =
      listed(available, VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME);
  const bool ext_calibrated_khr =
      listed(available, VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
  const bool ext_calibrated_ext =
      listed(available, VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);

  // A feature struct may only be chained when the version or the extension that
  // defines it is there, so the chain is built from what this device has.
  VkPhysicalDeviceFeatures2 features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  void **tail = &features.pNext;
  const PNextChainer chain{tail};

  VkPhysicalDeviceVulkan12Features features12{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceVulkan13Features features13{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceDescriptorIndexingFeatures indexing{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
  VkPhysicalDeviceBufferDeviceAddressFeatures address{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
  VkPhysicalDeviceTimelineSemaphoreFeatures timeline{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
  VkPhysicalDeviceDynamicRenderingFeatures dynamic{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES};
  VkPhysicalDeviceSynchronization2Features sync2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
  VkPhysicalDeviceScalarBlockLayoutFeatures scalar{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES};
  VkPhysicalDeviceShaderDrawParametersFeatures draw_params{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES};
  VkPhysicalDeviceFaultFeaturesKHR fault_khr{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_KHR};
  VkPhysicalDeviceFaultFeaturesEXT fault_ext{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT};
  VkPhysicalDeviceAddressBindingReportFeaturesEXT address_binding{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ADDRESS_BINDING_REPORT_FEATURES_EXT};

  chain(draw_params);

  if (out.core_12) {
    chain(features12);
  } else {
    if (ext_descriptor_indexing)
      chain(indexing);
    if (ext_buffer_device_address)
      chain(address);
    if (ext_timeline_semaphore)
      chain(timeline);
    if (ext_scalar_block_layout)
      chain(scalar);
  }

  if (out.core_13) {
    chain(features13);
  } else {
    if (ext_dynamic_rendering)
      chain(dynamic);
    if (ext_synchronization2)
      chain(sync2);
  }

  if (ext_device_fault_khr)
    chain(fault_khr);
  else if (ext_device_fault_ext)
    chain(fault_ext);
  if (ext_address_binding)
    chain(address_binding);

  vkGetPhysicalDeviceFeatures2(candidate, &features);

  const bool has_indexing =
      out.core_12
          ? features12.descriptorIndexing &&
                features12.runtimeDescriptorArray &&
                features12.descriptorBindingPartiallyBound &&
                features12.descriptorBindingVariableDescriptorCount &&
                features12.descriptorBindingSampledImageUpdateAfterBind &&
                features12.descriptorBindingStorageImageUpdateAfterBind &&
                features12.descriptorBindingStorageBufferUpdateAfterBind &&
                features12.shaderSampledImageArrayNonUniformIndexing &&
                features12.shaderStorageImageArrayNonUniformIndexing &&
                features12.shaderStorageBufferArrayNonUniformIndexing
          : ext_descriptor_indexing && indexing.runtimeDescriptorArray &&
                indexing.descriptorBindingPartiallyBound &&
                indexing.descriptorBindingVariableDescriptorCount &&
                indexing.descriptorBindingSampledImageUpdateAfterBind &&
                indexing.descriptorBindingStorageImageUpdateAfterBind &&
                indexing.descriptorBindingStorageBufferUpdateAfterBind &&
                indexing.shaderSampledImageArrayNonUniformIndexing &&
                indexing.shaderStorageImageArrayNonUniformIndexing &&
                indexing.shaderStorageBufferArrayNonUniformIndexing;
  const bool has_scalar_layout =
      out.core_12 ? features12.scalarBlockLayout != 0
                  : ext_scalar_block_layout && scalar.scalarBlockLayout != 0;
  const bool has_address = out.core_12 ? features12.bufferDeviceAddress != 0
                                       : ext_buffer_device_address &&
                                             address.bufferDeviceAddress != 0;
  const bool has_timeline =
      out.core_12 ? features12.timelineSemaphore != 0
                  : ext_timeline_semaphore && timeline.timelineSemaphore != 0;
  const bool has_dynamic =
      out.core_13 ? features13.dynamicRendering != 0
                  : ext_dynamic_rendering &&
                        (out.core_12 ||
                         (ext_render_pass2 && ext_depth_stencil_resolve)) &&
                        dynamic.dynamicRendering != 0;
  const bool has_sync2 =
      out.core_13 ? features13.synchronization2 != 0
                  : ext_synchronization2 && sync2.synchronization2 != 0;

  const auto require = [&out](const bool have, const nx::string_view what) {
    if (!have)
      out.missing.push_back(what);
  };
  require(has_swapchain, "VK_KHR_swapchain");
  require(has_indexing, "descriptor indexing");
  require(has_address, "buffer device address");
  require(has_timeline, "timeline semaphores");
  require(has_scalar_layout, "scalar block layout");
  require(has_dynamic, "dynamic rendering");
  require(has_sync2, "synchronization2");
  require(draw_params.shaderDrawParameters != 0, "shader draw parameters");
  require(features.features.samplerAnisotropy != 0, "sampler anisotropy");
  require(features.features.multiDrawIndirect != 0, "multi-draw indirect");
  require(features.features.shaderInt16 != 0, "shaderInt16");
  // Device-address-backed NxPtr values are represented as uint64_t in Slang.
  // Buffer device address alone does not enable the SPIR-V Int64 capability.
  require(features.features.shaderInt64 != 0, "shaderInt64");
  require(features.features.independentBlend != 0, "independent blending");
  require(features.features.shaderStorageImageReadWithoutFormat != 0,
          "formatless storage-image reads");
  require(features.features.shaderStorageImageWriteWithoutFormat != 0,
          "formatless storage-image writes");

  out.draw_indirect_count = out.core_12 || ext_draw_indirect_count;
  out.storage_image_extended_formats =
      features.features.shaderStorageImageExtendedFormats != 0;
  if (ext_device_fault_khr && fault_khr.deviceFault) {
    out.device_fault = VulkanFaultRoute::Khr;
    out.device_fault_vendor_binary = fault_khr.deviceFaultVendorBinary != 0;
  } else if (ext_device_fault_ext && fault_ext.deviceFault) {
    out.device_fault = VulkanFaultRoute::Ext;
    out.device_fault_vendor_binary = fault_ext.deviceFaultVendorBinary != 0;
  }
  out.address_binding_report =
      ext_address_binding && address_binding.reportAddressBinding;
  out.host_query_reset = out.core_12 && features12.hostQueryReset;
  out.calibrated_timestamps = ext_calibrated_khr ? VulkanCalibrationRoute::Khr
                              : ext_calibrated_ext
                                  ? VulkanCalibrationRoute::Ext
                                  : VulkanCalibrationRoute::None;
  return out;
}

bool VulkanDevice::supports(const Format format,
                            const TextureUsageFlags usage) const {
  if (m_physical_device == VK_NULL_HANDLE || format == Format::Unknown)
    return false;

  VkFormatProperties props{};
  vkGetPhysicalDeviceFormatProperties(m_physical_device, to_vk_format(format),
                                      &props);
  const VkFormatFeatureFlags have = props.optimalTilingFeatures;

  VkFormatFeatureFlags want = 0;
  if (usage.has_any(TextureUsage::Sampled))
    want |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
  if (usage.has_any(TextureUsage::Storage))
    want |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
  if (usage.has_any(TextureUsage::RenderTarget))
    want |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
  if (usage.has_any(TextureUsage::DepthStencil))
    want |= VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
  return (have & want) == want;
}

VulkanDevice::~VulkanDevice() { shutdown(); }

bool VulkanDevice::create_instance(const DeviceDesc &desc) {
  u32 available_version = 0;
  NX_VK_TRY(vkEnumerateInstanceVersion(&available_version));
  if (available_version < REQUIRED_API_VERSION) {
    m_init_report.failure = DeviceInitFailure::Unsupported;
    m_init_error =
        nx::format("Vulkan {}.{} is available, but Vulkan {}.{} is required.",
                   VK_API_VERSION_MAJOR(available_version),
                   VK_API_VERSION_MINOR(available_version),
                   VK_API_VERSION_MAJOR(REQUIRED_API_VERSION),
                   VK_API_VERSION_MINOR(REQUIRED_API_VERSION));
    nx::logc(nx::log::rhi, "Vulkan {}.{} found, but {}.{} is required",
             VK_API_VERSION_MAJOR(available_version),
             VK_API_VERSION_MINOR(available_version),
             VK_API_VERSION_MAJOR(REQUIRED_API_VERSION),
             VK_API_VERSION_MINOR(REQUIRED_API_VERSION));
    return false;
  }
  m_instance_api_version = available_version;

  const nx::string app_name(desc.application_name);

  VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app_info.pApplicationName = app_name.c_str();
  app_info.applicationVersion = desc.application_version;
  app_info.pEngineName = "nx2d";
  app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
  app_info.apiVersion = available_version;

  u32 ext_count = 0;
  NX_VK_TRY(
      vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr));
  nx::small_vector<VkExtensionProperties, 64> available(ext_count);
  NX_VK_TRY(vkEnumerateInstanceExtensionProperties(nullptr, &ext_count,
                                                   available.data()));

  nx::small_vector<const char *, 8> extensions;
  extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
  if (const char *platform_ext = platform_surface_extension())
    extensions.push_back(platform_ext);

  m_debug_utils = (desc.enable_validation ||
                   desc.gpu_crash_mode != GpuCrashMode::Disabled) &&
                  has_extension(available, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  if (m_debug_utils)
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  nx::small_vector<const char *, 4> layers;
  if (desc.enable_validation) {
    u32 layer_count = 0;
    NX_VK_TRY(vkEnumerateInstanceLayerProperties(&layer_count, nullptr));
    nx::small_vector<VkLayerProperties, 32> available_layers(layer_count);
    NX_VK_TRY(vkEnumerateInstanceLayerProperties(&layer_count,
                                                 available_layers.data()));

    bool found = false;
    for (const auto &layer : available_layers) {
      if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
        layers.push_back(layer.layerName);
        nx::logi(nx::log::rhi, "validation layer enabled: {}", layer.layerName);
        found = true;
        break;
      }
    }
    if (!found) {
      nx::logw(nx::log::rhi,
               "validation requested but VK_LAYER_KHRONOS_validation is not "
               "installed; continuing without it");
    }

    if (found && desc.enable_sync_validation) {
      u32 count = 0;
      NX_VK_TRY(vkEnumerateInstanceExtensionProperties(
          "VK_LAYER_KHRONOS_validation", &count, nullptr));
      nx::small_vector<VkExtensionProperties, 32> layer_extensions(count);
      NX_VK_TRY(vkEnumerateInstanceExtensionProperties(
          "VK_LAYER_KHRONOS_validation", &count, layer_extensions.data()));
      for (const auto &extension : layer_extensions) {
        if (std::strcmp(extension.extensionName,
                        VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME) == 0) {
          extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
          m_sync_validation = true;
          break;
        }
      }
    }
  }

  VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  info.pApplicationInfo = &app_info;
  info.enabledExtensionCount = nx::cast<u32>(extensions.size());
  info.ppEnabledExtensionNames = extensions.data();
  info.enabledLayerCount = nx::cast<u32>(layers.size());
  info.ppEnabledLayerNames = layers.data();

  VkDebugUtilsMessengerCreateInfoEXT messenger_info = debug_messenger_info(
      m_gpu_crash, desc.gpu_crash_mode == GpuCrashMode::Detailed, false);
  if (m_debug_utils)
    info.pNext = &messenger_info;

  constexpr VkValidationFeatureEnableEXT SYNC_FEATURES[] = {
      VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT};
  VkValidationFeaturesEXT validation_features{
      VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
  validation_features.enabledValidationFeatureCount = 1;
  validation_features.pEnabledValidationFeatures = SYNC_FEATURES;
  if (m_sync_validation) {
    validation_features.pNext = info.pNext;
    info.pNext = &validation_features;
    nx::logi(nx::log::rhi, "synchronization validation enabled");
  }

  NX_VK_TRY(vkCreateInstance(&info, nullptr, &m_instance));

  // Loads every instance-level entry point; must precede any other vk* call.
  volkLoadInstance(m_instance);

  if (m_debug_utils)
    (void)rebuild_debug_messenger(false);
  return true;
}

bool VulkanDevice::rebuild_debug_messenger(
    const bool address_bindings) noexcept {
  if (!m_debug_utils || m_instance == VK_NULL_HANDLE)
    return !address_bindings;

  if (m_messenger != VK_NULL_HANDLE) {
    vkDestroyDebugUtilsMessengerEXT(m_instance, m_messenger, nullptr);
    m_messenger = VK_NULL_HANDLE;
  }

  VkDebugUtilsMessengerCreateInfoEXT info = debug_messenger_info(
      m_gpu_crash, m_gpu_crash.detailed(), address_bindings);
  const VkResult result =
      vkCreateDebugUtilsMessengerEXT(m_instance, &info, nullptr, &m_messenger);
  if (result == VK_SUCCESS)
    return true;

  nx::logw(nx::log::rhi, "could not create the {} debug messenger: {}",
           address_bindings ? "address-binding" : "Vulkan",
           vk_result_string(result));
  if (!address_bindings)
    return false;

  info = debug_messenger_info(m_gpu_crash, m_gpu_crash.detailed(), false);
  if (vkCreateDebugUtilsMessengerEXT(m_instance, &info, nullptr,
                                     &m_messenger) != VK_SUCCESS)
    m_messenger = VK_NULL_HANDLE;
  return false;
}

bool VulkanDevice::select_physical_device() {
  u32 count = 0;
  NX_VK_TRY(vkEnumeratePhysicalDevices(m_instance, &count, nullptr));
  if (count == 0) {
    m_init_report.failure = DeviceInitFailure::Unsupported;
    m_init_error = "No Vulkan graphics device was found.";
    nx::logc(nx::log::rhi, "no Vulkan physical devices found");
    return false;
  }

  nx::small_vector<VkPhysicalDevice, 8> devices(count);
  NX_VK_TRY(vkEnumeratePhysicalDevices(m_instance, &count, devices.data()));

  VkPhysicalDevice best = VK_NULL_HANDLE;
  u64 best_score = 0;
  nx::vector<nx::string> rejected;

  for (VkPhysicalDevice candidate : devices) {
    VkPhysicalDeviceProperties2 props{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    vkGetPhysicalDeviceProperties2(candidate, &props);
    if (m_init_report.vendor_id == 0) {
      m_init_report.vendor_id = props.properties.vendorID;
      m_init_report.device_id = props.properties.deviceID;
    }

    const u32 version = props.properties.apiVersion;
    if (version < REQUIRED_API_VERSION) {
      rejected.push_back(nx::format(
          "{}: Vulkan {}.{}, below the {}.{} floor",
          nx::string_view(props.properties.deviceName),
          VK_API_VERSION_MAJOR(version), VK_API_VERSION_MINOR(version),
          VK_API_VERSION_MAJOR(REQUIRED_API_VERSION),
          VK_API_VERSION_MINOR(REQUIRED_API_VERSION)));
      nx::logd(nx::log::rhi, "skipping {}", rejected.back());
      continue;
    }

    const DeviceRoutes routes =
        probe_routes(candidate, props.properties, m_instance_api_version);
    if (!routes.missing.empty()) {
      nx::string missing;
      for (const nx::string_view feature : routes.missing) {
        if (!missing.empty())
          missing.append(", ");
        missing.append(feature);
      }
      rejected.push_back(
          nx::format("{} (Vulkan {}.{}): missing {}",
                     nx::string_view(props.properties.deviceName),
                     VK_API_VERSION_MAJOR(version),
                     VK_API_VERSION_MINOR(version), missing));
      nx::logd(nx::log::rhi, "skipping {}", rejected.back());
      continue;
    }

    u64 score = 0;
    if (props.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
      score += 1ull << 40;
    else if (props.properties.deviceType ==
             VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
      score += 1ull << 39;

    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(candidate, &mem);
    for (u32 i = 0; i < mem.memoryHeapCount; ++i)
      if (mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
        score += mem.memoryHeaps[i].size;

    if (score > best_score) {
      best_score = score;
      best = candidate;
      m_routes = routes;
    }
  }

  if (best == VK_NULL_HANDLE) {
    m_init_report.failure = DeviceInitFailure::Unsupported;
    m_init_error =
        "No compatible Vulkan graphics device was found. Update the operating "
        "system or graphics driver.\n\n";
    for (const nx::string &why : rejected) {
      m_init_error.append("- ");
      m_init_error.append(why);
      m_init_error.append("\n");
    }
    nx::logc(nx::log::rhi,
             "no usable Vulkan device. The floor is {}.{} with dynamic "
             "rendering, synchronization2, timeline semaphores, buffer device "
             "address, descriptor indexing and scalar block layout - by core "
             "feature or by extension, whichever the device offers",
             VK_API_VERSION_MAJOR(REQUIRED_API_VERSION),
             VK_API_VERSION_MINOR(REQUIRED_API_VERSION));
    for (const nx::string &why : rejected)
      nx::loge(nx::log::rhi, "  {}", why);
    return false;
  }

  m_physical_device = best;

  VkPhysicalDeviceProperties selected_properties{};
  vkGetPhysicalDeviceProperties(m_physical_device, &selected_properties);
  m_init_report.vendor_id = selected_properties.vendorID;
  m_init_report.device_id = selected_properties.deviceID;

  VkPhysicalDeviceDescriptorIndexingProperties indexing{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES};
  VkPhysicalDeviceMaintenance3Properties maintenance{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
  indexing.pNext = &maintenance;
  VkPhysicalDeviceProperties2 props{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  props.pNext = &indexing;
  vkGetPhysicalDeviceProperties2(m_physical_device, &props);

  m_device_name = props.properties.deviceName;
  m_caps.device_name = m_device_name;
  m_caps.api_version = props.properties.apiVersion < m_instance_api_version
                           ? props.properties.apiVersion
                           : m_instance_api_version;
  m_caps.driver_version = props.properties.driverVersion;
  m_caps.vendor_id = props.properties.vendorID;
  m_caps.discrete =
      props.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
  m_caps.descriptor_indexing = true;
  m_caps.buffer_device_address = true;
  m_caps.dynamic_rendering = true;
  m_caps.synchronization2 = true;
  m_caps.timeline_semaphores = true;
  m_caps.max_texture_2d = props.properties.limits.maxImageDimension2D;
  m_caps.max_buffer_size = maintenance.maxMemoryAllocationSize;
  m_caps.max_compute_work_groups = {
      props.properties.limits.maxComputeWorkGroupCount[0],
      props.properties.limits.maxComputeWorkGroupCount[1],
      props.properties.limits.maxComputeWorkGroupCount[2],
  };
  const VkSampleCountFlags color_samples =
      props.properties.limits.framebufferColorSampleCounts &
      props.properties.limits.sampledImageColorSampleCounts;
  for (u32 sample_count = 64; sample_count >= 1; sample_count >>= 1)
    if ((color_samples & nx::cast<VkSampleCountFlags>(sample_count)) != 0) {
      m_caps.max_msaa_samples = sample_count;
      break;
    }
  m_caps.max_push_constant_size = props.properties.limits.maxPushConstantsSize;
  m_caps.timestamp_period_ns = props.properties.limits.timestampPeriod;
  m_caps.timestamps = m_caps.timestamp_period_ns > 0.f &&
                      props.properties.limits.timestampComputeAndGraphics;
  m_caps.parallel_recording = true;
  m_caps.max_bindless_sampled_images =
      indexing.maxDescriptorSetUpdateAfterBindSampledImages;
  m_caps.max_bindless_storage_images =
      indexing.maxDescriptorSetUpdateAfterBindStorageImages;
  m_caps.max_bindless_samplers =
      indexing.maxDescriptorSetUpdateAfterBindSamplers;

  VkPhysicalDeviceMemoryProperties mem{};
  vkGetPhysicalDeviceMemoryProperties(m_physical_device, &mem);
  for (u32 i = 0; i < mem.memoryHeapCount; ++i) {
    if (mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
      m_caps.device_local_memory += mem.memoryHeaps[i].size;
  }
  for (u32 i = 0; i < mem.memoryTypeCount; ++i) {
    if (mem.memoryTypes[i].propertyFlags &
        VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) {
      m_caps.memoryless_attachments = true;
      break;
    }
  }

  nx::logi(nx::log::rhi, "device: {} ({}, Vulkan {}.{}.{}, {} MiB local)",
           m_caps.device_name, m_caps.discrete ? "discrete" : "integrated",
           VK_API_VERSION_MAJOR(m_caps.api_version),
           VK_API_VERSION_MINOR(m_caps.api_version),
           VK_API_VERSION_PATCH(m_caps.api_version),
           m_caps.device_local_memory / (1024 * 1024));
  nx::logd(nx::log::rhi, "features: {}",
           m_routes.core_13   ? "core 1.3"
           : m_routes.core_12 ? "core 1.2, dynamic rendering and "
                                "synchronization2 by extension"
                              : "1.1, everything above it by extension");
  return true;
}

bool VulkanDevice::create_logical_device() {
  u32 family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &family_count,
                                           nullptr);
  nx::small_vector<VkQueueFamilyProperties, 8> families(family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &family_count,
                                           families.data());

  constexpr u32 INVALID_FAMILY = ~0u;
  u32 graphics = INVALID_FAMILY, compute = INVALID_FAMILY,
      copy = INVALID_FAMILY;

  for (u32 i = 0; i < family_count; ++i) {
    const VkQueueFlags flags = families[i].queueFlags;
    if (graphics == INVALID_FAMILY && (flags & VK_QUEUE_GRAPHICS_BIT))
      graphics = i;
    if ((flags & VK_QUEUE_COMPUTE_BIT) && !(flags & VK_QUEUE_GRAPHICS_BIT))
      compute = i;
    if ((flags & VK_QUEUE_TRANSFER_BIT) && !(flags & VK_QUEUE_GRAPHICS_BIT) &&
        !(flags & VK_QUEUE_COMPUTE_BIT))
      copy = i;
  }

  if (graphics == INVALID_FAMILY) {
    nx::logc(nx::log::rhi, "no graphics queue family");
    return false;
  }

  m_caps.async_compute = compute != INVALID_FAMILY;
  m_caps.dedicated_transfer = copy != INVALID_FAMILY;
  if (compute == INVALID_FAMILY)
    compute = graphics;
  if (copy == INVALID_FAMILY)
    copy = graphics;

  m_queue_families[nx::cast<usize>(QueueType::Graphics)] = graphics;
  m_queue_families[nx::cast<usize>(QueueType::Compute)] = compute;
  m_queue_families[nx::cast<usize>(QueueType::Copy)] = copy;

  constexpr f32 priority = 1.f;
  nx::small_vector<VkDeviceQueueCreateInfo, 3> queue_infos;
  for (const u32 family : {graphics, compute, copy}) {
    bool duplicate = false;
    for (const auto &existing : queue_infos)
      if (existing.queueFamilyIndex == family)
        duplicate = true;
    if (duplicate)
      continue;

    VkDeviceQueueCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    info.queueFamilyIndex = family;
    info.queueCount = 1;
    info.pQueuePriorities = &priority;
    queue_infos.push_back(info);
  }

  VkPhysicalDeviceFeatures2 features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  void **tail = &features.pNext;
  const PNextChainer chain{tail};

  // Core at the floor, so it needs no road of its own. Slang emits the
  // DrawParameters capability for SV_VertexID / SV_InstanceID.
  VkPhysicalDeviceShaderDrawParametersFeatures draw_params{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES};
  draw_params.shaderDrawParameters = VK_TRUE;
  chain(draw_params);

  VkPhysicalDeviceVulkan12Features features12{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceVulkan13Features features13{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceDescriptorIndexingFeatures indexing{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
  VkPhysicalDeviceBufferDeviceAddressFeatures address{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
  VkPhysicalDeviceTimelineSemaphoreFeatures timeline{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
  VkPhysicalDeviceScalarBlockLayoutFeatures scalar{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES};
  VkPhysicalDeviceDynamicRenderingFeatures dynamic{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES};
  VkPhysicalDeviceSynchronization2Features sync2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
  VkPhysicalDeviceFaultFeaturesKHR fault_khr{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_KHR};
  VkPhysicalDeviceFaultFeaturesEXT fault_ext{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT};
  VkPhysicalDeviceAddressBindingReportFeaturesEXT address_binding{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ADDRESS_BINDING_REPORT_FEATURES_EXT};

  nx::small_vector<const char *, 16> device_extensions;
  device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

  if (m_routes.core_12) {
    features12.timelineSemaphore = VK_TRUE;
    features12.bufferDeviceAddress = VK_TRUE;
    features12.descriptorIndexing = VK_TRUE;
    features12.runtimeDescriptorArray = VK_TRUE;
    features12.descriptorBindingPartiallyBound = VK_TRUE;
    features12.descriptorBindingVariableDescriptorCount = VK_TRUE;
    features12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    features12.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
    features12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    features12.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
    features12.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
    features12.drawIndirectCount =
        m_routes.draw_indirect_count ? VK_TRUE : VK_FALSE;
    features12.scalarBlockLayout = VK_TRUE;
    chain(features12);
  } else {
    indexing.runtimeDescriptorArray = VK_TRUE;
    indexing.descriptorBindingPartiallyBound = VK_TRUE;
    indexing.descriptorBindingVariableDescriptorCount = VK_TRUE;
    indexing.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    indexing.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
    indexing.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    indexing.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    indexing.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
    indexing.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
    address.bufferDeviceAddress = VK_TRUE;
    timeline.timelineSemaphore = VK_TRUE;
    scalar.scalarBlockLayout = VK_TRUE;
    chain(indexing);
    chain(address);
    chain(timeline);
    chain(scalar);
    device_extensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    device_extensions.push_back(VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME);
    if (m_routes.draw_indirect_count)
      device_extensions.push_back(VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME);
  }

  if (m_routes.core_13) {
    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;
    chain(features13);
  } else {
    dynamic.dynamicRendering = VK_TRUE;
    sync2.synchronization2 = VK_TRUE;
    chain(dynamic);
    chain(sync2);
    device_extensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
  }

  if (m_gpu_crash.enabled()) {
    if (m_routes.device_fault == VulkanFaultRoute::Khr) {
      fault_khr.deviceFault = VK_TRUE;
      fault_khr.deviceFaultVendorBinary =
          m_gpu_crash.detailed() && m_routes.device_fault_vendor_binary
              ? VK_TRUE
              : VK_FALSE;
      chain(fault_khr);
      device_extensions.push_back(VK_KHR_DEVICE_FAULT_EXTENSION_NAME);
    } else if (m_routes.device_fault == VulkanFaultRoute::Ext) {
      fault_ext.deviceFault = VK_TRUE;
      fault_ext.deviceFaultVendorBinary =
          m_gpu_crash.detailed() && m_routes.device_fault_vendor_binary
              ? VK_TRUE
              : VK_FALSE;
      chain(fault_ext);
      device_extensions.push_back(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
    }
    if (m_gpu_crash.detailed() && m_debug_utils &&
        m_routes.address_binding_report) {
      address_binding.reportAddressBinding = VK_TRUE;
      chain(address_binding);
      device_extensions.push_back(
          VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME);
    }
  }

#if NX_PROFILE_ENABLED
  if (m_routes.host_query_reset)
    features12.hostQueryReset = VK_TRUE;
  if (m_routes.calibrated_timestamps == VulkanCalibrationRoute::Khr)
    device_extensions.push_back(VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
  else if (m_routes.calibrated_timestamps == VulkanCalibrationRoute::Ext)
    device_extensions.push_back(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
#endif

  features.features.samplerAnisotropy = VK_TRUE;
  features.features.multiDrawIndirect = VK_TRUE;
  features.features.shaderInt16 = VK_TRUE;
  features.features.shaderInt64 = VK_TRUE;
  features.features.independentBlend = VK_TRUE;
  features.features.shaderStorageImageExtendedFormats =
      m_routes.storage_image_extended_formats ? VK_TRUE : VK_FALSE;

  features.features.shaderStorageImageReadWithoutFormat = VK_TRUE;
  features.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;

  const u32 video_family = find_video_decode_family();
  const bool try_video = video_family != INVALID_FAMILY &&
                         video_family != graphics && video_family != compute &&
                         video_family != copy;
  const usize base_extensions = device_extensions.size();
  const usize base_queues = queue_infos.size();
  VkPhysicalDeviceVideoDecodeVP9FeaturesKHR vp9_feature{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_DECODE_VP9_FEATURES_KHR};
  void **video_link = nullptr;
  if (try_video) {
    device_extensions.push_back(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME);
    VkDeviceQueueCreateInfo video_queue{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    video_queue.queueFamilyIndex = video_family;
    video_queue.queueCount = 1;
    video_queue.pQueuePriorities = &priority;
    queue_infos.push_back(video_queue);
    vp9_feature.videoDecodeVP9 = VK_TRUE;
    video_link = tail;
    chain(vp9_feature);
  }

  VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  info.pNext = &features;
  info.queueCreateInfoCount = nx::cast<u32>(queue_infos.size());
  info.pQueueCreateInfos = queue_infos.data();
  info.enabledExtensionCount = nx::cast<u32>(device_extensions.size());
  info.ppEnabledExtensionNames = device_extensions.data();

  bool video_enabled = try_video;
  VkResult created =
      vkCreateDevice(m_physical_device, &info, nullptr, &m_device);
  if (created != VK_SUCCESS && video_enabled) {
    nx::logw(nx::log::rhi,
             "driver rejected the video-decode setup; the device comes "
             "up without hardware decode");
    device_extensions.resize(base_extensions);
    queue_infos.resize(base_queues);
    if (video_link != nullptr)
      *video_link = nullptr;
    info.queueCreateInfoCount = nx::cast<u32>(queue_infos.size());
    info.pQueueCreateInfos = queue_infos.data();
    info.enabledExtensionCount = nx::cast<u32>(device_extensions.size());
    info.ppEnabledExtensionNames = device_extensions.data();
    video_enabled = false;
    created = vkCreateDevice(m_physical_device, &info, nullptr, &m_device);
  }
  if (created != VK_SUCCESS) {
    nx::logc(nx::log::rhi, "vkCreateDevice failed: {}", nx::cast<i32>(created));
    return false;
  }

  volkLoadDevice(m_device);
  if (!route_promoted_entry_points())
    return false;

  bool address_tracking = m_gpu_crash.enabled() && m_gpu_crash.detailed() &&
                          m_debug_utils && m_routes.address_binding_report;
  if (address_tracking)
    address_tracking = rebuild_debug_messenger(true);
  m_gpu_crash.attach(m_device, m_device_name, m_routes.device_fault,
                     m_gpu_crash.detailed() &&
                         m_routes.device_fault_vendor_binary,
                     address_tracking);
  nx::logd(nx::log::rhi, "GPU crash diagnostics: fault={} address-bindings={}",
           m_routes.device_fault == VulkanFaultRoute::Khr   ? "KHR"
           : m_routes.device_fault == VulkanFaultRoute::Ext ? "EXT"
                                                            : "basic",
           address_tracking ? "on" : "off");

  for (usize i = 0; i < nx::cast<usize>(QueueType::Count); ++i)
    vkGetDeviceQueue(m_device, m_queue_families[i], 0, &m_queues[i]);

  if (video_enabled) {
    m_video_decode.available = true;
    m_video_decode.family = video_family;
    vkGetDeviceQueue(m_device, video_family, 0, &m_video_decode.queue);
    nx::logd(nx::log::rhi, "video: hardware VP9 decode on queue family {}",
             video_family);
  }

  m_caps.draw_indirect_count = m_routes.draw_indirect_count;

  nx::logd(nx::log::rhi, "queues: graphics={} compute={}{} copy={}{}", graphics,
           compute, m_caps.async_compute ? " (async)" : " (aliased)", copy,
           m_caps.dedicated_transfer ? " (dedicated)" : " (aliased)");
  return true;
}

bool VulkanDevice::route_promoted_entry_points() noexcept {
  if (m_routes.core_12) {
    vkGetBufferDeviceAddressKHR =
        reinterpret_cast<PFN_vkGetBufferDeviceAddressKHR>(
            vkGetBufferDeviceAddress);
    vkGetSemaphoreCounterValueKHR =
        reinterpret_cast<PFN_vkGetSemaphoreCounterValueKHR>(
            vkGetSemaphoreCounterValue);
    vkWaitSemaphoresKHR =
        reinterpret_cast<PFN_vkWaitSemaphoresKHR>(vkWaitSemaphores);
  }
  if (m_routes.core_13) {
    vkCmdBeginRenderingKHR =
        reinterpret_cast<PFN_vkCmdBeginRenderingKHR>(vkCmdBeginRendering);
    vkCmdEndRenderingKHR =
        reinterpret_cast<PFN_vkCmdEndRenderingKHR>(vkCmdEndRendering);
    vkCmdPipelineBarrier2KHR =
        reinterpret_cast<PFN_vkCmdPipelineBarrier2KHR>(vkCmdPipelineBarrier2);
    vkCmdWriteTimestamp2KHR =
        reinterpret_cast<PFN_vkCmdWriteTimestamp2KHR>(vkCmdWriteTimestamp2);
    vkQueueSubmit2KHR = reinterpret_cast<PFN_vkQueueSubmit2KHR>(vkQueueSubmit2);
  }

  nx::small_vector<nx::string_view, 8> missing;
  const auto require = [&missing](const auto entry,
                                  const nx::string_view name) {
    if (entry == nullptr)
      missing.push_back(name);
  };
  require(vkGetBufferDeviceAddressKHR, "vkGetBufferDeviceAddressKHR");
  require(vkGetSemaphoreCounterValueKHR, "vkGetSemaphoreCounterValueKHR");
  require(vkWaitSemaphoresKHR, "vkWaitSemaphoresKHR");
  require(vkCmdBeginRenderingKHR, "vkCmdBeginRenderingKHR");
  require(vkCmdEndRenderingKHR, "vkCmdEndRenderingKHR");
  require(vkCmdPipelineBarrier2KHR, "vkCmdPipelineBarrier2KHR");
  require(vkCmdWriteTimestamp2KHR, "vkCmdWriteTimestamp2KHR");
  require(vkQueueSubmit2KHR, "vkQueueSubmit2KHR");
  if (missing.empty())
    return true;

  m_init_error = "The Vulkan driver is missing required device commands: ";
  for (usize i = 0; i < missing.size(); ++i) {
    if (i != 0)
      m_init_error.append(", ");
    m_init_error.append(missing[i]);
  }
  m_init_error.push_back('.');
  nx::logc(nx::log::rhi, "{}", m_init_error);
  return false;
}

bool VulkanDevice::create_allocator() {
  // volk loaded the entry points, so VMA must be handed them explicitly
  // rather than resolving its own (VMA_STATIC/DYNAMIC_VULKAN_FUNCTIONS are 0).
  VmaVulkanFunctions functions{};
  functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

  VmaAllocatorCreateInfo info{};
  info.physicalDevice = m_physical_device;
  info.device = m_device;
  info.instance = m_instance;
  info.vulkanApiVersion = m_caps.api_version;
  info.pVulkanFunctions = &functions;
  // Buffers can expose GPU addresses, so the allocator must allocate memory
  // with the matching flag.
  info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

  NX_VK_DEVICE_TRY(*this, vmaCreateAllocator(&info, &m_allocator));
  return true;
}

bool VulkanDevice::create_frame_contexts(const u32 frames_in_flight) {
  if (!m_frames.empty())
    return true;

  m_frames_in_flight =
      frames_in_flight < 1 ? 1 : (frames_in_flight > 3 ? 3 : frames_in_flight);

  m_frames.clear();
  m_frames.reserve(m_frames_in_flight);
  for (u32 i = 0; i < m_frames_in_flight; ++i) {
    FrameContext &frame = m_frames.emplace_back();

    VkSemaphoreCreateInfo binary_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    NX_VK_DEVICE_TRY(*this, vkCreateSemaphore(m_device, &binary_info, nullptr,
                                              &frame.acquire_semaphore));
    set_object_name(vk_handle_value(frame.acquire_semaphore),
                    VK_OBJECT_TYPE_SEMAPHORE,
                    nx::format("frame {} acquire", i));
  }
  return true;
}

bool VulkanDevice::init(const DeviceDesc &desc) {
  m_init_error.clear();
  m_init_report = {.failure = DeviceInitFailure::Failed};
  m_gpu_crash.configure(desc.gpu_crash_mode, desc.break_on_validation_error);
  m_report_frame.store(0, std::memory_order_relaxed);

  if (desc.native_driver.custom()) {
    const auto get_instance_proc_addr =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(
            desc.native_driver.find("vkGetInstanceProcAddr"));
    if (get_instance_proc_addr == nullptr) {
      m_init_error = nx::format(
          "Graphics driver '{}' does not export vkGetInstanceProcAddr.",
          desc.native_driver.identity);
      nx::logc(nx::log::rhi, "graphics driver '{}' has no vkGetInstanceProcAddr",
               desc.native_driver.identity);
      return false;
    }
    volkInitializeCustom(get_instance_proc_addr);
    if (vkCreateInstance == nullptr || vkEnumerateInstanceVersion == nullptr) {
      m_init_error = nx::format(
          "Graphics driver '{}' does not expose the required Vulkan loader "
          "entry points.",
          desc.native_driver.identity);
      nx::logc(nx::log::rhi, "graphics driver '{}' has an incomplete Vulkan loader",
               desc.native_driver.identity);
      return false;
    }
    nx::logi(nx::log::rhi, "using graphics driver '{}'",
             desc.native_driver.identity);
  } else {
    if (const VkResult result = volkInitialize(); result != VK_SUCCESS) {
      m_init_report.failure = DeviceInitFailure::Unsupported;
      m_init_error = nx::format(
          "No Vulkan loader was found ({}). Install a graphics driver with "
          "Vulkan support.",
          vk_result_string(result));
      nx::logc(nx::log::rhi,
               "no Vulkan loader found ({}). Install a GPU driver with Vulkan "
               "support.",
               vk_result_string(result));
      return false;
    }
  }

  if (!create_instance(desc) || !select_physical_device() ||
      !create_logical_device() || !create_allocator() ||
      !create_bindless_heap() ||
      !create_pipeline_cache(desc.pipeline_cache_path)) {
    if (m_init_error.empty())
      m_init_error =
          "The Vulkan graphics device could not be initialized. See the log "
          "for technical details.";
    return false;
  }

  VkSemaphoreTypeCreateInfo type_info{
      VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  type_info.initialValue = 0;
  VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  semaphore_info.pNext = &type_info;
  NX_VK_DEVICE_TRY(*this, vkCreateSemaphore(m_device, &semaphore_info, nullptr,
                                            &m_timeline));
  set_object_name(vk_handle_value(m_timeline), VK_OBJECT_TYPE_SEMAPHORE,
                  "frame timeline");

  if (!m_uploader.init(this)) {
    m_init_error = "The Vulkan upload service could not be initialized.";
    return false;
  }

  if (!create_frame_contexts(desc.frames_in_flight)) {
    m_init_error = "Vulkan frame resources could not be initialized.";
    return false;
  }

#if NX_PROFILE_ENABLED
  if (m_caps.timestamps) {
    PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT domains = nullptr;
    PFN_vkGetCalibratedTimestampsEXT calibrated = nullptr;
    if (m_routes.calibrated_timestamps == VulkanCalibrationRoute::Khr) {
      domains = vkGetPhysicalDeviceCalibrateableTimeDomainsKHR;
      calibrated = vkGetCalibratedTimestampsKHR;
    } else if (m_routes.calibrated_timestamps == VulkanCalibrationRoute::Ext) {
      domains = vkGetPhysicalDeviceCalibrateableTimeDomainsEXT;
      calibrated = vkGetCalibratedTimestampsEXT;
    }
    m_gpu_profiler.init(
        m_physical_device, m_device, graphics_queue(), graphics_family(),
        nx::format("{} graphics", m_device_name), domains, calibrated);

    u32 family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &family_count,
                                             nullptr);
    nx::small_vector<VkQueueFamilyProperties, 8> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &family_count,
                                             families.data());
    const u32 copy = m_queue_families[nx::cast<usize>(QueueType::Copy)];
    if (m_routes.host_query_reset && copy < family_count &&
        families[copy].timestampValidBits > 0)
      m_copy_profiler.init_host_reset(m_physical_device, m_device,
                                      nx::format("{} copy", m_device_name),
                                      domains, calibrated);
  }
#endif

  m_init_report.failure = DeviceInitFailure::None;
  return true;
}

void VulkanDevice::shutdown() noexcept {
  if (m_device != VK_NULL_HANDLE)
    wait_idle();

  m_uploader.shutdown();
  m_gpu_profiler.shutdown();
  m_copy_profiler.shutdown();

  destroy_swapchain();

  collect_garbage(~0ull);

  m_textures.for_each(
      [this](VulkanTexture &texture) { destroy_texture_now(texture); });
  m_textures.clear();
  for (VulkanBuffer &buf : m_buffers)
    if (buf.buffer != VK_NULL_HANDLE)
      vmaDestroyBuffer(m_allocator, buf.buffer, buf.allocation);
  m_buffers.clear();
  for (VulkanSampler &sampler : m_samplers)
    if (sampler.sampler != VK_NULL_HANDLE)
      vkDestroySampler(m_device, sampler.sampler, nullptr);
  m_samplers.clear();
  for (VulkanQueryPool &queries : m_query_pools)
    if (queries.pool != VK_NULL_HANDLE)
      vkDestroyQueryPool(m_device, queries.pool, nullptr);
  m_query_pools.clear();
  for (VulkanShader &shader : m_shaders)
    if (shader.module != VK_NULL_HANDLE)
      vkDestroyShaderModule(m_device, shader.module, nullptr);
  m_shaders.clear();
  for (VulkanPipeline &pipeline : m_pipelines)
    if (pipeline.pipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(m_device, pipeline.pipeline, nullptr);
  m_pipelines.clear();

  save_pipeline_cache();
  if (m_pipeline_cache != VK_NULL_HANDLE) {
    vkDestroyPipelineCache(m_device, m_pipeline_cache, nullptr);
    m_pipeline_cache = VK_NULL_HANDLE;
  }

  if (m_pipeline_layout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
    m_pipeline_layout = VK_NULL_HANDLE;
  }
  if (m_bindless.layout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(m_device, m_bindless.layout, nullptr);
  if (m_bindless.pool != VK_NULL_HANDLE)
    vkDestroyDescriptorPool(m_device, m_bindless.pool, nullptr);
  m_bindless = {};

  for (auto &frame : m_frames) {
    if (frame.acquire_semaphore != VK_NULL_HANDLE)
      vkDestroySemaphore(m_device, frame.acquire_semaphore, nullptr);
    if (frame.slot.pool != VK_NULL_HANDLE)
      vkDestroyCommandPool(m_device, frame.slot.pool, nullptr);
    for (const nx::unique_ptr<CommandSlot> &recorder : frame.recorders)
      if (recorder->pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(m_device, recorder->pool, nullptr);
  }
  m_frames.clear();

  if (m_timeline != VK_NULL_HANDLE) {
    vkDestroySemaphore(m_device, m_timeline, nullptr);
    m_timeline = VK_NULL_HANDLE;
  }

  if (m_allocator != VK_NULL_HANDLE) {
    vmaDestroyAllocator(m_allocator);
    m_allocator = VK_NULL_HANDLE;
  }
  if (m_device != VK_NULL_HANDLE) {
    vkDestroyDevice(m_device, nullptr);
    m_device = VK_NULL_HANDLE;
  }
  m_gpu_crash.reset();
  m_report_frame.store(0, std::memory_order_relaxed);
  if (m_messenger != VK_NULL_HANDLE) {
    vkDestroyDebugUtilsMessengerEXT(m_instance, m_messenger, nullptr);
    m_messenger = VK_NULL_HANDLE;
  }
  if (m_instance != VK_NULL_HANDLE) {
    vkDestroyInstance(m_instance, nullptr);
    m_instance = VK_NULL_HANDLE;
  }
}

void VulkanDevice::wait_idle() noexcept {
  if (m_device == VK_NULL_HANDLE || device_status() == GpuDeviceStatus::Lost)
    return;

  if (NX_VK_DEVICE_CHECK(*this, vkDeviceWaitIdle(m_device)) == VK_SUCCESS) {
    collect_garbage(~0ull);
    m_uploader.reclaim();
  }
}

VkResult VulkanDevice::check_result(const VkResult result,
                                    const char *const call,
                                    const char *const file,
                                    const i32 line) const noexcept {
  return m_gpu_crash.check(result, call, file, line,
                           m_report_frame.load(std::memory_order_acquire));
}

void VulkanDevice::flush_uploads() noexcept {
  if (m_device == VK_NULL_HANDLE || m_frame_active ||
      device_status() == GpuDeviceStatus::Lost)
    return;

  m_uploader.flush_batch();

  VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  pool_info.queueFamilyIndex =
      m_queue_families[nx::cast<usize>(QueueType::Graphics)];
  VkCommandPool pool = VK_NULL_HANDLE;
  if (NX_VK_DEVICE_CHECK(
          *this, vkCreateCommandPool(m_device, &pool_info, nullptr, &pool)) !=
      VK_SUCCESS)
    return;

  VkCommandBufferAllocateInfo alloc{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  alloc.commandPool = pool;
  alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (NX_VK_DEVICE_CHECK(*this, vkAllocateCommandBuffers(m_device, &alloc,
                                                         &cmd)) != VK_SUCCESS) {
    vkDestroyCommandPool(m_device, pool, nullptr);
    return;
  }

  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (NX_VK_DEVICE_CHECK(*this, vkBeginCommandBuffer(cmd, &begin)) !=
      VK_SUCCESS) {
    vkDestroyCommandPool(m_device, pool, nullptr);
    return;
  }
  const u64 upload_wait = m_uploader.record_acquires(cmd);
  if (NX_VK_DEVICE_CHECK(*this, vkEndCommandBuffer(cmd)) != VK_SUCCESS) {
    vkDestroyCommandPool(m_device, pool, nullptr);
    return;
  }

  ++m_timeline_value;
  m_report_frame.store(m_timeline_value, std::memory_order_release);
  const u64 value = m_timeline_value;

  VkSemaphoreSubmitInfo wait_info{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
  wait_info.semaphore = m_uploader.timeline();
  wait_info.value = upload_wait;
  wait_info.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

  VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
  signal.semaphore = m_timeline;
  signal.value = value;
  signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

  VkCommandBufferSubmitInfo cmd_info{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
  cmd_info.commandBuffer = cmd;

  VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  submit.waitSemaphoreInfoCount = upload_wait > 0 ? 1u : 0u;
  submit.pWaitSemaphoreInfos = &wait_info;
  submit.commandBufferInfoCount = 1;
  submit.pCommandBufferInfos = &cmd_info;
  submit.signalSemaphoreInfoCount = 1;
  submit.pSignalSemaphoreInfos = &signal;
  if (NX_VK_DEVICE_CHECK(
          *this,
          vkQueueSubmit2KHR(m_queues[nx::cast<usize>(QueueType::Graphics)], 1,
                            &submit, VK_NULL_HANDLE)) != VK_SUCCESS) {
    vkDestroyCommandPool(m_device, pool, nullptr);
    return;
  }

  VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  wait.semaphoreCount = 1;
  wait.pSemaphores = &m_timeline;
  wait.pValues = &value;
  if (NX_VK_DEVICE_CHECK(*this,
                         vkWaitSemaphoresKHR(m_device, &wait, UINT64_MAX)) !=
      VK_SUCCESS) {
    vkDestroyCommandPool(m_device, pool, nullptr);
    return;
  }

  vkDestroyCommandPool(m_device, pool, nullptr);
  m_uploader.reclaim();
}

void VulkanDevice::set_object_name(const u64 object, const VkObjectType type,
                                   const nx::string_view name) {
  if (!m_debug_utils || m_device == VK_NULL_HANDLE || object == 0 ||
      name.empty())
    return;

  const nx::string terminated(name);
  VkDebugUtilsObjectNameInfoEXT info{
      VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
  info.objectType = type;
  info.objectHandle = object;
  info.pObjectName = terminated.c_str();
  NX_VK_DEVICE_CHECK(*this, vkSetDebugUtilsObjectNameEXT(m_device, &info));
  m_gpu_crash.name_object(object, type, name);
}

void VulkanDevice::destroy_texture_now(VulkanTexture &texture) noexcept {
  for (const u32 slot : texture.uav_indices)
    free_rw_slot(slot);
  for (const VkImageView uav : texture.uavs)
    vkDestroyImageView(m_device, uav, nullptr);
  for (const VkImageView retired : texture.retired_srvs)
    vkDestroyImageView(m_device, retired, nullptr);
  if (texture.srv != VK_NULL_HANDLE)
    vkDestroyImageView(m_device, texture.srv, nullptr);
  if (texture.rtv != VK_NULL_HANDLE)
    vkDestroyImageView(m_device, texture.rtv, nullptr);
  if (texture.owns_image && texture.image != VK_NULL_HANDLE)
    vmaDestroyImage(m_allocator, texture.image, texture.allocation);
}

void VulkanDevice::collect_garbage(
    const u64 completed_timeline_value) noexcept {
  for (usize i = 0; i < m_pending_texture_views.size();) {
    const PendingTextureView pending = m_pending_texture_views[i];
    if (pending.publish_after > completed_timeline_value) {
      ++i;
      continue;
    }

    VulkanTexture *const texture = m_textures.try_get(pending.texture);
    if (texture == nullptr) {
      vkDestroyImageView(m_device, pending.view, nullptr);
    } else {
      if (texture->srv != VK_NULL_HANDLE)
        texture->retired_srvs.push_back(texture->srv);
      texture->srv = pending.view;
      texture->sampled_base_mip = pending.base_mip;
      write_texture_descriptor(nx::cast<u32>(pending.texture.index()),
                               texture->srv);
    }
    m_pending_texture_views.erase(m_pending_texture_views.begin() +
                                  nx::cast<isize>(i));
  }

  for (usize i = m_pending_deletes.size(); i > 0; --i) {
    PendingDelete &pending = m_pending_deletes[i - 1];
    if (pending.timeline_value > completed_timeline_value)
      continue;

    if (VulkanTexture *texture = m_textures.try_get(pending.texture)) {
      destroy_texture_now(*texture);
      m_textures.erase(pending.texture);
    }
    if (VulkanBuffer *buffer = m_buffers.try_get(pending.buffer)) {
      vmaDestroyBuffer(m_allocator, buffer->buffer, buffer->allocation);
      m_buffers.erase(pending.buffer);
    }
    if (VulkanSampler *sampler = m_samplers.try_get(pending.sampler)) {
      vkDestroySampler(m_device, sampler->sampler, nullptr);
      m_samplers.erase(pending.sampler);
    }
    if (VulkanPipeline *pipeline = m_pipelines.try_get(pending.pipeline)) {
      vkDestroyPipeline(m_device, pipeline->pipeline, nullptr);
      m_pipelines.erase(pending.pipeline);
    }

    m_pending_deletes.erase(m_pending_deletes.begin() + nx::cast<isize>(i - 1));
  }
}

} // namespace nxe::rhi
