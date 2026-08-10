#include "core/rendering/rhi/vulkan/vk_device.h"

#include <cstring>

#include "core/foundation/core/scope_guard.h"
#include "core/foundation/diagnostics/diagnostic.h"
#include "core/foundation/diagnostics/log.h"

#include <cstring>

namespace nxe::rhi {
namespace {

const nx::log::Category log_rhi = nx::log::category("rhi");

constexpr u32 REQUIRED_API_VERSION = VK_API_VERSION_1_1;

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*types*/,
    const VkDebugUtilsMessengerCallbackDataEXT *data, void *user_data) {
  if (data == nullptr || data->pMessage == nullptr)
    return VK_FALSE;

  if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    nx::loge(log_rhi, "{}", data->pMessage);
    if (user_data != nullptr && *nx::cast<bool *>(user_data))
      nx::dbg_break();
  } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    nx::logw(log_rhi, "{}", data->pMessage);
  } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    nx::logd(log_rhi, "{}", data->pMessage);
  }
  return VK_FALSE; // Never abort the offending call.
}

bool has_extension(const nx::small_vector<VkExtensionProperties, 64> &available,
                   const char *name) noexcept {
  for (const auto &ext : available)
    if (std::strcmp(ext.extensionName, name) == 0)
      return true;
  return false;
}

/// Platform surface extension for the compiled target.
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

[[nodiscard]] bool listed(const nx::small_vector<VkExtensionProperties, 128> &all,
                         const char *name) {
  for (const VkExtensionProperties &one : all)
    if (std::strcmp(one.extensionName, name) == 0)
      return true;
  return false;
}

} // namespace

/// Which road each capability takes on this device: core where its version has
/// it, an extension where one carries it, and neither is a device we skip.
DeviceRoutes VulkanDevice::probe_routes(VkPhysicalDevice candidate,
                                        const VkPhysicalDeviceProperties &props) {
  DeviceRoutes out;
  out.core_12 = props.apiVersion >= VK_API_VERSION_1_2;
  out.core_13 = props.apiVersion >= VK_API_VERSION_1_3;

  u32 count = 0;
  vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, nullptr);
  nx::small_vector<VkExtensionProperties, 128> available(count);
  vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count,
                                       available.data());

  const bool has_swapchain =
      listed(available, VK_KHR_SWAPCHAIN_EXTENSION_NAME);

  const bool ext_buffer_device_address =
      listed(available, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  const bool ext_timeline_semaphore =
      listed(available, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
  const bool ext_synchronization2 =
      listed(available, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
  const bool ext_dynamic_rendering =
      listed(available, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
  // Feature-only: no entry point of ours, so these are wanted by extension
  // solely where the version does not carry them.
  const bool ext_descriptor_indexing =
      listed(available, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
  const bool ext_scalar_block_layout =
      listed(available, VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME);
  const bool ext_draw_indirect_count =
      listed(available, VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME);
  // Dynamic rendering is defined in terms of these two; below 1.2 they are
  // extensions of their own and have to be enabled with it.
  const bool ext_render_pass2 =
      listed(available, VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
  const bool ext_depth_stencil_resolve =
      listed(available, VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME);

  // A feature struct may only be chained when the version or the extension that
  // defines it is there, so the chain is built from what this device has.
  VkPhysicalDeviceFeatures2 features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  void **tail = &features.pNext;
  const auto chain = [&tail](auto &block) {
    *tail = &block;
    tail = const_cast<void **>(&block.pNext);
  };

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

  vkGetPhysicalDeviceFeatures2(candidate, &features);

  const bool has_indexing =
      out.core_12 ? features12.descriptorIndexing &&
                        features12.runtimeDescriptorArray &&
                        features12.descriptorBindingPartiallyBound &&
                        features12.descriptorBindingSampledImageUpdateAfterBind &&
                        features12.shaderSampledImageArrayNonUniformIndexing
                  : ext_descriptor_indexing && indexing.runtimeDescriptorArray &&
                        indexing.descriptorBindingPartiallyBound &&
                        indexing.descriptorBindingSampledImageUpdateAfterBind &&
                        indexing.shaderSampledImageArrayNonUniformIndexing;
  const bool has_scalar_layout =
      out.core_12 ? features12.scalarBlockLayout != 0
                  : ext_scalar_block_layout && scalar.scalarBlockLayout != 0;
  // The four that carry an entry point need the extension as well as the
  // feature, because the engine calls the KHR name and only an enabled
  // extension resolves it.
  const bool has_address =
      ext_buffer_device_address && (out.core_12 ? features12.bufferDeviceAddress != 0
                                                : address.bufferDeviceAddress != 0);
  const bool has_timeline =
      ext_timeline_semaphore && (out.core_12 ? features12.timelineSemaphore != 0
                                             : timeline.timelineSemaphore != 0);
  const bool has_dynamic =
      ext_dynamic_rendering &&
      (out.core_12 || (ext_render_pass2 && ext_depth_stencil_resolve)) &&
      (out.core_13 ? features13.dynamicRendering != 0
                   : dynamic.dynamicRendering != 0);
  const bool has_sync2 =
      ext_synchronization2 && (out.core_13 ? features13.synchronization2 != 0
                                           : sync2.synchronization2 != 0);

  const auto require = [&out](const bool have, const nx::string_view what) {
    if (!have && out.missing.empty())
      out.missing = what;
  };
  require(has_swapchain, "VK_KHR_swapchain");
  require(has_indexing, "descriptor indexing");
  require(has_address, "buffer device address");
  require(has_timeline, "timeline semaphores");
  require(has_scalar_layout, "scalar block layout");
  require(has_dynamic, "dynamic rendering");
  require(has_sync2, "synchronization2");

  out.draw_indirect_count = out.core_12 || ext_draw_indirect_count;
  return out;
}

VulkanDevice::~VulkanDevice() { shutdown(); }

// Instance

bool VulkanDevice::create_instance(const DeviceDesc &desc) {
  u32 available_version = 0;
  NX_VK_TRY(vkEnumerateInstanceVersion(&available_version));
  if (available_version < REQUIRED_API_VERSION) {
    nx::logc(log_rhi, "Vulkan {}.{} found, but {}.{} is required",
             VK_API_VERSION_MAJOR(available_version),
             VK_API_VERSION_MINOR(available_version),
             VK_API_VERSION_MAJOR(REQUIRED_API_VERSION),
             VK_API_VERSION_MINOR(REQUIRED_API_VERSION));
    return false;
  }

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

  m_debug_utils = desc.enable_validation &&
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
        nx::logi(log_rhi, "validation layer enabled: {}", layer.layerName);
        found = true;
        break;
      }
    }
    if (!found) {
      // Not fatal: the layer ships with the SDK, which a player will not have.
      nx::logw(log_rhi,
               "validation requested but VK_LAYER_KHRONOS_validation is not "
               "installed; continuing without it");
    }

    // VK_EXT_validation_features comes from the layer, so it is absent from the
    // global extension list and has to be queried against the layer by name.
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

  // Chaining the messenger onto instance creation catches errors raised during
  // vkCreateInstance itself, which a later vkCreateDebugUtilsMessengerEXT
  // would miss.
  VkDebugUtilsMessengerCreateInfoEXT messenger_info{
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
  messenger_info.messageSeverity =
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
  messenger_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  messenger_info.pfnUserCallback = debug_callback;
  messenger_info.pUserData = &m_break_on_error;
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
    nx::logi(log_rhi, "synchronization validation enabled");
  }

  NX_VK_TRY(vkCreateInstance(&info, nullptr, &m_instance));

  // Loads every instance-level entry point; must precede any other vk* call.
  volkLoadInstance(m_instance);

  if (m_debug_utils) {
    NX_VK_CHECK(vkCreateDebugUtilsMessengerEXT(m_instance, &messenger_info,
                                               nullptr, &m_messenger));
  }
  return true;
}

// Physical device

bool VulkanDevice::select_physical_device() {
  u32 count = 0;
  NX_VK_TRY(vkEnumeratePhysicalDevices(m_instance, &count, nullptr));
  if (count == 0) {
    nx::logc(log_rhi, "no Vulkan physical devices found");
    return false;
  }

  nx::small_vector<VkPhysicalDevice, 8> devices(count);
  NX_VK_TRY(vkEnumeratePhysicalDevices(m_instance, &count, devices.data()));

  VkPhysicalDevice best = VK_NULL_HANDLE;
  u64 best_score = 0;

  for (VkPhysicalDevice candidate : devices) {
    VkPhysicalDeviceProperties2 props{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    vkGetPhysicalDeviceProperties2(candidate, &props);

    if (props.properties.apiVersion < REQUIRED_API_VERSION) {
      nx::logd(log_rhi, "skipping {}: Vulkan {}.{}, below the {}.{} floor",
               props.properties.deviceName,
               VK_API_VERSION_MAJOR(props.properties.apiVersion),
               VK_API_VERSION_MINOR(props.properties.apiVersion),
               VK_API_VERSION_MAJOR(REQUIRED_API_VERSION),
               VK_API_VERSION_MINOR(REQUIRED_API_VERSION));
      continue;
    }

    const DeviceRoutes routes = probe_routes(candidate, props.properties);
    if (!routes.missing.empty()) {
      nx::logd(log_rhi, "skipping {} (Vulkan {}.{}): no {}",
               props.properties.deviceName,
               VK_API_VERSION_MAJOR(props.properties.apiVersion),
               VK_API_VERSION_MINOR(props.properties.apiVersion),
               routes.missing);
      continue;
    }

    // Prefer discrete.
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
    nx::logc(log_rhi,
             "no device supports the required Vulkan 1.3 feature set "
             "(dynamic rendering, sync2, timeline semaphores, buffer device "
             "address, descriptor indexing)");
    return false;
  }

  m_physical_device = best;

  VkPhysicalDeviceDescriptorIndexingProperties indexing{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES};
  VkPhysicalDeviceProperties2 props{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  props.pNext = &indexing;
  vkGetPhysicalDeviceProperties2(m_physical_device, &props);

  m_device_name = props.properties.deviceName;
  m_caps.device_name = m_device_name;
  m_caps.api_version = props.properties.apiVersion;
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
  const VkSampleCountFlags color_samples =
      props.properties.limits.framebufferColorSampleCounts &
      props.properties.limits.sampledImageColorSampleCounts;
  for (u32 count = 64; count >= 1; count >>= 1)
    if ((color_samples & nx::cast<VkSampleCountFlags>(count)) != 0) {
      m_caps.max_msaa_samples = count;
      break;
    }
  m_caps.max_push_constant_size = props.properties.limits.maxPushConstantsSize;
  m_caps.timestamp_period_ns = props.properties.limits.timestampPeriod;
  m_caps.timestamps = m_caps.timestamp_period_ns > 0.f &&
                      props.properties.limits.timestampComputeAndGraphics;
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
    // A lazily-allocated memory type is how a tiler reports that transient
    // attachments never need real backing storage.
  }
  for (u32 i = 0; i < mem.memoryTypeCount; ++i) {
    if (mem.memoryTypes[i].propertyFlags &
        VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) {
      m_caps.memoryless_attachments = true;
      break;
    }
  }

  nx::logi(log_rhi, "device: {} ({}, Vulkan {}.{}.{}, {} MiB local)",
           m_caps.device_name, m_caps.discrete ? "discrete" : "integrated",
           VK_API_VERSION_MAJOR(m_caps.api_version),
           VK_API_VERSION_MINOR(m_caps.api_version),
           VK_API_VERSION_PATCH(m_caps.api_version),
           m_caps.device_local_memory / (1024 * 1024));
  return true;
}

// Logical device

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
    // Prefer families that are *only* compute / only transfer: those are the
    // genuinely asynchronous hardware queues.
    if ((flags & VK_QUEUE_COMPUTE_BIT) && !(flags & VK_QUEUE_GRAPHICS_BIT))
      compute = i;
    if ((flags & VK_QUEUE_TRANSFER_BIT) && !(flags & VK_QUEUE_GRAPHICS_BIT) &&
        !(flags & VK_QUEUE_COMPUTE_BIT))
      copy = i;
  }

  if (graphics == INVALID_FAMILY) {
    nx::logc(log_rhi, "no graphics queue family");
    return false;
  }

  m_caps.async_compute = compute != INVALID_FAMILY;
  m_caps.dedicated_transfer = copy != INVALID_FAMILY;
  // Alias onto graphics where the hardware has nothing dedicated, so callers
  // never have to ask whether a queue is "real".
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
  const auto chain = [&tail](auto &block) {
    *tail = &block;
    tail = const_cast<void **>(&block.pNext);
  };

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

  nx::small_vector<const char *, 12> device_extensions;
  device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  // Enabled whatever the version, because the engine calls the KHR entry points
  // and only an enabled extension resolves them. Where the version has these in
  // core the enable is a formality; where it does not, it is the whole point.
  device_extensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  device_extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
  device_extensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
  device_extensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);

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
    features12.drawIndirectCount = m_routes.draw_indirect_count ? VK_TRUE : VK_FALSE;
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
    // Feature-only below 1.2, so wanted here and nowhere else.
    device_extensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    device_extensions.push_back(VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME);
    if (m_routes.draw_indirect_count)
      device_extensions.push_back(VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME);
    // What dynamic rendering is defined in terms of. Core from 1.2 up.
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
  }

  features.features.samplerAnisotropy = VK_TRUE;
  features.features.multiDrawIndirect = VK_TRUE;
  features.features.shaderInt16 = VK_TRUE;
  features.features.independentBlend = VK_TRUE;
  // Lets one bindless rwTextures[] array hold images of differing formats: the
  // shader declares the type, the descriptor carries no format.
  features.features.shaderStorageImageReadWithoutFormat = VK_TRUE;
  features.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;

  VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  info.pNext = &features;
  info.queueCreateInfoCount = nx::cast<u32>(queue_infos.size());
  info.pQueueCreateInfos = queue_infos.data();
  info.enabledExtensionCount = nx::cast<u32>(device_extensions.size());
  info.ppEnabledExtensionNames = device_extensions.data();

  NX_VK_TRY(vkCreateDevice(m_physical_device, &info, nullptr, &m_device));

  // Swaps every entry point for the device-specific one, skipping the loader's
  // dispatch trampoline. Worth a measurable amount on call-heavy frames.
  volkLoadDevice(m_device);

  for (usize i = 0; i < nx::cast<usize>(QueueType::Count); ++i)
    vkGetDeviceQueue(m_device, m_queue_families[i], 0, &m_queues[i]);

  m_caps.draw_indirect_count = m_routes.draw_indirect_count;

  nx::logi(log_rhi, "queues: graphics={} compute={}{} copy={}{}", graphics,
           compute, m_caps.async_compute ? " (async)" : " (aliased)", copy,
           m_caps.dedicated_transfer ? " (dedicated)" : " (aliased)");
  return true;
}

bool VulkanDevice::create_allocator() {
  // volk loaded the entry points, so VMA must be handed them explicitly
  // rather than resolving its own (VMA_STATIC/DYNAMIC_VULKAN_FUNCTIONS are 0).
  VmaVulkanFunctions functions{};
  functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

  // TODO: Extend initialization here with vendor dependent extensions and other
  // useful ones
  VmaAllocatorCreateInfo info{};
  info.physicalDevice = m_physical_device;
  info.device = m_device;
  info.instance = m_instance;
  info.vulkanApiVersion = REQUIRED_API_VERSION;
  info.pVulkanFunctions = &functions;
  // Buffers can expose GPU addresses, so the allocator must allocate memory
  // with the matching flag.
  info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

  NX_VK_TRY(vmaCreateAllocator(&info, &m_allocator));
  return true;
}

bool VulkanDevice::create_frame_contexts(const u32 frames_in_flight) {
  // Idempotent: the swapchain is rebuilt whenever the OS hands us a new
  // surface (every Android resume), but the frame contexts outlive it and must
  // not be recreated underneath in-flight work. The timeline they signal is
  // older still - init() makes it, since headless submits need it too.
  if (!m_frames.empty())
    return true;

  m_frames_in_flight =
      frames_in_flight < 1 ? 1 : (frames_in_flight > 3 ? 3 : frames_in_flight);

  m_frames.clear();
  for (u32 i = 0; i < m_frames_in_flight; ++i) {
    FrameContext frame;

    // The slot and its pool are created on demand by open_slot().
    VkSemaphoreCreateInfo binary_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    NX_VK_TRY(vkCreateSemaphore(m_device, &binary_info, nullptr,
                                &frame.acquire_semaphore));

    m_frames.push_back(std::move(frame));
  }
  return true;
}

// Lifetime

bool VulkanDevice::init(const DeviceDesc &desc) {
  m_break_on_error = desc.break_on_validation_error;

  // volk resolves the loader itself; a missing ICD shows up here rather than
  // as a link error, which is the whole point of using it.
  if (const VkResult result = volkInitialize(); result != VK_SUCCESS) {
    nx::logc(log_rhi,
             "no Vulkan loader found ({}). Install a GPU driver with Vulkan "
             "support.",
             vk_result_string(result));
    return false;
  }

  if (!create_instance(desc))
    return false;
  if (!select_physical_device())
    return false;
  if (!create_logical_device())
    return false;
  if (!create_allocator())
    return false;
  if (!create_bindless_heap())
    return false;
  if (!create_pipeline_cache(desc.pipeline_cache_path))
    return false;

  // Device-scoped, not swapchain-scoped: it is what says a graphics submit has
  // finished, and a headless device (a test, an asset tool) makes those too -
  // flush_uploads() signals it with no frame contexts anywhere in sight.
  VkSemaphoreTypeCreateInfo type_info{
      VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  type_info.initialValue = 0;
  VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  semaphore_info.pNext = &type_info;
  NX_VK_TRY(vkCreateSemaphore(m_device, &semaphore_info, nullptr, &m_timeline));
  set_object_name(vk_handle_value(m_timeline), VK_OBJECT_TYPE_SEMAPHORE,
                  "frame timeline");

  if (!m_uploader.init(this))
    return false;

  if (!create_frame_contexts(desc.frames_in_flight))
    return false;

  return true;
}

void VulkanDevice::shutdown() noexcept {
  if (m_device != VK_NULL_HANDLE)
    wait_idle();

  // Before the allocator and device it borrows go away.
  m_uploader.shutdown();

  destroy_swapchain();

  // Everything is idle, so the deferred queue can be drained unconditionally.
  collect_garbage(~0ull);

  // Retire whatever the caller left behind.
  for (VulkanTexture &texture : m_textures)
    destroy_texture_now(texture);
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
    // Destroying a pool frees the command buffers allocated from it.
    if (frame.slot.pool != VK_NULL_HANDLE)
      vkDestroyCommandPool(m_device, frame.slot.pool, nullptr);
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
  if (m_device != VK_NULL_HANDLE)
    NX_VK_CHECK(vkDeviceWaitIdle(m_device));
}

void VulkanDevice::flush_uploads() noexcept {
  if (m_device == VK_NULL_HANDLE || m_frame_active)
    return;

  m_uploader.flush_batch();

  // One-shot pool: this is an offline path, not something a frame runs, so
  // there is nothing to amortise a persistent pool over.
  VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  pool_info.queueFamilyIndex =
      m_queue_families[nx::cast<usize>(QueueType::Graphics)];
  VkCommandPool pool = VK_NULL_HANDLE;
  if (NX_VK_CHECK(vkCreateCommandPool(m_device, &pool_info, nullptr, &pool)) !=
      VK_SUCCESS)
    return;

  VkCommandBufferAllocateInfo alloc{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  alloc.commandPool = pool;
  alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (NX_VK_CHECK(vkAllocateCommandBuffers(m_device, &alloc, &cmd)) !=
      VK_SUCCESS) {
    vkDestroyCommandPool(m_device, pool, nullptr);
    return;
  }

  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &begin);
  // Bumping the timeline only *after* this call keeps the uploader's view of
  // "the frame about to be recorded" the same as it is under begin_frame,
  // which is what its staging holds are pinned to.
  const u64 upload_wait = m_uploader.record_acquires(cmd);
  NX_VK_CHECK(vkEndCommandBuffer(cmd));

  ++m_timeline_value;
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
  NX_VK_CHECK(vkQueueSubmit2(m_queues[nx::cast<usize>(QueueType::Graphics)], 1,
                             &submit, VK_NULL_HANDLE));

  VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  wait.semaphoreCount = 1;
  wait.pSemaphores = &m_timeline;
  wait.pValues = &value;
  NX_VK_CHECK(vkWaitSemaphores(m_device, &wait, UINT64_MAX));

  vkDestroyCommandPool(m_device, pool, nullptr);
  m_uploader.reclaim();
}

void VulkanDevice::set_object_name(const u64 object, const VkObjectType type,
                                   const nx::string_view name) {
  if (!m_debug_utils || m_device == VK_NULL_HANDLE)
    return;

  const nx::string terminated(name);
  VkDebugUtilsObjectNameInfoEXT info{
      VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
  info.objectType = type;
  info.objectHandle = object;
  info.pObjectName = terminated.c_str();
  NX_VK_CHECK(vkSetDebugUtilsObjectNameEXT(m_device, &info));
}

void VulkanDevice::destroy_texture_now(VulkanTexture &texture) noexcept {
  for (const u32 slot : texture.uav_indices)
    free_rw_slot(slot);
  for (const VkImageView uav : texture.uavs)
    vkDestroyImageView(m_device, uav, nullptr);
  if (texture.srv != VK_NULL_HANDLE)
    vkDestroyImageView(m_device, texture.srv, nullptr);
  if (texture.rtv != VK_NULL_HANDLE)
    vkDestroyImageView(m_device, texture.rtv, nullptr);
  if (texture.owns_image && texture.image != VK_NULL_HANDLE)
    vmaDestroyImage(m_allocator, texture.image, texture.allocation);
}

void VulkanDevice::collect_garbage(
    const u64 completed_timeline_value) noexcept {
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
