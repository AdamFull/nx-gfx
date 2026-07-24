#include "rhi/vulkan/vk_device.h"

#include "foundation/core/scope_guard.h"
#include "foundation/diagnostics/diagnostic.h"
#include "foundation/diagnostics/log.h"

#include <cstring>

namespace nxe::rhi {
namespace {

const nx::log::Category log_rhi = nx::log::category("rhi");

constexpr u32 REQUIRED_API_VERSION = VK_API_VERSION_1_3;

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

} // namespace

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
  app_info.apiVersion = REQUIRED_API_VERSION;

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

    // Chained feature query: everything the bindless model needs is core in
    // 1.2/1.3, so this is one call rather than an extension hunt.
    VkPhysicalDeviceVulkan13Features features13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features features12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    features12.pNext = &features13;
    VkPhysicalDeviceFeatures2 features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &features12;
    vkGetPhysicalDeviceFeatures2(candidate, &features);

    const bool usable =
        props.properties.apiVersion >= REQUIRED_API_VERSION &&
        features13.dynamicRendering && features13.synchronization2 &&
        features12.timelineSemaphore && features12.bufferDeviceAddress &&
        features12.descriptorIndexing && features12.runtimeDescriptorArray &&
        features12.descriptorBindingPartiallyBound &&
        features12.descriptorBindingSampledImageUpdateAfterBind &&
        features12.shaderSampledImageArrayNonUniformIndexing;

    if (!usable) {
      nx::logd(log_rhi, "skipping {}: missing required 1.3 features",
               props.properties.deviceName);
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
  m_caps.max_push_constant_size = props.properties.limits.maxPushConstantsSize;
  m_caps.timestamp_period_ns = props.properties.limits.timestampPeriod;
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

  VkPhysicalDeviceVulkan13Features features13{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  features13.dynamicRendering = VK_TRUE;
  features13.synchronization2 = VK_TRUE;
  features13.maintenance4 = VK_TRUE;

  VkPhysicalDeviceVulkan12Features features12{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  features12.pNext = &features13;
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
  features12.drawIndirectCount = VK_TRUE;
  features12.scalarBlockLayout = VK_TRUE;

  VkPhysicalDeviceFeatures2 features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features.pNext = &features12;
  features.features.samplerAnisotropy = VK_TRUE;
  features.features.multiDrawIndirect = VK_TRUE;
  features.features.shaderInt16 = VK_TRUE;
  features.features.independentBlend = VK_TRUE;

  const char *device_extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  info.pNext = &features;
  info.queueCreateInfoCount = nx::cast<u32>(queue_infos.size());
  info.pQueueCreateInfos = queue_infos.data();
  info.enabledExtensionCount = 1;
  info.ppEnabledExtensionNames = device_extensions;

  NX_VK_TRY(vkCreateDevice(m_physical_device, &info, nullptr, &m_device));

  // Swaps every entry point for the device-specific one, skipping the loader's
  // dispatch trampoline. Worth a measurable amount on call-heavy frames.
  volkLoadDevice(m_device);

  for (usize i = 0; i < nx::cast<usize>(QueueType::Count); ++i)
    vkGetDeviceQueue(m_device, m_queue_families[i], 0, &m_queues[i]);

  m_caps.draw_indirect_count = true;

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
  // surface (every Android resume), but the frame contexts and the timeline
  // outlive it and must not be recreated underneath in-flight work.
  if (!m_frames.empty())
    return true;

  m_frames_in_flight =
      frames_in_flight < 1 ? 1 : (frames_in_flight > 3 ? 3 : frames_in_flight);

  VkSemaphoreTypeCreateInfo type_info{
      VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  type_info.initialValue = 0;

  VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  semaphore_info.pNext = &type_info;
  NX_VK_TRY(vkCreateSemaphore(m_device, &semaphore_info, nullptr, &m_timeline));
  set_object_name(vk_handle_value(m_timeline), VK_OBJECT_TYPE_SEMAPHORE,
                  "frame timeline");

  m_frames.clear();
  for (u32 i = 0; i < m_frames_in_flight; ++i) {
    FrameContext frame;

    VkCommandPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    // Reset the whole pool each frame rather than individual buffers: it is
    // one call and lets the driver recycle the backing blocks wholesale.
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex =
        m_queue_families[nx::cast<usize>(QueueType::Graphics)];
    NX_VK_TRY(vkCreateCommandPool(m_device, &pool_info, nullptr,
                                  &frame.command_pool));

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

  return true;
}

void VulkanDevice::shutdown() noexcept {
  if (m_device != VK_NULL_HANDLE)
    wait_idle();

  destroy_swapchain();

  // Everything is idle, so the deferred queue can be drained unconditionally.
  collect_garbage(~0ull);

  for (auto &frame : m_frames) {
    if (frame.acquire_semaphore != VK_NULL_HANDLE)
      vkDestroySemaphore(m_device, frame.acquire_semaphore, nullptr);
    if (frame.command_pool != VK_NULL_HANDLE)
      vkDestroyCommandPool(m_device, frame.command_pool, nullptr);
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

void VulkanDevice::collect_garbage(
    const u64 completed_timeline_value) noexcept {
  for (usize i = m_pending_deletes.size(); i > 0; --i) {
    PendingDelete &pending = m_pending_deletes[i - 1];
    if (pending.timeline_value > completed_timeline_value)
      continue;

    if (pending.view != VK_NULL_HANDLE)
      vkDestroyImageView(m_device, pending.view, nullptr);
    if (pending.image != VK_NULL_HANDLE && pending.allocation != VK_NULL_HANDLE)
      vmaDestroyImage(m_allocator, pending.image, pending.allocation);
    if (pending.buffer != VK_NULL_HANDLE)
      vmaDestroyBuffer(m_allocator, pending.buffer, pending.allocation);

    m_pending_deletes.erase(m_pending_deletes.begin() + nx::cast<isize>(i - 1));
  }
}

} // namespace nxe::rhi
