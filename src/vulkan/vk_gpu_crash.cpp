#include "src/vulkan/vk_gpu_crash.h"

#include "core/foundation/diagnostics/crash.h"
#include "core/foundation/diagnostics/diagnostic.h"
#include "core/foundation/diagnostics/log.h"
#include "core/foundation/strings/format.h"

#include <cstring>

namespace nx::gfx::detail {
namespace {

constexpr usize MAX_FAULTS = 64;

nx::string_view address_type_name(const VkDeviceFaultAddressTypeKHR type) {
  switch (type) {
  case VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_KHR:
    return "invalid read";
  case VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_KHR:
    return "invalid write";
  case VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_KHR:
    return "invalid execute";
  case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_KHR:
    return "unknown instruction pointer";
  case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_INVALID_KHR:
    return "invalid instruction pointer";
  case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_KHR:
    return "faulting instruction pointer";
  default:
    return "unknown";
  }
}

nx::string_view object_type_name(const VkObjectType type) {
  switch (type) {
  case VK_OBJECT_TYPE_BUFFER:
    return "buffer";
  case VK_OBJECT_TYPE_IMAGE:
    return "image";
  case VK_OBJECT_TYPE_DEVICE_MEMORY:
    return "device memory";
  case VK_OBJECT_TYPE_IMAGE_VIEW:
    return "image view";
  case VK_OBJECT_TYPE_SAMPLER:
    return "sampler";
  case VK_OBJECT_TYPE_SHADER_MODULE:
    return "shader module";
  case VK_OBJECT_TYPE_PIPELINE_CACHE:
    return "pipeline cache";
  case VK_OBJECT_TYPE_PIPELINE_LAYOUT:
    return "pipeline layout";
  case VK_OBJECT_TYPE_PIPELINE:
    return "pipeline";
  case VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT:
    return "descriptor set layout";
  case VK_OBJECT_TYPE_DESCRIPTOR_POOL:
    return "descriptor pool";
  case VK_OBJECT_TYPE_DESCRIPTOR_SET:
    return "descriptor set";
  case VK_OBJECT_TYPE_COMMAND_POOL:
    return "command pool";
  case VK_OBJECT_TYPE_COMMAND_BUFFER:
    return "command buffer";
  case VK_OBJECT_TYPE_SEMAPHORE:
    return "semaphore";
  case VK_OBJECT_TYPE_QUERY_POOL:
    return "query pool";
  case VK_OBJECT_TYPE_SWAPCHAIN_KHR:
    return "swapchain";
  case VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR:
    return "acceleration structure";
  default:
    return "Vulkan object";
  }
}

template <usize N> void copy_text(char (&out)[N], const char *text) noexcept {
  if (text == nullptr) {
    out[0] = '\0';
    return;
  }
  usize i = 0;
  while (i + 1 < N && text[i] != '\0') {
    out[i] = text[i];
    ++i;
  }
  out[i] = '\0';
}

template <usize N>
void copy_text(char (&out)[N], const nx::string_view text) noexcept {
  const usize count = nx::min<usize>(N - 1, text.size());
  if (count != 0)
    std::memcpy(out, text.data(), count);
  out[count] = '\0';
}

void append_address(gfx::gpu_crash_report &out,
                    const VkDeviceFaultAddressInfoKHR &address) {
  if (address.addressType == VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_KHR)
    return;
  out.addresses.push_back({nx::string(address_type_name(address.addressType)),
                           address.reportedAddress, address.addressPrecision});
}

void append_vendor(gfx::gpu_crash_report &out,
                   const VkDeviceFaultVendorInfoKHR &vendor) {
  out.vendor_faults.push_back({nx::string(vendor.description),
                               vendor.vendorFaultCode, vendor.vendorFaultData});
}

} // namespace

void VulkanGpuCrashReporter::configure(
    const gfx::gpu_crash_mode mode, const bool break_on_validation_error,
    const nx::string_view application) noexcept {
  m_enabled = mode != gfx::gpu_crash_mode::Disabled;
  m_detailed = mode == gfx::gpu_crash_mode::Detailed;
  m_break_on_validation_error = break_on_validation_error;
  if (m_detailed)
    (void)m_aftermath.start(application);
}

void VulkanGpuCrashReporter::attach(const VkDevice device,
                                    const nx::string_view device_name,
                                    const VulkanFaultRoute route,
                                    const bool vendor_binary,
                                    const bool address_tracking) {
  m_device = device;
  m_device_name = device_name;
  m_route = route;
  m_vendor_binary = vendor_binary;
  m_address_tracking = address_tracking;
  m_lost.store(false, std::memory_order_release);
  m_reported.store(false, std::memory_order_release);
}

void VulkanGpuCrashReporter::reset() noexcept {
  m_device = VK_NULL_HANDLE;
  m_device_name.clear();
  m_route = VulkanFaultRoute::None;
  m_vendor_binary = false;
  m_address_tracking = false;
  m_lost.store(false, std::memory_order_release);
  m_reported.store(false, std::memory_order_release);
  const nx::scoped_lock lock(m_binding_lock);
  m_binding_sequence = 0;
  for (BindingEvent &binding : m_bindings)
    binding = {};
  m_object_sequence = 0;
  for (ObjectNameEvent &object : m_objects)
    object = {};
}

VkBool32 VulkanGpuCrashReporter::debug_message(
    const VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    const VkDebugUtilsMessengerCallbackDataEXT *const data) noexcept {
  if (data != nullptr && m_address_tracking) {
    for (const VkBaseInStructure *next =
             static_cast<const VkBaseInStructure *>(data->pNext);
         next != nullptr; next = next->pNext) {
      if (next->sType !=
          VK_STRUCTURE_TYPE_DEVICE_ADDRESS_BINDING_CALLBACK_DATA_EXT)
        continue;
      remember_binding(
          *data,
          *reinterpret_cast<const VkDeviceAddressBindingCallbackDataEXT *>(
              next));
    }
  }

  const char *const message =
      data != nullptr && data->pMessage != nullptr ? data->pMessage : nullptr;
  if (message == nullptr)
    return VK_FALSE;
  if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    nx::loge(nx::log::rhi, "{}", message);
    if (m_break_on_validation_error)
      nx::dbg_break();
  } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    nx::logw(nx::log::rhi, "{}", message);
  } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    nx::logd(nx::log::rhi, "{}", message);
  }
  return VK_FALSE;
}

void VulkanGpuCrashReporter::name_object(const u64 object,
                                         const VkObjectType type,
                                         const nx::string_view name) noexcept {
  if (!m_enabled || object == 0 || name.empty())
    return;

  const nx::scoped_lock lock(m_binding_lock);
  ObjectNameEvent &identity = m_objects[m_object_sequence % OBJECT_CAPACITY];
  identity = {};
  identity.sequence = ++m_object_sequence;
  identity.object = object;
  identity.object_type = type;
  copy_text(identity.name, name);

  for (BindingEvent &binding : m_bindings) {
    if (binding.sequence != 0 && binding.object == object &&
        binding.object_type == type)
      copy_text(binding.name, name);
  }
}

void VulkanGpuCrashReporter::remember_binding(
    const VkDebugUtilsMessengerCallbackDataEXT &data,
    const VkDeviceAddressBindingCallbackDataEXT &binding) {
  const nx::scoped_lock lock(m_binding_lock);
  BindingEvent &event = m_bindings[m_binding_sequence % BINDING_CAPACITY];
  event = {};
  event.sequence = ++m_binding_sequence;
  event.address = binding.baseAddress;
  event.size = binding.size;
  event.bound = binding.bindingType == VK_DEVICE_ADDRESS_BINDING_TYPE_BIND_EXT;
  event.internal =
      (binding.flags & VK_DEVICE_ADDRESS_BINDING_INTERNAL_OBJECT_BIT_EXT) != 0;
  if (data.objectCount != 0 && data.pObjects != nullptr) {
    const VkDebugUtilsObjectNameInfoEXT &object = data.pObjects[0];
    event.object = object.objectHandle;
    event.object_type = object.objectType;
    copy_text(event.name, object.pObjectName);
  }
}

VkResult VulkanGpuCrashReporter::check(const VkResult result,
                                       const char *const call,
                                       const char *const file, const i32 line,
                                       const u64 frame) noexcept {
  vk_report(result, call, file, line);
  if (result == VK_ERROR_DEVICE_LOST) {
    m_lost.store(true, std::memory_order_release);
  }
  if (m_enabled && result == VK_ERROR_DEVICE_LOST) {
    bool expected = false;
    if (m_reported.compare_exchange_strong(expected, true,
                                           std::memory_order_acq_rel)) {
      try {
        report_device_lost(call, file, line, frame);
      } catch (...) {
        nx::loge(nx::log::rhi, "GPU crash diagnostics could not be collected");
      }
    }
  }
  return result;
}

void VulkanGpuCrashReporter::report_device_lost(const char *const call,
                                                const char *const file,
                                                const i32 line,
                                                const u64 frame) {
  gfx::gpu_crash_report report;
  report.backend = gfx::gpu_backend::Vulkan;
  report.device = m_device_name;
  report.operation = call;
  report.result = "VK_ERROR_DEVICE_LOST";
  report.frame = frame;
  report.description = nx::format("device loss observed at {}:{}", file, line);

  if (m_device != VK_NULL_HANDLE) {
    if (m_route == VulkanFaultRoute::Khr)
      collect_khr(report);
    else if (m_route == VulkanFaultRoute::Ext)
      collect_ext(report);
  }
  append_objects_and_bindings(report);
  m_aftermath.collect(report);
  if (!report_gpu_crash(report))
    nx::loge(nx::log::rhi, "GPU crash report could not be persisted: {}",
             gpu_crash_json(report));
}

void VulkanGpuCrashReporter::collect_khr(gfx::gpu_crash_report &out) {
  if (vkGetDeviceFaultReportsKHR == nullptr)
    return;
  u32 count = 0;
  VkResult result = vkGetDeviceFaultReportsKHR(m_device, 0, &count, nullptr);
  if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
    out.description.append(
        nx::format("; vkGetDeviceFaultReportsKHR failed with {}",
                   vk_result_string(result)));
    return;
  }
  count = nx::min<u32>(count, nx::cast<u32>(MAX_FAULTS));
  nx::vector<VkDeviceFaultInfoKHR> faults(count);
  for (VkDeviceFaultInfoKHR &fault : faults)
    fault.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_KHR;
  if (count != 0) {
    result = vkGetDeviceFaultReportsKHR(m_device, 0, &count, faults.data());
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
      out.description.append(
          nx::format("; vkGetDeviceFaultReportsKHR failed with {}",
                     vk_result_string(result)));
      return;
    }
  }
  faults.resize(nx::min<usize>(faults.size(), count));
  for (const VkDeviceFaultInfoKHR &fault : faults) {
    if (out.description.empty())
      out.description = fault.description;
    else if (fault.description[0] != '\0')
      out.description.append(nx::format("; {}", fault.description));
    append_address(out, fault.faultAddressInfo);
    append_address(out, fault.instructionAddressInfo);
    if (fault.flags & VK_DEVICE_FAULT_FLAG_VENDOR_KHR)
      append_vendor(out, fault.vendorInfo);
  }

  if (!m_vendor_binary || vkGetDeviceFaultDebugInfoKHR == nullptr)
    return;
  VkDeviceFaultDebugInfoKHR debug{
      VK_STRUCTURE_TYPE_DEVICE_FAULT_DEBUG_INFO_KHR};
  result = vkGetDeviceFaultDebugInfoKHR(m_device, &debug);
  if (result != VK_SUCCESS || debug.vendorBinarySize == 0 ||
      debug.vendorBinarySize > nx::crash::ATTACHMENT_SIZE_CAPACITY)
    return;
  out.vendor_binary = nx::blob<u8>(debug.vendorBinarySize);
  debug.pVendorBinaryData = out.vendor_binary.data();
  result = vkGetDeviceFaultDebugInfoKHR(m_device, &debug);
  if (result != VK_SUCCESS)
    out.vendor_binary = {};
}

void VulkanGpuCrashReporter::collect_ext(gfx::gpu_crash_report &out) {
  if (vkGetDeviceFaultInfoEXT == nullptr)
    return;
  VkDeviceFaultCountsEXT counts{VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
  VkResult result = vkGetDeviceFaultInfoEXT(m_device, &counts, nullptr);
  if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
    out.description.append(nx::format(
        "; vkGetDeviceFaultInfoEXT failed with {}", vk_result_string(result)));
    return;
  }

  counts.addressInfoCount =
      nx::min<u32>(counts.addressInfoCount, nx::cast<u32>(MAX_FAULTS));
  counts.vendorInfoCount =
      nx::min<u32>(counts.vendorInfoCount, nx::cast<u32>(MAX_FAULTS));
  if (!m_vendor_binary ||
      counts.vendorBinarySize > nx::crash::ATTACHMENT_SIZE_CAPACITY)
    counts.vendorBinarySize = 0;
  nx::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
  nx::vector<VkDeviceFaultVendorInfoEXT> vendors(counts.vendorInfoCount);
  out.vendor_binary = nx::blob<u8>(nx::cast<usize>(counts.vendorBinarySize));

  VkDeviceFaultInfoEXT info{VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
  info.pAddressInfos = addresses.data();
  info.pVendorInfos = vendors.data();
  info.pVendorBinaryData = out.vendor_binary.data();
  result = vkGetDeviceFaultInfoEXT(m_device, &counts, &info);
  if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
    out.description.append(nx::format(
        "; vkGetDeviceFaultInfoEXT failed with {}", vk_result_string(result)));
    out.vendor_binary = {};
    return;
  }
  if (info.description[0] != '\0')
    out.description.append(nx::format("; {}", info.description));
  for (usize i = 0;
       i < nx::min<usize>(addresses.size(), counts.addressInfoCount); ++i)
    append_address(out, addresses[i]);
  for (usize i = 0; i < nx::min<usize>(vendors.size(), counts.vendorInfoCount);
       ++i)
    append_vendor(out, vendors[i]);
}

void VulkanGpuCrashReporter::append_objects_and_bindings(
    gfx::gpu_crash_report &out) {
  BindingEvent snapshot[BINDING_CAPACITY];
  ObjectNameEvent object_snapshot[OBJECT_CAPACITY];
  u64 bindings_written = 0;
  u64 objects_written = 0;
  {
    const nx::scoped_lock lock(m_binding_lock);
    bindings_written = m_binding_sequence;
    for (usize i = 0; i < BINDING_CAPACITY; ++i)
      snapshot[i] = m_bindings[i];
    objects_written = m_object_sequence;
    for (usize i = 0; i < OBJECT_CAPACITY; ++i)
      object_snapshot[i] = m_objects[i];
  }

  const u64 object_count = nx::min<u64>(objects_written, OBJECT_CAPACITY);
  const u64 first_object = objects_written - object_count + 1;
  out.objects.reserve(nx::cast<usize>(object_count));
  for (u64 sequence = first_object; sequence <= objects_written; ++sequence) {
    const ObjectNameEvent &object =
        object_snapshot[(sequence - 1) % OBJECT_CAPACITY];
    if (object.sequence != sequence)
      continue;
    out.objects.push_back({nx::string(object_type_name(object.object_type)),
                           nx::string(object.name), object.object});
  }

  const u64 binding_count = nx::min<u64>(bindings_written, BINDING_CAPACITY);
  const u64 first_binding = bindings_written - binding_count + 1;
  out.bindings.reserve(nx::cast<usize>(binding_count));
  for (u64 sequence = first_binding; sequence <= bindings_written; ++sequence) {
    const BindingEvent &event = snapshot[(sequence - 1) % BINDING_CAPACITY];
    if (event.sequence != sequence)
      continue;

    nx::string_view name(event.name);
    if (name.empty()) {
      for (u64 offset = 0; offset < object_count; ++offset) {
        const u64 object_sequence = objects_written - offset;
        const ObjectNameEvent &object =
            object_snapshot[(object_sequence - 1) % OBJECT_CAPACITY];
        if (object.sequence == object_sequence &&
            object.object == event.object &&
            object.object_type == event.object_type) {
          name = object.name;
          break;
        }
      }
    }
    out.bindings.push_back({nx::string(object_type_name(event.object_type)),
                            nx::string(name), event.object, event.address,
                            event.size, event.bound, event.internal});
  }
}

} // namespace nx::gfx::detail
