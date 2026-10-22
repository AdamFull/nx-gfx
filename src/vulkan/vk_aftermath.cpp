#include "src/vulkan/vk_aftermath.h"

#include "core/foundation/diagnostics/log.h"
#include "core/foundation/serialization/json.h"
#include "core/foundation/strings/format.h"
#include "core/foundation/threading/thread.h"

#include <chrono>

#if defined(NX_GFX_AFTERMATH)
// The SDK declares its Vulkan functions only after vulkan.h itself.
#if !defined(VK_NO_PROTOTYPES)
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include "GFSDK_Aftermath.h"
#include "GFSDK_Aftermath_GpuCrashDump.h"
#include "GFSDK_Aftermath_GpuCrashDumpDecoding.h"
#endif

namespace nx::gfx::detail {

#if defined(NX_GFX_AFTERMATH)

namespace {

/// How long a lost device's dump may take to begin, and then to arrive.
constexpr auto DUMP_START = std::chrono::seconds(3);
constexpr auto DUMP_FINISH = std::chrono::seconds(15);
constexpr auto DUMP_POLL = std::chrono::milliseconds(50);

[[nodiscard]] bool succeeded(const GFSDK_Aftermath_Result result) noexcept {
  return GFSDK_Aftermath_SUCCEED(result);
}

[[nodiscard]] GFSDK_Aftermath_SpirvCode spirv_of(const void *const code,
                                                 const usize size) noexcept {
  GFSDK_Aftermath_SpirvCode out{};
  out.pData = code;
  out.size = nx::cast<u32>(size);
  return out;
}

/// The faulted warps' places in the decoded JSON, where it maps them.
void add_fault_places(gfx::gpu_crash_report &report,
                      const nx::string_view json) {
  const auto parsed = nx::json::parse(json);
  if (!parsed)
    return;
  const auto visit = [&](const auto &self,
                         const nx::json::value &node) -> void {
    if (node.is_array()) {
      for (const nx::json::value &item : node.as_array())
        self(self, item);
      return;
    }
    if (!node.is_object())
      return;
    for (const auto &[key, child] : node.as_object()) {
      if (key.view() == "Faulted Warps" && child.is_array()) {
        for (const nx::json::value &warp : child.as_array()) {
          if (!warp.is_object())
            continue;
          nx::string place;
          for (const auto &[field, text] : warp.as_object())
            if (text.is_string() && (field.view() == "Shader mapping" ||
                                     field.view() == "Shader GPU PC Address" ||
                                     field.view() == "Fault Name"))
              place.append(nx::format("{}{}: {}", place.empty() ? "" : "; ",
                                      field.view(), text.as_string().view()));
          if (!place.empty())
            report.vendor_faults.push_back(
                {.description =
                     nx::format("Aftermath faulted warp: {}", place)});
        }
        continue;
      }
      self(self, child);
    }
  };
  visit(visit, *parsed);
}

} // namespace

struct VulkanAftermath::Api {
  decltype(&GFSDK_Aftermath_EnableGpuCrashDumps) enable = nullptr;
  decltype(&GFSDK_Aftermath_DisableGpuCrashDumps) disable = nullptr;
  decltype(&GFSDK_Aftermath_GetCrashDumpStatus) status = nullptr;
  decltype(&GFSDK_Aftermath_GetShaderHashSpirv) shader_hash = nullptr;
  decltype(&GFSDK_Aftermath_GetShaderDebugNameSpirv) debug_name = nullptr;
  decltype(&GFSDK_Aftermath_GetShaderDebugInfoIdentifier) debug_info_id =
      nullptr;
  decltype(&GFSDK_Aftermath_GpuCrashDump_CreateDecoder) create_decoder =
      nullptr;
  decltype(&GFSDK_Aftermath_GpuCrashDump_DestroyDecoder) destroy_decoder =
      nullptr;
  decltype(&GFSDK_Aftermath_GpuCrashDump_GetPageFaultInfo) page_fault = nullptr;
  decltype(&GFSDK_Aftermath_GpuCrashDump_GetActiveShadersInfoCount)
      shader_count = nullptr;
  decltype(&GFSDK_Aftermath_GpuCrashDump_GetActiveShadersInfo) shaders =
      nullptr;
  decltype(&GFSDK_Aftermath_GetShaderHashForShaderInfo) shader_info_hash =
      nullptr;
  decltype(&GFSDK_Aftermath_GpuCrashDump_GetEventMarkersInfoCount)
      marker_count = nullptr;
  decltype(&GFSDK_Aftermath_GpuCrashDump_GetEventMarkersInfo) markers = nullptr;
  decltype(&GFSDK_Aftermath_GpuCrashDump_GenerateJSON) generate_json = nullptr;
  decltype(&GFSDK_Aftermath_GpuCrashDump_GetJSON) json = nullptr;
};

VulkanAftermath::~VulkanAftermath() {
  if (m_active)
    (void)m_api->disable();
  delete m_api;
}

bool VulkanAftermath::start(const nx::string_view application) noexcept {
  if (m_active)
    return true;
  // Beside the executable first, as a shipped one is; else the SDK's.
  if (!m_library.is_open() && !m_library.open(nx::os::shared_library::decorate(
                                  "GFSDK_Aftermath_Lib.x64"))) {
#if defined(NX_GFX_AFTERMATH_LIBRARY)
    if (const auto opened = m_library.open(NX_GFX_AFTERMATH_LIBRARY); !opened) {
      nx::logw(nx::log::rhi, "Nsight Aftermath cannot be loaded: {}",
               opened.error());
      return false;
    }
#else
    return false;
#endif
  }
  Api api;
  const auto resolve = [this](auto &out, const char *const name) {
    out = reinterpret_cast<std::remove_reference_t<decltype(out)>>(
        m_library.symbol(name));
    return out != nullptr;
  };
  if (!resolve(api.enable, "GFSDK_Aftermath_EnableGpuCrashDumps") ||
      !resolve(api.disable, "GFSDK_Aftermath_DisableGpuCrashDumps") ||
      !resolve(api.status, "GFSDK_Aftermath_GetCrashDumpStatus") ||
      !resolve(api.shader_hash, "GFSDK_Aftermath_GetShaderHashSpirv") ||
      !resolve(api.debug_name, "GFSDK_Aftermath_GetShaderDebugNameSpirv") ||
      !resolve(api.debug_info_id,
               "GFSDK_Aftermath_GetShaderDebugInfoIdentifier") ||
      !resolve(api.create_decoder,
               "GFSDK_Aftermath_GpuCrashDump_CreateDecoder") ||
      !resolve(api.destroy_decoder,
               "GFSDK_Aftermath_GpuCrashDump_DestroyDecoder") ||
      !resolve(api.page_fault,
               "GFSDK_Aftermath_GpuCrashDump_GetPageFaultInfo") ||
      !resolve(api.shader_count,
               "GFSDK_Aftermath_GpuCrashDump_GetActiveShadersInfoCount") ||
      !resolve(api.shaders,
               "GFSDK_Aftermath_GpuCrashDump_GetActiveShadersInfo") ||
      !resolve(api.shader_info_hash,
               "GFSDK_Aftermath_GetShaderHashForShaderInfo") ||
      !resolve(api.marker_count,
               "GFSDK_Aftermath_GpuCrashDump_GetEventMarkersInfoCount") ||
      !resolve(api.markers,
               "GFSDK_Aftermath_GpuCrashDump_GetEventMarkersInfo") ||
      !resolve(api.generate_json,
               "GFSDK_Aftermath_GpuCrashDump_GenerateJSON") ||
      !resolve(api.json, "GFSDK_Aftermath_GpuCrashDump_GetJSON")) {
    nx::logw(nx::log::rhi, "Nsight Aftermath lacks an entry point it needs");
    return false;
  }
  m_application = nx::string(application);
  // Debug information only for the shaders a crash involves, when it comes.
  const GFSDK_Aftermath_Result enabled = api.enable(
      GFSDK_Aftermath_Version_API,
      GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_Vulkan,
      GFSDK_Aftermath_GpuCrashDumpFeatureFlags_DeferDebugInfoCallbacks,
      &on_dump, &on_debug_info, &on_description, &on_resolve_marker, this);
  if (!succeeded(enabled)) {
    nx::logw(nx::log::rhi, "Nsight Aftermath does not start: 0x{:08x}",
             static_cast<u32>(enabled));
    return false;
  }
  m_api = new Api(api);
  m_active = true;
  nx::logi(nx::log::rhi, "GPU crash dumps through Nsight Aftermath");
  return true;
}

VkDeviceDiagnosticsConfigFlagsNV VulkanAftermath::diagnostics_flags() noexcept {
  return VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV |
         VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV |
         VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV |
         VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_ERROR_REPORTING_BIT_NV;
}

void VulkanAftermath::shader(const void *const code, const usize size,
                             const nx::string_view name) noexcept {
  if (!m_active)
    return;
  const GFSDK_Aftermath_SpirvCode spirv = spirv_of(code, size);
  GFSDK_Aftermath_ShaderBinaryHash hash{};
  if (!succeeded(
          m_api->shader_hash(GFSDK_Aftermath_Version_API, &spirv, &hash)))
    return;
  // The binary is its own source of debug information: what runs is what
  // was built, with the information in it.
  GFSDK_Aftermath_ShaderDebugName debug_name{};
  const bool named = succeeded(m_api->debug_name(GFSDK_Aftermath_Version_API,
                                                 &spirv, &spirv, &debug_name));
  const auto *const bytes = static_cast<const u8 *>(code);
  try {
    const nx::scoped_lock<nx::mutex> held(m_lock);
    if (m_shaders.size() >= MOST_SHADERS && !m_shaders.contains(hash.hash))
      return;
    m_shaders.insert_or_assign(
        hash.hash, Shader{.name = nx::string(name),
                          .code = nx::blob<u8>(bytes, bytes + size)});
    if (named)
      m_debug_names.insert_or_assign(nx::string(debug_name.name), hash.hash);
  } catch (...) {
  }
}

void VulkanAftermath::checkpoint(const VkCommandBuffer cmd,
                                 const nx::string_view label) noexcept {
  if (!m_checkpoints || vkCmdSetCheckpointNV == nullptr)
    return;
  u64 id = 0;
  try {
    const nx::scoped_lock<nx::mutex> held(m_lock);
    if (const auto found = m_label_ids.find(label);
        found != m_label_ids.end()) {
      id = found->second;
    } else if (m_labels.size() < MOST_LABELS) {
      m_labels.emplace_back(label);
      id = m_labels.size();
      m_label_ids.emplace(nx::string(label), id);
    }
  } catch (...) {
  }
  // The marker is the label's number, not a pointer: the dump keeps only it,
  // and on_resolve_marker names it.
  vkCmdSetCheckpointNV(
      cmd, reinterpret_cast<const void *>(static_cast<uintptr_t>(id)));
}

nx::string VulkanAftermath::label_of(const u64 id) const {
  const nx::scoped_lock<nx::mutex> held(m_lock);
  if (id == 0 || id > m_labels.size())
    return nx::format("checkpoint {}", id);
  return m_labels[id - 1];
}

void VulkanAftermath::on_dump(const void *const dump, const u32 size,
                              void *const self) {
  VulkanAftermath &owner = *static_cast<VulkanAftermath *>(self);
  const auto *const bytes = static_cast<const u8 *>(dump);
  const nx::scoped_lock<nx::mutex> held(owner.m_lock);
  owner.m_dump = nx::blob<u8>(bytes, bytes + size);
}

void VulkanAftermath::on_debug_info(const void *const info, const u32 size,
                                    void *const self) {
  VulkanAftermath &owner = *static_cast<VulkanAftermath *>(self);
  GFSDK_Aftermath_ShaderDebugInfoIdentifier identifier{};
  if (!succeeded(owner.m_api->debug_info_id(GFSDK_Aftermath_Version_API, info,
                                            size, &identifier)))
    return;
  const auto *const bytes = static_cast<const u8 *>(info);
  const nx::scoped_lock<nx::mutex> held(owner.m_lock);
  owner.m_debug_infos.insert_or_assign(
      DebugInfoKey{{identifier.id[0], identifier.id[1]}},
      nx::blob<u8>(bytes, bytes + size));
}

void VulkanAftermath::on_description(void (*const add)(u32, const char *),
                                     void *const self) {
  const VulkanAftermath &owner = *static_cast<const VulkanAftermath *>(self);
  add(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName,
      owner.m_application.empty() ? "nx-gfx" : owner.m_application.c_str());
}

void VulkanAftermath::on_resolve_marker(const void *const marker, const u32,
                                        void *const self,
                                        void (*const resolve)(const void *,
                                                              u32)) {
  const VulkanAftermath &owner = *static_cast<const VulkanAftermath *>(self);
  const nx::string label = owner.label_of(reinterpret_cast<uintptr_t>(marker));
  resolve(label.c_str(), nx::cast<u32>(label.size() + 1));
}

void VulkanAftermath::collect(gfx::gpu_crash_report &report) noexcept {
  if (!m_active)
    return;
  try {
    using clock = std::chrono::steady_clock;
    const clock::time_point began = clock::now();
    GFSDK_Aftermath_CrashDump_Status status =
        GFSDK_Aftermath_CrashDump_Status_Unknown;
    while (succeeded(m_api->status(&status)) &&
           status != GFSDK_Aftermath_CrashDump_Status_CollectingDataFailed &&
           status != GFSDK_Aftermath_CrashDump_Status_Finished) {
      const auto waited = clock::now() - began;
      if (waited > DUMP_FINISH ||
          (status == GFSDK_Aftermath_CrashDump_Status_NotStarted &&
           waited > DUMP_START))
        break;
      nx::this_thread::sleep_for(DUMP_POLL);
    }
    nx::blob<u8> dump;
    {
      const nx::scoped_lock<nx::mutex> held(m_lock);
      dump = std::move(m_dump);
    }
    if (dump.empty()) {
      report.description.append(
          nx::format("; Nsight Aftermath made no dump (status {})",
                     static_cast<u32>(status)));
      return;
    }
    decode(report, dump);
    report.attachments.push_back(
        {.name = "gpu-aftermath.nv-gpudmp", .bytes = std::move(dump)});
    const nx::scoped_lock<nx::mutex> held(m_lock);
    for (const auto &[key, info] : m_debug_infos)
      report.attachments.push_back(
          {.name = nx::format("gpu-aftermath-{:016x}-{:016x}.nvdbg", key.id[0],
                              key.id[1]),
           .bytes = info});
  } catch (...) {
    nx::loge(nx::log::rhi, "the Nsight Aftermath dump could not be read");
  }
}

void VulkanAftermath::decode(gfx::gpu_crash_report &report,
                             const nx::blob<u8> &dump) {
  GFSDK_Aftermath_GpuCrashDump_Decoder decoder{};
  if (!succeeded(m_api->create_decoder(GFSDK_Aftermath_Version_API, dump.data(),
                                       nx::cast<u32>(dump.size()), &decoder))) {
    report.description.append("; the Nsight Aftermath dump does not decode");
    return;
  }

  GFSDK_Aftermath_GpuCrashDump_PageFaultInfo fault{};
  if (succeeded(m_api->page_fault(decoder, &fault)))
    report.vendor_faults.push_back(
        {.description = nx::format(
             "Aftermath page fault: type {}, access {}, engine {}, client {}",
             static_cast<u32>(fault.faultType),
             static_cast<u32>(fault.accessType), static_cast<u32>(fault.engine),
             static_cast<u32>(fault.client)),
         .code = static_cast<u64>(fault.faultType),
         .data = fault.faultingGpuVA});

  u32 count = 0;
  if (succeeded(m_api->shader_count(decoder, &count)) && count != 0) {
    nx::vector<GFSDK_Aftermath_GpuCrashDump_ShaderInfo> shaders(count);
    if (succeeded(m_api->shaders(decoder, count, shaders.data())))
      for (const GFSDK_Aftermath_GpuCrashDump_ShaderInfo &shader : shaders) {
        GFSDK_Aftermath_ShaderBinaryHash hash{};
        nx::string name = "an unknown shader";
        if (succeeded(m_api->shader_info_hash(decoder, &shader, &hash))) {
          const nx::scoped_lock<nx::mutex> held(m_lock);
          if (const auto found = m_shaders.find(hash.hash);
              found != m_shaders.end())
            name = found->second.name;
        }
        report.vendor_faults.push_back(
            {.description =
                 nx::format("Aftermath active shader: {}{} (type {})", name,
                            shader.isInternal ? " (driver internal)" : "",
                            static_cast<u32>(shader.shaderType)),
             .code = hash.hash,
             .data = shader.shaderHash});
      }
  }

  count = 0;
  if (succeeded(m_api->marker_count(decoder, &count)) && count != 0) {
    nx::vector<GFSDK_Aftermath_GpuCrashDump_EventMarkerInfo> markers(count);
    if (succeeded(m_api->markers(decoder, count, markers.data())))
      for (const GFSDK_Aftermath_GpuCrashDump_EventMarkerInfo &marker :
           markers) {
        const nx::string label =
            marker.markerDataOwnership ==
                    GFSDK_Aftermath_EventMarkerDataOwnership_User
                ? label_of(reinterpret_cast<uintptr_t>(marker.markerData))
                : nx::string(static_cast<const char *>(marker.markerData));
        report.vendor_faults.push_back(
            {.description =
                 nx::format("Aftermath last checkpoint: {} (context status {})",
                            label, static_cast<u32>(marker.contextStatus)),
             .code = marker.contextId,
             .data = static_cast<u64>(marker.contextStatus)});
      }
  }

  // The decoder asks for what it maps the dump against: the driver's debug
  // information by identifier, and the SPIR-V by hash and by debug name.
  const auto debug_info =
      [](const GFSDK_Aftermath_ShaderDebugInfoIdentifier *const identifier,
         const PFN_GFSDK_Aftermath_SetData set, void *const self) {
        VulkanAftermath &owner = *static_cast<VulkanAftermath *>(self);
        const nx::scoped_lock<nx::mutex> held(owner.m_lock);
        const auto found = owner.m_debug_infos.find(
            DebugInfoKey{{identifier->id[0], identifier->id[1]}});
        if (found != owner.m_debug_infos.end())
          set(found->second.data(), nx::cast<u32>(found->second.size()));
      };
  const auto binary = [](const GFSDK_Aftermath_ShaderBinaryHash *const hash,
                         const PFN_GFSDK_Aftermath_SetData set,
                         void *const self) {
    VulkanAftermath &owner = *static_cast<VulkanAftermath *>(self);
    const nx::scoped_lock<nx::mutex> held(owner.m_lock);
    const auto found = owner.m_shaders.find(hash->hash);
    if (found != owner.m_shaders.end())
      set(found->second.code.data(), nx::cast<u32>(found->second.code.size()));
  };
  const auto source = [](const GFSDK_Aftermath_ShaderDebugName *const name,
                         const PFN_GFSDK_Aftermath_SetData set,
                         void *const self) {
    VulkanAftermath &owner = *static_cast<VulkanAftermath *>(self);
    const nx::scoped_lock<nx::mutex> held(owner.m_lock);
    const auto named = owner.m_debug_names.find(nx::string_view(name->name));
    if (named == owner.m_debug_names.end())
      return;
    const auto found = owner.m_shaders.find(named->second);
    if (found != owner.m_shaders.end())
      set(found->second.code.data(), nx::cast<u32>(found->second.code.size()));
  };
  u32 json_size = 0;
  if (succeeded(m_api->generate_json(
          decoder, GFSDK_Aftermath_GpuCrashDumpDecoderFlags_ALL_INFO,
          GFSDK_Aftermath_GpuCrashDumpFormatterFlags_UTF8_OUTPUT, debug_info,
          binary, source, this, &json_size)) &&
      json_size != 0) {
    nx::blob<u8> json(json_size);
    if (succeeded(m_api->json(decoder, json_size,
                              reinterpret_cast<char *>(json.data())))) {
      // The terminating zero is not part of the file.
      usize length = json.size();
      while (length != 0 && json[length - 1] == 0)
        --length;
      add_fault_places(
          report,
          nx::string_view(reinterpret_cast<const char *>(json.data()), length));
      report.attachments.push_back(
          {.name = "gpu-aftermath.json",
           .bytes = nx::blob<u8>(json.data(), json.data() + length)});
    }
  }
  (void)m_api->destroy_decoder(decoder);
}

#else

struct VulkanAftermath::Api {};

VulkanAftermath::~VulkanAftermath() = default;

bool VulkanAftermath::start(nx::string_view) noexcept { return false; }

VkDeviceDiagnosticsConfigFlagsNV VulkanAftermath::diagnostics_flags() noexcept {
  return 0;
}

void VulkanAftermath::shader(const void *, usize, nx::string_view) noexcept {}

void VulkanAftermath::checkpoint(VkCommandBuffer, nx::string_view) noexcept {}

void VulkanAftermath::collect(gfx::gpu_crash_report &) noexcept {}

#endif

} // namespace nx::gfx::detail
