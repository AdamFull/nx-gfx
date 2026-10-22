#pragma once

#include "nx/gfx/crash.h"
#include "src/vulkan/vk_common.h"

#include "core/foundation/containers/hash_map.h"
#include "core/foundation/containers/string_map.h"
#include "core/foundation/platform/shared_library.h"
#include "core/foundation/threading/sync.h"

namespace nx::gfx::detail {

/// NVIDIA Nsight Aftermath, where the build found its SDK and its library
/// loads, as NVIDIA's own integration does it: it keeps every shader's
/// SPIR-V and the driver's debug information for the shaders in a crash, so
/// the dump decodes to the shader, the instruction and, for shaders built
/// with debug information, the source line; the pass checkpoints resolve to
/// their names. The crash report carries the decoded JSON, the dump and the
/// debug information files, which Nsight Graphics opens with the shaders.
/// Without the SDK every call does nothing. It watches devices made after
/// start(), so it starts before any is.
class VulkanAftermath {
public:
  VulkanAftermath() = default;
  ~VulkanAftermath();
  VulkanAftermath(const VulkanAftermath &) = delete;
  VulkanAftermath &operator=(const VulkanAftermath &) = delete;

  /// Loads the library and starts watching Vulkan devices, describing dumps
  /// as @p application's; false where it cannot, when it stays off.
  bool start(nx::string_view application) noexcept;
  [[nodiscard]] bool active() const noexcept { return m_active; }

  /// The flags of VkDeviceDiagnosticsConfigCreateInfoNV for a watched
  /// device.
  [[nodiscard]] static VkDeviceDiagnosticsConfigFlagsNV
  diagnostics_flags() noexcept;

  /// Keeps shader @p code (SPIR-V) as @p name, for dumps to decode against.
  void shader(const void *code, usize size, nx::string_view name) noexcept;
  /// Marks @p cmd's place as @p label, where checkpoints are enabled.
  void checkpoint(VkCommandBuffer cmd, nx::string_view label) noexcept;
  void enable_checkpoints(bool enabled) noexcept { m_checkpoints = enabled; }

  /// Waits, a bounded time, for the driver's dump of a lost device, and adds
  /// what it holds to @p report.
  void collect(gfx::gpu_crash_report &report) noexcept;

private:
  struct Api;
  struct Shader {
    nx::string name;
    nx::blob<u8> code;
  };
  struct DebugInfoKey {
    u64 id[2] = {};
    [[nodiscard]] friend bool operator==(const DebugInfoKey &,
                                         const DebugInfoKey &) = default;
  };
  struct DebugInfoHash {
    [[nodiscard]] usize operator()(const DebugInfoKey &key) const noexcept {
      return static_cast<usize>(key.id[0] ^
                                (key.id[1] * 0x9e3779b97f4a7c15ull));
    }
  };

  static void on_dump(const void *dump, u32 size, void *self);
  static void on_debug_info(const void *info, u32 size, void *self);
  static void on_description(void (*add)(u32 key, const char *value),
                             void *self);
  static void on_resolve_marker(const void *marker, u32 size, void *self,
                                void (*resolve)(const void *data, u32 size));
  void decode(gfx::gpu_crash_report &report, const nx::blob<u8> &dump);
  [[nodiscard]] nx::string label_of(u64 id) const;

  /// Labels a checkpoint names, kept for the life of the process; past
  /// this many, new ones share one.
  static constexpr usize MOST_LABELS = 4096;
  /// Shaders it keeps.
  static constexpr usize MOST_SHADERS = 16384;

  nx::os::shared_library m_library;
  Api *m_api = nullptr;
  bool m_active = false;
  bool m_checkpoints = false;
  nx::string m_application;
  mutable nx::mutex m_lock;
  nx::blob<u8> m_dump;
  nx::string_map<u64> m_label_ids;
  nx::vector<nx::string> m_labels;
  /// By Aftermath's binary hash, and that hash by the shader's debug name.
  nx::hash_map<u64, Shader> m_shaders;
  nx::string_map<u64> m_debug_names;
  nx::hash_map<DebugInfoKey, nx::blob<u8>, DebugInfoHash> m_debug_infos;
};

} // namespace nx::gfx::detail
