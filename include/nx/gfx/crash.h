#pragma once

#include "core/foundation/containers/blob.h"
#include "core/foundation/core/foundation.h"
#include "core/foundation/strings/utf8_string.h"

namespace nx::gfx {

enum class gpu_backend : u8 { Vulkan, Metal };

enum class gpu_crash_mode : u8 {
  Disabled,
  Basic,
  Detailed,
};

/// Terminal health reported by an RHI device. A lost device cannot be made
/// ready again; recovery requires constructing a new device generation.
enum class gpu_device_status : u8 { Ready, Lost };

struct gpu_fault_address {
  nx::string kind;
  u64 address = 0;
  u64 precision = 0;
};

struct gpu_address_binding {
  nx::string object_type;
  nx::string object_name;
  u64 object = 0;
  u64 address = 0;
  u64 size = 0;
  bool bound = false;
  bool internal = false;
};

struct gpu_object_identity {
  nx::string object_type;
  nx::string object_name;
  u64 object = 0;
};

struct gpu_vendor_fault {
  nx::string description;
  u64 code = 0;
  u64 data = 0;
};

struct gpu_encoder_fault {
  nx::string label;
  nx::string state;
  nx::vector<nx::string> debug_signposts;
};

/// Backend-neutral crash payload. Backends translate native diagnostics into
/// this bounded representation; persistence remains owned by nx::crash.
struct gpu_crash_report {
  gpu_backend backend = gpu_backend::Vulkan;
  nx::string device;
  nx::string operation;
  nx::string result;
  nx::string description;
  u64 frame = 0;
  nx::vector<gpu_fault_address> addresses;
  nx::vector<gpu_object_identity> objects;
  nx::vector<gpu_address_binding> bindings;
  nx::vector<gpu_vendor_fault> vendor_faults;
  nx::vector<gpu_encoder_fault> encoders;
  nx::blob<u8> vendor_binary;
};

[[nodiscard]] nx::string gpu_crash_json(const gpu_crash_report &report);
[[nodiscard]] bool report_gpu_crash(const gpu_crash_report &report) noexcept;

} // namespace nx::gfx
