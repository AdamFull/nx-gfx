#pragma once

#include "core/foundation/containers/blob.h"
#include "core/foundation/core/foundation.h"
#include "core/foundation/strings/utf8_string.h"

namespace nxe::rhi {

enum class GpuBackend : u8 { Vulkan, Metal };

enum class GpuCrashMode : u8 {
  Disabled,
  Basic,
  Detailed,
};

/// Terminal health reported by an RHI device. A lost device cannot be made
/// ready again; recovery requires constructing a new device generation.
enum class GpuDeviceStatus : u8 { Ready, Lost };

struct GpuFaultAddress {
  nx::string kind;
  u64 address = 0;
  u64 precision = 0;
};

struct GpuAddressBinding {
  nx::string object_type;
  nx::string object_name;
  u64 object = 0;
  u64 address = 0;
  u64 size = 0;
  bool bound = false;
  bool internal = false;
};

struct GpuObjectIdentity {
  nx::string object_type;
  nx::string object_name;
  u64 object = 0;
};

struct GpuVendorFault {
  nx::string description;
  u64 code = 0;
  u64 data = 0;
};

struct GpuEncoderFault {
  nx::string label;
  nx::string state;
  nx::vector<nx::string> debug_signposts;
};

/// Backend-neutral crash payload. Backends translate native diagnostics into
/// this bounded representation; persistence remains owned by nx::crash.
struct GpuCrashReport {
  GpuBackend backend = GpuBackend::Vulkan;
  nx::string device;
  nx::string operation;
  nx::string result;
  nx::string description;
  u64 frame = 0;
  nx::vector<GpuFaultAddress> addresses;
  nx::vector<GpuObjectIdentity> objects;
  nx::vector<GpuAddressBinding> bindings;
  nx::vector<GpuVendorFault> vendor_faults;
  nx::vector<GpuEncoderFault> encoders;
  nx::blob<u8> vendor_binary;
};

[[nodiscard]] nx::string gpu_crash_json(const GpuCrashReport &report);
[[nodiscard]] bool report_gpu_crash(const GpuCrashReport &report) noexcept;

} // namespace nxe::rhi
