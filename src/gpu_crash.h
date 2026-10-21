#pragma once

#include "nx/gfx/crash.h"

namespace nxe::rhi {

// The names before nx::gfx. Only the backend's own sources and projects made
// before nx::gfx (through app/engine.h) use them.
using GpuBackend = nx::gfx::gpu_backend;
using GpuCrashMode = nx::gfx::gpu_crash_mode;
using GpuDeviceStatus = nx::gfx::gpu_device_status;
using GpuFaultAddress = nx::gfx::gpu_fault_address;
using GpuAddressBinding = nx::gfx::gpu_address_binding;
using GpuObjectIdentity = nx::gfx::gpu_object_identity;
using GpuVendorFault = nx::gfx::gpu_vendor_fault;
using GpuEncoderFault = nx::gfx::gpu_encoder_fault;
using GpuCrashReport = nx::gfx::gpu_crash_report;

using nx::gfx::gpu_crash_json;
using nx::gfx::report_gpu_crash;

} // namespace nxe::rhi
