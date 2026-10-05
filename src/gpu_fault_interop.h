#pragma once

// What the GPU-fault marker shader is handed, shared by C++ and Slang as
// nx_interop.h describes; gpu_fault.slang includes it.

#include "rendering/rhi/shaders/nx_interop.h"

/// Writes @p marker at @p target. The default target is an address no buffer
/// has, which is what a deliberate fault wants.
struct GpuFaultPush {
  NxPtr<uint> target = {0x10};
  uint marker = 0x47505521u;
  uint padding = 0u;
};
NX_SHARED_SIZE(GpuFaultPush, 16);
