#pragma once

// What the key sort's kernels are handed, shared by C++ and Slang as
// nx_interop.h describes; key_sort.slang includes it.

#include "rendering/rhi/shaders/nx_interop.h"

/// The most pairs the local kernel sorts whole in group memory.
NX_CONST(uint, NX_KEY_SORT_LOCAL, 2048u);

/// Sorts `count` {key, value} pairs at `pairs`, a power of two, ascending by
/// key and then by value.
struct GpuKeySortPush {
  NxPtr<uint2> pairs = {};
  uint count = 0u;
  /// A global pass's stage and step, each as its log2: stage << 16 | step.
  uint schedule = 0u;
};
NX_SHARED_SIZE(GpuKeySortPush, 16);
