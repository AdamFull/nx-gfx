#pragma once

#include "core/foundation/core/foundation.h"
#include "rendering/rhi/descs.h"
#include "rendering/rhi/device.h"
#include "rendering/rhi/handles.h"

namespace nxe::rhi {

/// The kernels' shader, built from rendering/shaders/key_sort.slang.
inline constexpr nx::string_view KEY_SORT_SHADER = "key_sort";

struct KeySortPipelines {
  PipelineHandle local;
  PipelineHandle pass;

  [[nodiscard]] bool valid() const noexcept {
    return local.valid() && pass.valid();
  }
};

/// The two kernels' descriptions, their shader left for the caller to set.
[[nodiscard]] ComputePipelineDesc key_sort_local_desc() noexcept;
[[nodiscard]] ComputePipelineDesc key_sort_pass_desc() noexcept;

/// Records sorting the first @p count {key, value} pairs at @p address in
/// @p buffer ascending by key and then by value. @p count is a power of two;
/// fewer real pairs are padded past them with all-ones keys. One dispatch
/// sorts as many as NX_KEY_SORT_LOCAL in group memory; more take a bitonic
/// network, a dispatch a pass, each behind a barrier on @p buffer. The pairs
/// are read and written by shaders before and after.
void record_key_sort(CommandContext &cmd, const KeySortPipelines &pipelines,
                     BufferHandle buffer, u64 address, u32 count);

} // namespace nxe::rhi
