#include "rendering/rhi/key_sort.h"

#include "rendering/rhi/key_sort_interop.h"

#include <bit>

namespace nxe::rhi {

ComputePipelineDesc key_sort_local_desc() noexcept {
  return {.name = "key sort local",
          .compute = {.entry_point = "cs_sort_local"},
          .group_size = {256, 1, 1}};
}

ComputePipelineDesc key_sort_pass_desc() noexcept {
  return {.name = "key sort pass",
          .compute = {.entry_point = "cs_sort_pass"},
          .group_size = {64, 1, 1}};
}

void record_key_sort(CommandContext &cmd, const KeySortPipelines &pipelines,
                     const BufferHandle buffer, const u64 address,
                     const u32 count) {
  if (count < 2 || !pipelines.valid())
    return;
  GpuKeySortPush push{.pairs = {address}, .count = count};
  const BufferBarrier between{.buffer = buffer,
                              .from = ResourceState::ShaderReadWrite,
                              .to = ResourceState::ShaderReadWrite};
  cmd.barrier(nullptr, 0, &between, 1);
  if (count <= NX_KEY_SORT_LOCAL) {
    cmd.bind_pipeline(pipelines.local);
    cmd.push_constants(&push, sizeof(push));
    cmd.dispatch(1);
    cmd.barrier(nullptr, 0, &between, 1);
  } else {
    // Stages of 2, 4, ... count pairs, each with steps of half the stage
    // down to 1: every step reads what the one before it wrote.
    cmd.bind_pipeline(pipelines.pass);
    const u32 stages = nx::cast<u32>(std::countr_zero(count));
    for (u32 stage = 1; stage <= stages; ++stage)
      for (u32 step = stage; step-- > 0;) {
        push.schedule = stage << 16 | step;
        cmd.push_constants(&push, sizeof(push));
        cmd.dispatch((count + 63) / 64);
        cmd.barrier(nullptr, 0, &between, 1);
      }
  }
}

} // namespace nxe::rhi
