#pragma once

#include "core/foundation/threading/thread_pool.h"
#include "nx/gfx/gfx.h"

namespace nx::gfx::test {

/// Rendering tests intentionally wait at fixture setup, but compilation still
/// goes through the production asynchronous RHI contract. A shared pool keeps
/// setup cheap without retaining device-owned objects between test cases.
[[nodiscard]] inline nx::thread_pool &pipeline_compile_workers() {
  static nx::thread_pool workers({.thread_count = 2,
                                  .name = "nx-test-pipeline",
                                  .collect_statistics = false});
  return workers;
}

/// A shader from its binary and reflection sidecar text, as the development
/// fallback loads one; a sidecar that does not parse leaves it without one.
[[nodiscard]] inline nx::gfx::shader_handle
create_shader(nx::gfx::device &device, nx::gfx::shader_desc desc,
              const nx::string_view reflection_json) {
  nx::gfx::shader_reflection reflection;
  if (!reflection_json.empty() &&
      nx::gfx::reflect_from_json(reflection_json, reflection))
    desc.reflection = &reflection;
  return nx::gfx::create_shader(device, desc);
}

[[nodiscard]] inline nx::gfx::pipeline_handle
compile_pipeline(nx::gfx::device &device,
                 const nx::gfx::graphics_pipeline_desc &description) {
  nx::future<nx::gfx::pipeline_job> pending = nx::gfx::compile_pipeline_async(
      pipeline_compile_workers(),
      nx::gfx::prepare_pipeline(device, description),
      nx::task_priority::Critical);
  if (!pending.valid())
    return {};
  try {
    nx::gfx::pipeline_job compiled = pending.get();
    return nx::gfx::publish_pipeline(device, std::move(compiled));
  } catch (...) {
    return {};
  }
}

[[nodiscard]] inline nx::gfx::pipeline_handle
compile_pipeline(nx::gfx::device &device,
                 const nx::gfx::compute_pipeline_desc &description) {
  nx::future<nx::gfx::pipeline_job> pending = nx::gfx::compile_pipeline_async(
      pipeline_compile_workers(),
      nx::gfx::prepare_pipeline(device, description),
      nx::task_priority::Critical);
  if (!pending.valid())
    return {};
  try {
    nx::gfx::pipeline_job compiled = pending.get();
    return nx::gfx::publish_pipeline(device, std::move(compiled));
  } catch (...) {
    return {};
  }
}

} // namespace nx::gfx::test
