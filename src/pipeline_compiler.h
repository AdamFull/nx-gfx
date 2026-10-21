#pragma once

#include "nx/gfx/pipeline_job.h"

namespace nxe::rhi {

// The names before nx::gfx. Only the backend's own sources and projects made
// before nx::gfx (through app/engine.h) use them.
using PipelineJob = nx::gfx::pipeline_job;

using nx::gfx::compile_pipeline_async;

namespace detail {
using PipelineJobAccess = nx::gfx::detail::pipeline_job_access;
}

} // namespace nxe::rhi
