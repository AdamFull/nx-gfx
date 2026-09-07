#pragma once

#include "core/foundation/vfs/asset_pipeline.h"
#include "core/rendering/rhi/shader_nva.h"

namespace nxe::rhi {

struct ShaderSourceRead {
  nx::blob<u8> code;
  nx::blob<u8> reflection;
  nx::string selected_file;
  u64 mtime_ns = 0;
  bool cooked = false;
  bool success = false;
};

struct PreparedShaderLoad {
  ShaderBlob blob;
  nx::string selected_file;
  u64 mtime_ns = 0;
  bool success = false;
};

using ShaderLoadPipeline =
    nx::vfs::StagedAssetPipeline<ShaderSourceRead, PreparedShaderLoad>;

/// Acquires /shaders/<name>.nxb first. Development builds may fall back to the
/// backend-native binary plus its reflection sidecar.
[[nodiscard]] ShaderSourceRead
acquire_shader_source(nx::vfs::AsyncIoContext &context, nx::string_view name,
                      const nx::cancellation_token &cancellation);
[[nodiscard]] PreparedShaderLoad
prepare_shader_source(ShaderSourceRead &&source, nx::string_view name,
                      const nx::cancellation_token &cancellation);
[[nodiscard]] u64
prepared_shader_bytes(const PreparedShaderLoad &prepared) noexcept;

} // namespace nxe::rhi
