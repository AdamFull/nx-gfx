#include "core/rendering/rhi/shader_loading.h"

#include "core/foundation/serialization/asset_policy.h"
#include "core/foundation/strings/format.h"
#include "core/rendering/rhi/descs.h"

namespace nxe::rhi {
namespace {

constexpr u64 MAX_SHADER_BYTES = 64u * 1024u * 1024u;

[[nodiscard]] constexpr ShaderTarget runtime_target() noexcept {
#if defined(NX_RHI_METAL)
  return ShaderTarget::Metallib;
#else
  return ShaderTarget::Spirv;
#endif
}

[[nodiscard]] bool readable(const nx::vfs::FileInfo &info) noexcept {
  return info.exists && !info.is_directory && info.size != 0 &&
         info.size <= MAX_SHADER_BYTES;
}

} // namespace

ShaderSourceRead
acquire_shader_source(nx::vfs::AsyncIoContext &context,
                      const nx::string_view name,
                      const nx::cancellation_token &cancellation) {
  cancellation.throw_if_requested();
  ShaderSourceRead out;
  if (name.empty())
    return out;

  const nx::string cooked = nx::format("/shaders/{}.nxb", name);
  const nx::vfs::FileInfo cooked_info = context.stat(cooked.view());
  if (readable(cooked_info)) {
    auto bytes = context.read(cooked.view(), MAX_SHADER_BYTES);
    if (!bytes)
      return out;
    out.code = std::move(bytes.value());
    out.selected_file = cooked;
    out.mtime_ns = cooked_info.mtime_ns;
    out.cooked = true;
    out.success = true;
    return out;
  }

  if (!nx::asset_policy::can_fallback_to_authored_source(cooked_info.exists))
    return out;

  const nx::string binary =
      nx::format("/shaders/{}{}", name, SHADER_BINARY_EXT);
  const nx::vfs::FileInfo binary_info = context.stat(binary.view());
  if (!readable(binary_info))
    return out;
  auto code = context.read(binary.view(), MAX_SHADER_BYTES);
  if (!code)
    return out;

  const nx::string sidecar = nx::format("/shaders/{}.refl.json", name);
  const nx::vfs::FileInfo sidecar_info = context.stat(sidecar.view());
  if (sidecar_info.exists && !sidecar_info.is_directory &&
      sidecar_info.size <= MAX_SHADER_BYTES) {
    auto reflection = context.read(sidecar.view(), MAX_SHADER_BYTES);
    if (reflection)
      out.reflection = std::move(reflection.value());
  }

  out.code = std::move(code.value());
  out.selected_file = binary;
  out.mtime_ns = binary_info.mtime_ns;
  out.success = true;
  return out;
}

PreparedShaderLoad
prepare_shader_source(ShaderSourceRead &&source, const nx::string_view,
                      const nx::cancellation_token &cancellation) {
  cancellation.throw_if_requested();
  PreparedShaderLoad out;
  out.selected_file = std::move(source.selected_file);
  out.mtime_ns = source.mtime_ns;
  if (!source.success)
    return out;

  if (source.cooked) {
    if (!decode_shader({source.code.data(), source.code.size()}, out.blob) ||
        out.blob.target != runtime_target())
      return out;
  } else {
    out.blob.code.assign(source.code.begin(), source.code.end());
    if (!source.reflection.empty())
      out.blob.reflection_json = nx::string(nx::string_view(
          reinterpret_cast<const char *>(source.reflection.data()),
          source.reflection.size()));
    out.blob.target = runtime_target();
  }
  cancellation.throw_if_requested();
  out.success = !out.blob.code.empty();
  return out;
}

u64 prepared_shader_bytes(const PreparedShaderLoad &prepared) noexcept {
  return prepared.blob.code.size() + prepared.blob.reflection_json.size();
}

} // namespace nxe::rhi
