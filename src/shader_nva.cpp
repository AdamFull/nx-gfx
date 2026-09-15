#include "rendering/rhi/shader_nva.h"

#include "core/foundation/serialization/nva.h"

#include <cstring>

namespace nxe::rhi {
namespace {

constexpr u32 FORMAT = nx::nva::fourcc('S', 'H', 'D', 'R');
constexpr u32 CHUNK_META = nx::nva::fourcc('M', 'E', 'T', 'A');
constexpr u32 CHUNK_CODE = nx::nva::fourcc('C', 'O', 'D', 'E');
constexpr u32 CHUNK_REFL = nx::nva::fourcc('R', 'E', 'F', 'L');
constexpr u16 VERSION = 1;

} // namespace

nx::vector<u8> encode_shader(const std::span<const u8> code,
                             const nx::string_view reflection_json,
                             const ShaderTarget target) {
  nx::nva::Writer meta;
  meta.u32v(nx::cast<u32>(target));

  nx::nva::ChunkWriter w(FORMAT, VERSION);
  w.add(CHUNK_META, meta.span());
  w.add(CHUNK_CODE, code);
  w.add(CHUNK_REFL, {reinterpret_cast<const u8 *>(reflection_json.data()),
                     reflection_json.size()});
  return w.finish();
}

bool decode_shader(const std::span<const u8> bytes, ShaderBlob &out) {
  const std::optional<nx::nva::View> view = nx::nva::View::open(bytes, FORMAT);
  if (!view || view->version() != VERSION)
    return false;
  const std::span<const u8> meta = view->chunk(CHUNK_META);
  const std::span<const u8> code = view->chunk(CHUNK_CODE);
  const std::span<const u8> refl = view->chunk(CHUNK_REFL);
  if (meta.empty() || code.empty())
    return false;

  nx::nva::Reader r(meta);
  const u32 target = r.u32v();
  if (!r.ok() || target > nx::cast<u32>(ShaderTarget::Metallib))
    return false;
  ShaderBlob decoded;
  decoded.target = nx::cast<ShaderTarget>(target);
  decoded.code.resize(code.size());
  std::memcpy(decoded.code.data(), code.data(), code.size());
  decoded.reflection_json = nx::string(nx::string_view(
      reinterpret_cast<const char *>(refl.data()), refl.size()));
  out = std::move(decoded);
  return true;
}

} // namespace nxe::rhi
