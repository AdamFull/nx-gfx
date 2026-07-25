#pragma once

/**
 * @file spirv_reflect.h
 * @brief The little bit of SPIR-V reflection the RHI needs (namespace
 * nxe::rhi).
 *
 * Not a general reflection library - just enough to cross-check what a caller
 * claims about a shader against what the shader actually says.
 *
 * This is portable C++ with no Vulkan types on purpose. Only the Vulkan backend
 * consumes SPIR-V, but keeping the parser out of it means the parser can be
 * unit-tested on any platform, including ones that build the Metal backend
 * instead.
 */

#include "foundation/core/foundation.h"
#include "foundation/strings/utf8_string_view.h"

#include <span>

namespace nxe::rhi {

/// One compute entry point and the workgroup size it declares.
struct SpirvComputeEntry {
  /// Points into @p code, so it lives exactly as long as the blob does.
  nx::string_view name;
  u32 local_size[3] = {};
};

/// Lists the compute entry points in a SPIR-V blob.
///
/// Enumeration rather than lookup because the caller that has the blob
/// (create_shader) is not the one that knows the entry point name
/// (create_compute_pipeline), and retaining the blob just to answer one
/// question later would cost more than remembering the answers.
///
/// Entry points whose size is not stated as literals - a specialisation
/// constant, say - are reported with a zero size, meaning "not knowable here".
///
/// @param code The SPIR-V blob, as handed to create_shader.
/// @param out  Receives the entries, up to its size.
/// @return How many entries were written; 0 if @p code is not little-endian
///         SPIR-V or holds no compute entry point.
[[nodiscard]] usize
spirv_compute_entries(std::span<const u8> code,
                      std::span<SpirvComputeEntry> out) noexcept;

/// Convenience wrapper for a single named entry point.
///
/// Returns false when the entry point is absent or its size is not knowable. A
/// false result means "could not tell", never "mismatch" - callers treat it as
/// no information rather than as an error.
[[nodiscard]] bool spirv_local_size(std::span<const u8> code,
                                    nx::string_view entry_point,
                                    u32 out_size[3]) noexcept;

} // namespace nxe::rhi
