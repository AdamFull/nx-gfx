#pragma once

/**
 * @file upload.h
 * @brief Portable completion token for the async uploader (namespace nxe::rhi).
 *
 * The uploader records copies on the transfer queue and returns immediately;
 * an UploadTicket names the point on the uploader's timeline at which that work
 * is done. Poll it with Uploader::is_complete() or block on Uploader::wait().
 */

#include "foundation/core/foundation.h"
#include "foundation/strings/utf8_string_view.h"

namespace nxe::rhi {

struct UploadTicket {
  u64 value = 0;
  bool failed = false;

  [[nodiscard]] bool valid() const noexcept { return value != 0; }
  [[nodiscard]] bool ok() const noexcept { return !failed; }

  /// The ticket a failed upload returns.
  [[nodiscard]] static UploadTicket failure() noexcept { return {0, true}; }
};

/// Lifecycle of an asynchronous load (see Uploader::load_texture).
enum class LoadStatus : u8 {
  None,      ///< No such request (or it was released).
  Pending,   ///< Queued; the worker has not started it.
  Decoding,  ///< The worker is reading / decoding it.
  Uploading, ///< Decoded and created; its GPU upload is in flight.
  Ready,     ///< Uploaded and sampleable.
  Failed,    ///< Missing file, malformed data, or an unsupported format.
};

[[nodiscard]] nx::string_view to_string(LoadStatus status) noexcept;

/// Opaque id for an in-flight load, decoupled from TextureHandle until Ready.
/// The generation guards against a slot being recycled under an old request.
struct LoadRequest {
  u32 id = 0;
  u32 generation = 0;
  [[nodiscard]] bool valid() const noexcept { return id != 0; }
};

} // namespace nxe::rhi
