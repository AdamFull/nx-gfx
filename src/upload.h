#pragma once

/**
 * @file upload.h
 * @brief Portable completion token for the async uploader (namespace nxe::rhi).
 *
 * The uploader records copies on the transfer queue and returns immediately;
 * an UploadTicket names the point on the uploader's timeline at which that work
 * is done. Poll it with Uploader::is_complete() or block on Uploader::wait().
 * A default-constructed (value 0) ticket is "nothing to wait for" and reports
 * complete.
 */

#include "foundation/core/foundation.h"

namespace nxe::rhi {

struct UploadTicket {
  u64 value = 0;
  [[nodiscard]] bool valid() const noexcept { return value != 0; }
};

} // namespace nxe::rhi
