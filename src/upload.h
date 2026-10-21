#pragma once

#include "nx/gfx/upload.h"

namespace nxe::rhi {

// The names before nx::gfx. Only the backend's own sources and projects made
// before nx::gfx (through app/engine.h) use them.
using UploadTicket = nx::gfx::upload_ticket;
using ReadbackResult = nx::gfx::readback_result;
using UploaderStats = nx::gfx::uploader_stats;

} // namespace nxe::rhi
