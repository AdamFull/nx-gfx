#pragma once

#include "nx/gfx/upload.h"

namespace nxe::rhi {

// The names before nx::gfx, until every caller has moved to nx/gfx (G6).
using UploadTicket = nx::gfx::upload_ticket;

} // namespace nxe::rhi
