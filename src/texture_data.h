#pragma once

#include "nx/gfx/upload.h"

namespace nxe::rhi {

// The names before nx::gfx. Only the backend's own sources and projects made
// before nx::gfx (through app/engine.h) use them.
using ImageSubresource = nx::gfx::image_subresource;
using TextureData = nx::gfx::texture_data;
using ImageUploadDesc = nx::gfx::image_upload_desc;

using nx::gfx::MAX_IMAGE_ARRAY_LAYERS;
using nx::gfx::validate_image_upload;

namespace detail {
using nx::gfx::detail::checked_mul;
}

} // namespace nxe::rhi
