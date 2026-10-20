#pragma once

#include "nx/gfx/bindless.h"

namespace nxe::rhi {

// The names before nx::gfx, until every caller has moved to nx/gfx (G6).

using nx::gfx::BINDLESS_BINDING_GLOBALS;
using nx::gfx::BINDLESS_BINDING_RW_TEXTURES;
using nx::gfx::BINDLESS_BINDING_SAMPLERS;
using nx::gfx::BINDLESS_BINDING_TEXTURES;
using nx::gfx::BINDLESS_INVALID_INDEX;
using nx::gfx::BINDLESS_SET;
using nx::gfx::DEFAULT_SAMPLER;
using nx::gfx::MAX_BINDLESS_RW_TEXTURES;
using nx::gfx::MAX_BINDLESS_SAMPLERS;
using nx::gfx::MAX_BINDLESS_TEXTURES;
using nx::gfx::NULL_PAGE_BYTES;
using nx::gfx::PUSH_CONSTANT_SIZE;

} // namespace nxe::rhi
