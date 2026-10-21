#pragma once

// The word a sampled texture handle is made of (texture.hpp,
// nxg_texture.slang), shared by C++ and Slang; Slang reads it through the
// nxg_texture_layout module, which nxg_texture re-exports.

#include "nx/gfx/shaders/shared.h"

// Its bindless slot in the upper half, its sampler and sampling flags in the
// lower.
NX_CONST(uint, NX_TEXTURE_NONE, 0xFFFFu);
NX_CONST(uint, NX_TEXTURE_LINEAR, 0x4000u);
NX_CONST(uint, NX_TEXTURE_PREMULTIPLIED, 0x8000u);
NX_CONST(uint, NX_TEXTURE_SAMPLER_MASK, 0x3FFFu);
