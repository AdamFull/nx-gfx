#pragma once

// The words the texture handles are made of (nx_texture.hpp,
// nx_texture.slang), shared by C++ and Slang; Slang reads them through the
// nx_texture_layout module, which nx_core re-exports.

#include "rendering/rhi/shaders/nx_shared.h"

// A sampled texture's word: its bindless slot in the upper half, its sampler
// and sampling flags in the lower.
NX_CONST(uint, NX_TEXTURE_NONE, 0xFFFFu);
NX_CONST(uint, NX_TEXTURE_LINEAR, 0x4000u);
NX_CONST(uint, NX_TEXTURE_PREMULTIPLIED, 0x8000u);
NX_CONST(uint, NX_TEXTURE_SAMPLER_MASK, 0x3FFFu);

// An image resolver's second word: an atlas region rather than a texture,
// the atlas's index above and the region's below.
NX_CONST(uint, NX_IMAGE_ATLAS, 0x80000000u);
NX_CONST(uint, NX_IMAGE_ATLAS_INDEX_MASK, 0x7FFFu);
NX_CONST(uint, NX_IMAGE_REGION_MASK, 0xFFFFu);
