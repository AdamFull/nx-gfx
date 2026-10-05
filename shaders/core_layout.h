#pragma once

// Shared descriptor-space ABI. Keep these values in the C++/Slang layout
// header so pipeline layouts and shader declarations cannot drift apart.
#define NX_BINDLESS_SET 0u
#define NX_BINDLESS_BINDING_SAMPLERS 0u
#define NX_BINDLESS_BINDING_TEXTURES 1u
#define NX_BINDLESS_BINDING_RW_TEXTURES 2u

// These are ABI capacities, not per-device advertised limits. Metal lays the
// argument-buffer arrays out at these exact sizes; Vulkan clamps the number of
// descriptors populated at run time.
#define NX_BINDLESS_MAX_SAMPLERS 1024u
#define NX_BINDLESS_MAX_TEXTURES 65535u
#define NX_BINDLESS_MAX_RW_TEXTURES 8192u

// Shared C++/Slang layout for the compact sampled-texture handle. The upper
// half identifies the bindless texture and the lower half identifies its
// sampler plus sampling metadata.
#define NX_TEXTURE_NONE 0xFFFFu
#define NX_TEXTURE_LINEAR 0x4000u
#define NX_TEXTURE_PREMULTIPLIED 0x8000u
#define NX_TEXTURE_SAMPLER_MASK 0x3FFFu

// An image resolver's second word: an atlas region rather than a texture,
// the atlas's index above and the region's below.
#define NX_IMAGE_ATLAS 0x80000000u
#define NX_IMAGE_ATLAS_INDEX_MASK 0x7FFFu
#define NX_IMAGE_REGION_MASK 0xFFFFu
