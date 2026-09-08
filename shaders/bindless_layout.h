#pragma once

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
