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
