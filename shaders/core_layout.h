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

// Compact sampled-texture ABI shared by CPU push/structured-buffer data and
// shaders. The descriptor index occupies the upper 16 bits; the lower half
// stores the sampler index and sampling flags above. Slang supplies the
// methods in core.slang; C++ only needs the identical one-word representation.
#ifdef __cplusplus
#include <cstdint>

template <typename T> struct NxTexture2D {
  std::uint32_t value;
};

template <typename T> struct NxRWTexture2D {
  std::uint32_t index;
};

template <typename T>
[[nodiscard]] constexpr NxTexture2D<T>
nx_texture_2d(const std::uint32_t value) noexcept {
  return {value};
}

template <typename T>
[[nodiscard]] constexpr NxTexture2D<T>
nx_texture_2d(const std::uint32_t texture, const std::uint32_t sampler,
              const bool premultiplied = false,
              const bool linear = false) noexcept {
  const std::uint32_t index =
      texture >= NX_TEXTURE_NONE ? NX_TEXTURE_NONE : texture;
  return {(index << 16u) | (sampler & NX_TEXTURE_SAMPLER_MASK) |
          (premultiplied ? NX_TEXTURE_PREMULTIPLIED : 0u) |
          (linear ? NX_TEXTURE_LINEAR : 0u)};
}

template <typename T>
[[nodiscard]] constexpr std::uint32_t
nx_texture_value(const NxTexture2D<T> texture) noexcept {
  return texture.value;
}

template <typename T>
[[nodiscard]] constexpr NxRWTexture2D<T>
nx_rw_texture_2d(const std::uint32_t index) noexcept {
  return {index};
}

template <typename T>
[[nodiscard]] constexpr std::uint32_t
nx_rw_texture_index(const NxRWTexture2D<T> texture) noexcept {
  return texture.index;
}

static_assert(sizeof(NxTexture2D<std::uint32_t>) == sizeof(std::uint32_t));
static_assert(sizeof(NxRWTexture2D<std::uint32_t>) == sizeof(std::uint32_t));
#endif
