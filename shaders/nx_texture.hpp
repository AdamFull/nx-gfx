#pragma once

// The bindless texture handles, the C++ half of the pair whose Slang half is
// nx_texture.slang: one word each, made and read with the same names as in
// Slang. C++ carries them in shared structs and hands them to shaders; only
// shaders sample.

#include "rendering/rhi/shaders/core_layout.h"

#include <cstdint>

namespace nx_texture_shape {
struct Tex2D;
struct Cube;
struct Array2D;
struct Tex3D;
} // namespace nx_texture_shape

/// A sampled texture of @p Shape holding T texels: its bindless slot
/// (NX_TEXTURE_NONE for none) in the upper 16 bits, its sampler and the
/// NX_TEXTURE_LINEAR and NX_TEXTURE_PREMULTIPLIED flags in the lower.
template <typename Shape, typename T> struct NxSampledTexture {
  std::uint32_t value;

  [[nodiscard]] static constexpr NxSampledTexture
  from_packed(const std::uint32_t packed) noexcept {
    return {packed};
  }

  [[nodiscard]] static constexpr NxSampledTexture
  from_indices(const std::uint32_t texture, const std::uint32_t sampler,
               const std::uint32_t flags = 0) noexcept {
    const std::uint32_t slot =
        texture < NX_TEXTURE_NONE ? texture : NX_TEXTURE_NONE;
    return {(slot << 16u) | (sampler & NX_TEXTURE_SAMPLER_MASK) |
            (flags & (NX_TEXTURE_LINEAR | NX_TEXTURE_PREMULTIPLIED))};
  }

  [[nodiscard]] static constexpr NxSampledTexture none() noexcept {
    return from_indices(NX_TEXTURE_NONE, 0);
  }

  [[nodiscard]] constexpr std::uint32_t texture_index() const noexcept {
    return value >> 16u;
  }
  [[nodiscard]] constexpr std::uint32_t sampler_index() const noexcept {
    return value & NX_TEXTURE_SAMPLER_MASK;
  }
  [[nodiscard]] constexpr bool is_linear() const noexcept {
    return (value & NX_TEXTURE_LINEAR) != 0;
  }
  [[nodiscard]] constexpr bool is_premultiplied() const noexcept {
    return (value & NX_TEXTURE_PREMULTIPLIED) != 0;
  }
  [[nodiscard]] constexpr bool is_valid() const noexcept {
    return texture_index() != NX_TEXTURE_NONE;
  }

  [[nodiscard]] friend constexpr bool
  operator==(NxSampledTexture, NxSampledTexture) noexcept = default;
};

/// A storage texture of @p Shape holding T texels: its storage slot.
template <typename Shape, typename T> struct NxStorageTexture {
  std::uint32_t index;

  [[nodiscard]] static constexpr NxStorageTexture
  from_index(const std::uint32_t index) noexcept {
    return {index};
  }

  [[nodiscard]] constexpr bool is_valid() const noexcept {
    return index != 0xFFFFFFFFu;
  }

  [[nodiscard]] friend constexpr bool
  operator==(NxStorageTexture, NxStorageTexture) noexcept = default;
};

template <typename T>
using NxTexture2D = NxSampledTexture<nx_texture_shape::Tex2D, T>;
template <typename T>
using NxTextureCube = NxSampledTexture<nx_texture_shape::Cube, T>;
template <typename T>
using NxTexture2DArray = NxSampledTexture<nx_texture_shape::Array2D, T>;
template <typename T>
using NxTexture3D = NxSampledTexture<nx_texture_shape::Tex3D, T>;
template <typename T>
using NxRWTexture2D = NxStorageTexture<nx_texture_shape::Tex2D, T>;
/// A storage 2D array, or a cube's faces as one.
template <typename T>
using NxRWTexture2DArray = NxStorageTexture<nx_texture_shape::Array2D, T>;
template <typename T>
using NxRWTexture3D = NxStorageTexture<nx_texture_shape::Tex3D, T>;

/// A 2D image in two words: a texture, or a region of an atlas with a
/// texture to fall back on.
template <typename T> struct NxImage2DResolver {
  NxTexture2D<T> texture;
  std::uint32_t metadata;

  [[nodiscard]] static constexpr NxImage2DResolver
  from_texture(const NxTexture2D<T> texture) noexcept {
    return {texture, 0u};
  }

  [[nodiscard]] static constexpr NxImage2DResolver
  from_atlas(const NxTexture2D<T> fallback, const std::uint32_t atlas,
             const std::uint32_t region) noexcept {
    return {fallback, NX_IMAGE_ATLAS |
                          ((atlas & NX_IMAGE_ATLAS_INDEX_MASK) << 16u) |
                          (region & NX_IMAGE_REGION_MASK)};
  }

  [[nodiscard]] constexpr bool is_atlas() const noexcept {
    return (metadata & NX_IMAGE_ATLAS) != 0;
  }
  [[nodiscard]] constexpr NxTexture2D<T> as_texture() const noexcept {
    return texture;
  }

  [[nodiscard]] friend constexpr bool
  operator==(NxImage2DResolver, NxImage2DResolver) noexcept = default;
};

static_assert(sizeof(NxTexture2D<std::uint32_t>) == 4 &&
                  sizeof(NxRWTexture2D<std::uint32_t>) == 4 &&
                  sizeof(NxImage2DResolver<std::uint32_t>) == 8,
              "texture handles keep their word sizes on both sides");
