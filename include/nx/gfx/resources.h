#pragma once

#include "nx/gfx/types.h"

namespace nx::gfx {

struct texture_desc {
  nx::string_view name = "texture";
  gfx::format format = gfx::format::Unknown;
  texture_dimension dimension = texture_dimension::Tex2D;
  u32 width = 1;
  u32 height = 1;
  u32 depth_or_layers = 1;
  u32 mip_levels = 1;
  u32 sample_count = 1;
  texture_usage_flags usage = texture_usage::Sampled;
  memory_usage memory = memory_usage::DeviceLocal;
  resource_state initial_state = resource_state::Undefined;
  /// First mip exposed by the sampled view. Streaming clients use this to
  /// create a texture with only its already-uploaded coarse tail visible.
  u32 sampled_base_mip = 0;
};

/// What a resource needs from a heap. Query it before packing, then place the
/// resource at an offset that satisfies the alignment.
struct memory_requirements {
  u64 size = 0;
  u64 alignment = 1;
  u32 memory_type_bits = ~0u;
};

/// Memory the caller sub-allocates itself. Resources placed at non-overlapping
/// offsets are independent; resources placed at overlapping offsets alias, and
/// must not be in use at the same time.
struct heap_desc {
  nx::string_view name = "heap";
  u64 size = 0;
  u64 alignment = 256;
  u32 memory_type_bits = ~0u;
  memory_usage memory = memory_usage::DeviceLocal;
};

struct buffer_desc {
  nx::string_view name = "buffer";
  u64 size = 0;
  buffer_usage_flags usage = buffer_usage::Storage;
  memory_usage memory = memory_usage::DeviceLocal;
  /// Keep the mapping alive for the buffer's lifetime. Upload-heap ring
  /// buffers want this; mapping per frame is pure overhead.
  bool persistently_mapped = false;
};

struct sampler_desc {
  nx::string_view name = "sampler";
  filter_mode min_filter = filter_mode::Linear;
  filter_mode mag_filter = filter_mode::Linear;
  filter_mode mip_filter = filter_mode::Linear;
  address_mode address_u = address_mode::ClampToEdge;
  address_mode address_v = address_mode::ClampToEdge;
  address_mode address_w = address_mode::ClampToEdge;
  f32 mip_lod_bias = 0.f;
  f32 max_anisotropy = 1.f;
  f32 min_lod = 0.f;
  f32 max_lod = 1000.f;
  compare_op compare = compare_op::Never;
};

/// The sampler at bindless slot DEFAULT_SAMPLER: sampler_desc's defaults,
/// bilinear and clamped to the edge.
inline constexpr sampler_desc DEFAULT_SAMPLER_DESC{.name = "default sampler"};

} // namespace nx::gfx
