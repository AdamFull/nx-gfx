#pragma once

#include "core/foundation/core/enum.h"
#include "core/foundation/core/foundation.h"
#include "core/foundation/core/handle.h"

namespace nx::gfx {

enum class format : u16 {
  Unknown = 0,

  // 8-bit
  R8_UNORM,
  R8_SNORM,
  R8_UINT,
  R8_SINT,
  RG8_UNORM,
  RG8_UINT,
  RGBA8_UNORM,
  RGBA8_SRGB,
  RGBA8_SNORM,
  RGBA8_UINT,
  BGRA8_UNORM,
  BGRA8_SRGB,

  // 16-bit
  R16_UNORM,
  R16_FLOAT,
  R16_UINT,
  R16_SINT,
  RG16_FLOAT,
  RG16_UNORM,
  RGBA16_FLOAT,
  RGBA16_UNORM,

  // 32-bit
  R32_FLOAT,
  R32_UINT,
  R32_SINT,
  RG32_FLOAT,
  RG32_UINT,
  RGBA32_FLOAT,
  RGBA32_UINT,

  RGB10A2_UNORM,
  RG11B10_FLOAT,

  D16_UNORM,
  D32_FLOAT,
  D24_UNORM_S8_UINT,
  D32_FLOAT_S8_UINT,

  BC1_RGBA_UNORM,
  BC1_RGBA_SRGB,
  BC3_RGBA_UNORM,
  BC3_RGBA_SRGB,
  BC4_R_UNORM,
  BC5_RG_UNORM,
  BC7_RGBA_UNORM,
  BC7_RGBA_SRGB,

  ASTC_4x4_UNORM,
  ASTC_4x4_SRGB,
  ASTC_6x6_UNORM,
  ASTC_6x6_SRGB,
  ASTC_8x8_UNORM,
  ASTC_8x8_SRGB,

  BC1_RGB_UNORM, ///< DXT1 without the 1-bit alpha.
  BC1_RGB_SRGB,
  BC2_RGBA_UNORM,
  BC2_RGBA_SRGB,
  BC6H_RGB_UFLOAT,
  BC6H_RGB_SFLOAT,

  Count
};

enum class resource_state : u16 {
  Undefined = 0, ///< Contents are garbage; only valid as a barrier source.
  General,

  CopySrc,
  CopyDst,

  ShaderReadOnly,  ///< Sampled, or a read-only storage resource.
  ShaderWriteOnly, ///< Write-only storage image / buffer.
  ShaderReadWrite,

  ColorAttachment,
  ColorAttachmentRead,
  ColorAttachmentWrite,

  DepthStencilRead,
  DepthStencilWrite,
  DepthStencilReadWrite,

  VertexBuffer,
  IndexBuffer,
  IndirectBuffer,
  UniformBuffer,

  HostRead,
  HostWrite,

  Present,
  Count
};

[[nodiscard]] constexpr bool
is_write_state(const resource_state state) noexcept {
  switch (state) {
  case resource_state::General:
  case resource_state::CopyDst:
  case resource_state::ShaderWriteOnly:
  case resource_state::ShaderReadWrite:
  case resource_state::ColorAttachment:
  case resource_state::ColorAttachmentWrite:
  case resource_state::DepthStencilWrite:
  case resource_state::DepthStencilReadWrite:
  case resource_state::HostWrite:
    return true;
  default:
    return false;
  }
}

enum class texture_usage : u16 {
  None = 0,
  Sampled = 1 << 0,
  Storage = 1 << 1,
  RenderTarget = 1 << 2,
  DepthStencil = 1 << 3,
  CopySrc = 1 << 4,
  CopyDst = 1 << 5,
  /// Attachment that never leaves tile memory. Free on mobile tilers, ignored
  /// elsewhere. Cannot be combined with Sampled or CopySrc.
  Transient = 1 << 6,
};
NX_ENUM_FLAGS_DEFINE(texture_usage, texture_usage_flags)

enum class buffer_usage : u16 {
  None = 0,
  Vertex = 1 << 0,
  Index = 1 << 1,
  Uniform = 1 << 2,
  Storage = 1 << 3,
  Indirect = 1 << 4,
  CopySrc = 1 << 5,
  CopyDst = 1 << 6,
  DeviceAddress = 1 << 7,
};
NX_ENUM_FLAGS_DEFINE(buffer_usage, buffer_usage_flags)

/// The classes of subgroup operation a compute shader may use.
enum class subgroup_op : u16 {
  None = 0,
  Basic = 1 << 0,
  Vote = 1 << 1,
  Arithmetic = 1 << 2,
  Ballot = 1 << 3,
  Shuffle = 1 << 4,
  ShuffleRelative = 1 << 5,
  Clustered = 1 << 6,
  Quad = 1 << 7,
};
NX_ENUM_FLAGS_DEFINE(subgroup_op, subgroup_op_flags)

enum class memory_usage : u8 { DeviceLocal, Upload, Readback, Count };

enum class texture_dimension : u8 { Tex2D, Tex2DArray, TexCube, Tex3D, Count };

enum class load_op : u8 { Load, Clear, DontCare, Count };

enum class store_op : u8 { Store, DontCare, Count };

enum class filter_mode : u8 { Nearest, Linear, Count };

enum class address_mode : u8 {
  Repeat,
  MirroredRepeat,
  ClampToEdge,
  ClampToBorder,
  Count
};

enum class compare_op : u8 {
  Never,
  Less,
  Equal,
  LessOrEqual,
  Greater,
  NotEqual,
  GreaterOrEqual,
  Always,
  Count
};

enum class shader_stage : u8 { Vertex, Fragment, Compute, Count };

enum class primitive_topology : u8 {
  TriangleList,
  TriangleStrip,
  LineList,
  LineStrip,
  PointList,
  Count
};

/// Index buffer element width. 16-bit halves the bandwidth and is enough for
/// any single 2D batch, so it is the default everywhere it can be.
enum class index_type : u8 { U16, U32, Count };

enum class cull_mode : u8 { None, Front, Back, Count };

enum class front_face : u8 { CounterClockwise, Clockwise, Count };

enum class blend_mode : u8 {
  Opaque,
  AlphaBlend,
  Premultiplied,
  Additive,
  PremultipliedAdditive,
  Multiply,
  Screen,
  Count
};

enum class queue_type : u8 {
  Graphics, ///< Also valid for compute and copy.
  Compute,
  Copy,
  Count
};

enum class present_mode : u8 {
  Fifo,
  /// Wait for vblank but replace a queued frame if we beat it. No tearing,
  /// lower latency than Fifo. Falls back to Fifo when unsupported.
  Mailbox,
  /// No wait, may tear. Falls back to Fifo when unsupported.
  Immediate,
  Count
};

/// How the window system combines the backbuffer with what is behind the
/// window. Blending writes premultiplied colour, so a see-through window asks
/// for Premultiplied.
enum class surface_alpha : u8 {
  Opaque,
  Premultiplied,
  /// The compositor multiplies colour by alpha itself.
  Unmultiplied,
  /// The surface follows the window's own composition setting.
  WindowSystem,
  Count
};

enum class acquire_result : u8 {
  Success,
  Suboptimal,
  OutOfDate,
  SurfaceLost,
  Count
};

[[nodiscard]] nx::string_view to_string(format fmt) noexcept;
[[nodiscard]] format format_from_string(nx::string_view name) noexcept;
[[nodiscard]] texture_usage
texture_usage_from_string(nx::string_view name) noexcept;
[[nodiscard]] buffer_usage
buffer_usage_from_string(nx::string_view name) noexcept;
[[nodiscard]] nx::string_view to_string(resource_state state) noexcept;
[[nodiscard]] nx::string_view to_string(queue_type queue) noexcept;
[[nodiscard]] nx::string_view to_string(surface_alpha alpha) noexcept;

/// Bytes per pixel for uncompressed formats, or bytes per block for compressed
/// ones. Returns 0 for format::Unknown.
[[nodiscard]] u32 format_texel_size(format fmt) noexcept;
void format_block_extent(format fmt, u32 &width, u32 &height) noexcept;
[[nodiscard]] bool format_is_depth(format fmt) noexcept;
[[nodiscard]] bool format_is_stencil(format fmt) noexcept;
[[nodiscard]] bool format_is_srgb(format fmt) noexcept;
[[nodiscard]] bool format_is_compressed(format fmt) noexcept;
/// The channels a texel of @p fmt holds, as a shader reads them: 1 for R
/// and depth formats, 2 for RG (BC5 among them), 3 for RGB, else 4; 0 for
/// format::Unknown.
[[nodiscard]] u32 format_channels(format fmt) noexcept;

[[nodiscard]] bool format_is_float(format fmt) noexcept;

/// Bytes in one row of blocks for a mip of the given width. Compressed rows are
/// counted in whole blocks, so the width is rounded up to the block size.
[[nodiscard]] u64 format_row_pitch(format fmt, u32 width) noexcept;
/// Tightly-packed byte size of one mip subresource (one array layer / one 3D
/// slice stack). Rounds up to whole blocks on each axis.
[[nodiscard]] u64 format_subresource_size(format fmt, u32 width, u32 height,
                                          u32 depth = 1) noexcept;

// Textures and buffers use 32 index bits: a streaming atlas set churns slots
// fast enough that handle32's 8 version bits would wrap in a long session.
using texture_handle = nx::basic_handle<u64, 32, struct texture_tag>;
using buffer_handle = nx::basic_handle<u64, 32, struct buffer_tag>;

// These are created once and live for the process, so 24 index / 8 version bits
// are ample.

using heap_handle = nx::basic_handle<u32, 24, struct heap_tag>;
using sampler_handle = nx::basic_handle<u32, 24, struct sampler_tag>;
using pipeline_handle = nx::basic_handle<u32, 24, struct pipeline_tag>;
using shader_handle = nx::basic_handle<u32, 24, struct shader_tag>;
using query_pool_handle = nx::basic_handle<u32, 24, struct query_pool_tag>;
using swapchain_handle = nx::basic_handle<u32, 24, struct swapchain_tag>;

template <typename Repr, usize IndexBits, typename Tag>
[[nodiscard]] constexpr u32
bindless_index(const nx::basic_handle<Repr, IndexBits, Tag> handle) noexcept {
  return nx::cast<u32>(handle.index());
}

struct extent2d {
  u32 width = 0;
  u32 height = 0;
};

struct extent3d {
  u32 width = 0;
  u32 height = 0;
  u32 depth = 1;
};

struct offset2d {
  i32 x = 0;
  i32 y = 0;
};

struct rect2d {
  offset2d offset;
  extent2d extent;
};

struct viewport {
  f32 x = 0.f, y = 0.f;
  f32 width = 0.f, height = 0.f;
  f32 min_depth = 0.f, max_depth = 1.f;
};

union clear_value {
  f32 color[4];
  struct {
    f32 depth;
    u32 stencil;
  } depth_stencil;
};

[[nodiscard]] constexpr clear_value
clear_color(const f32 r, const f32 g, const f32 b, const f32 a = 1.f) noexcept {
  clear_value v{};
  v.color[0] = r;
  v.color[1] = g;
  v.color[2] = b;
  v.color[3] = a;
  return v;
}

[[nodiscard]] constexpr clear_value
clear_depth(const f32 depth = 1.f, const u32 stencil = 0) noexcept {
  clear_value v{};
  v.depth_stencil.depth = depth;
  v.depth_stencil.stencil = stencil;
  return v;
}

inline constexpr u32 MAX_COLOR_ATTACHMENTS = 8;

/// A version as the graphics API packs it (Vulkan's VK_MAKE_VERSION layout).
[[nodiscard]] constexpr u32 make_version(const u32 major, const u32 minor,
                                         const u32 patch) noexcept {
  return (major << 22u) | (minor << 12u) | patch;
}

} // namespace nx::gfx
