#pragma once

/**
 * @file types.h
 * @brief Portable RHI enums and flags (namespace nxe::rhi).
 *
 */

#include "core/foundation/core/enum.h"

namespace nxe::rhi {

enum class Format : u16 {
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

  // Packed
  RGB10A2_UNORM,
  RG11B10_FLOAT,

  // Depth / stencil
  D16_UNORM,
  D32_FLOAT,
  D24_UNORM_S8_UINT,
  D32_FLOAT_S8_UINT,

  // Block compressed - desktop
  BC1_RGBA_UNORM,
  BC1_RGBA_SRGB,
  BC3_RGBA_UNORM,
  BC3_RGBA_SRGB,
  BC4_R_UNORM,
  BC5_RG_UNORM,
  BC7_RGBA_UNORM,
  BC7_RGBA_SRGB,

  // Block compressed - mobile
  ASTC_4x4_UNORM,
  ASTC_4x4_SRGB,
  ASTC_6x6_UNORM,
  ASTC_6x6_SRGB,
  ASTC_8x8_UNORM,
  ASTC_8x8_SRGB,

  // Extra block-compressed formats DDS/KTX assets commonly carry. Appended
  // rather than grouped so no existing enumerator changes value.
  BC1_RGB_UNORM, ///< DXT1 without the 1-bit alpha.
  BC1_RGB_SRGB,
  BC2_RGBA_UNORM, ///< DXT3 (explicit alpha).
  BC2_RGBA_SRGB,
  BC6H_RGB_UFLOAT, ///< HDR, unsigned.
  BC6H_RGB_SFLOAT, ///< HDR, signed.

  Count
};

/// Whether barrier() advances the RHI's tracked texture states. Skip it when
/// the caller owns the timeline and may be recording on several threads.
enum class StateTracking : u8 { Update, Skip };

enum class ResourceState : u16 {
  Undefined = 0, ///< Contents are garbage; only valid as a barrier source.
  General,       ///< Any access; the catch-all when nothing tighter applies.

  CopySrc,
  CopyDst,

  ShaderReadOnly,  ///< Sampled, or a read-only storage resource.
  ShaderWriteOnly, ///< Write-only storage image / buffer.
  ShaderReadWrite, ///< Read-write storage image / buffer (UAV).

  ColorAttachment, ///< Read and write (blending).
  ColorAttachmentRead,
  ColorAttachmentWrite,

  DepthStencilRead, ///< Depth test with writes disabled; sampleable.
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

enum class TextureUsage : u16 {
  None = 0,
  Sampled = 1 << 0,      ///< Readable from shaders.
  Storage = 1 << 1,      ///< Read/write from shaders (RWTexture).
  RenderTarget = 1 << 2, ///< Colour attachment.
  DepthStencil = 1 << 3, ///< Depth/stencil attachment.
  CopySrc = 1 << 4,
  CopyDst = 1 << 5,
  /// Attachment that never leaves tile memory. Free on mobile tilers, ignored
  /// elsewhere. Cannot be combined with Sampled or CopySrc.
  Transient = 1 << 6,
};
NX_ENUM_FLAGS_DEFINE(TextureUsage, TextureUsageFlags)

enum class BufferUsage : u16 {
  None = 0,
  Vertex = 1 << 0,
  Index = 1 << 1,
  Uniform = 1 << 2,
  Storage = 1 << 3,
  Indirect = 1 << 4,
  CopySrc = 1 << 5,
  CopyDst = 1 << 6,
  /// Expose a GPU virtual address, so shaders can hold a pointer to it. This is
  /// how the bindless path reaches buffer data.
  DeviceAddress = 1 << 7,
};
NX_ENUM_FLAGS_DEFINE(BufferUsage, BufferUsageFlags)

enum class MemoryUsage : u8 {
  /// VRAM. Fastest for the GPU, not host-visible on discrete cards.
  DeviceLocal,
  /// Host-visible, write-combined. For per-frame data the CPU produces.
  Upload,
  /// Host-visible, cached. For data the CPU reads back.
  Readback,
  Count
};

enum class TextureDimension : u8 { Tex2D, Tex2DArray, TexCube, Tex3D, Count };

enum class LoadOp : u8 {
  Load,     ///< Preserve existing contents (costly on tilers).
  Clear,    ///< Clear to the supplied value.
  DontCare, ///< Contents undefined; cheapest.
  Count
};

enum class StoreOp : u8 {
  Store,
  DontCare, ///< Discard after the pass; required for transient attachments.
  Count
};

enum class FilterMode : u8 { Nearest, Linear, Count };

enum class AddressMode : u8 {
  Repeat,
  MirroredRepeat,
  ClampToEdge,
  ClampToBorder,
  Count
};

enum class CompareOp : u8 {
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

enum class ShaderStage : u8 { Vertex, Fragment, Compute, Count };

enum class PrimitiveTopology : u8 {
  TriangleList,
  TriangleStrip,
  LineList,
  LineStrip,
  PointList,
  Count
};

/// Index buffer element width. 16-bit halves the bandwidth and is enough for
/// any single 2D batch, so it is the default everywhere it can be.
enum class IndexType : u8 { U16, U32, Count };

enum class CullMode : u8 { None, Front, Back, Count };

enum class FrontFace : u8 { CounterClockwise, Clockwise, Count };

/// Common blend presets, so the renderer names intent instead of spelling out
/// src/dst factors. 2D wants little more than these.
enum class BlendMode : u8 {
  Opaque,        ///< No blending.
  AlphaBlend,    ///< src.a over dst (straight alpha).
  Premultiplied, ///< src over dst, src colour already alpha-weighted.
  Additive,      ///< src.a * src + dst (straight alpha).
  PremultipliedAdditive,
  Multiply,
  Screen,
  Count
};

enum class QueueType : u8 {
  Graphics, ///< Also valid for compute and copy.
  Compute,  ///< Async compute where the device exposes it.
  Copy,     ///< DMA / transfer-only queue for streaming.
  Count
};

/// Presentation pacing.
enum class PresentMode : u8 {
  /// Wait for vblank, queue depth 1. Always supported; no tearing.
  Fifo,
  /// Wait for vblank but replace a queued frame if we beat it. No tearing,
  /// lower latency than Fifo. Falls back to Fifo when unsupported.
  Mailbox,
  /// No wait, may tear. Falls back to Fifo when unsupported.
  Immediate,
  Count
};

enum class AcquireResult : u8 {
  Success,
  /// Acquired, but the swapchain no longer matches the surface. The frame is
  /// still renderable; recreate afterwards.
  Suboptimal,
  /// Nothing acquired. Recreate the swapchain and skip this frame.
  OutOfDate,
  /// Nothing acquired and nothing to render into (minimized, or the OS took
  /// the surface away). Skip the frame entirely.
  SurfaceLost,
  Count
};

enum class VideoCodec : u8 { VP9, AV1, Count };

struct VideoDecodeCaps {
  bool supported = false;
  u32 max_width = 0;
  u32 max_height = 0;
  u32 max_dpb_slots = 0;
  bool dpb_and_output_coincide = false;
};

[[nodiscard]] nx::string_view to_string(Format format) noexcept;
/// The format @p name spells, case-insensitively, or Unknown. The names are
/// to_string's, so a config and a log say the same thing.
[[nodiscard]] Format format_from_string(nx::string_view name) noexcept;
/// One flag by name ("sampled", "render_target"), or None. Callers combine.
[[nodiscard]] TextureUsage texture_usage_from_string(nx::string_view name) noexcept;
[[nodiscard]] BufferUsage buffer_usage_from_string(nx::string_view name) noexcept;
[[nodiscard]] nx::string_view to_string(ResourceState state) noexcept;
[[nodiscard]] nx::string_view to_string(QueueType queue) noexcept;

/// Bytes per pixel for uncompressed formats, or bytes per block for compressed
/// ones. Returns 0 for Format::Unknown.
[[nodiscard]] u32 format_texel_size(Format format) noexcept;
/// Block footprint in texels; {1, 1} for uncompressed formats.
void format_block_extent(Format format, u32 &width, u32 &height) noexcept;
[[nodiscard]] bool format_is_depth(Format format) noexcept;
[[nodiscard]] bool format_is_stencil(Format format) noexcept;
[[nodiscard]] bool format_is_srgb(Format format) noexcept;
/// True for block-compressed formats (BC*, ASTC*).
[[nodiscard]] bool format_is_compressed(Format format) noexcept;

[[nodiscard]] bool format_is_float(Format format) noexcept;

/// Bytes in one row of blocks for a mip of the given width. Compressed rows are
/// counted in whole blocks, so the width is rounded up to the block size.
[[nodiscard]] u64 format_row_pitch(Format format, u32 width) noexcept;
/// Tightly-packed byte size of one mip subresource (one array layer / one 3D
/// slice stack). Rounds up to whole blocks on each axis.
[[nodiscard]] u64 format_subresource_size(Format format, u32 width, u32 height,
                                          u32 depth = 1) noexcept;

} // namespace nxe::rhi
